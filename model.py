import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import numpy as np
import time
from guider import RuntimeGuider
from checker import RuntimeChecker


rate = 0.1 # 合法API序列的权值



class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder, decoder, config, tokenizer, beam_size=None,max_length=None,sos_id=None,eos_id=None):
        super(Seq2Seq, self).__init__()
        self.gdr = RuntimeGuider(tokenizer)
        self.chr = RuntimeChecker(tokenizer)
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id

        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of whether we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        
    
    def forward(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, args=None):   
        if target_ids is not None:  
            outputs = self.encoder(source_ids, attention_mask=source_mask)
            encoder_output = outputs[0].permute([1,0,2]).contiguous()
            attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous()
            out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(1-source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss,loss*active_loss.sum(),active_loss.sum()
            return outputs
        else:
            #Predict 
            return self.generate_apiseq(source_ids, source_mask, target_ids, target_mask, args)
    
    def generate_apiseq(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, args=None):
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        preds=[]    
        for i in range(source_ids.shape[0]):
            context=encoder_output[:,i:i+1]
            context_mask=source_mask[i:i+1,:]
            beam = Beam(self.beam_size, self.sos_id, self.eos_id, self.chr)
            input_ids=beam.getCurrentState()
            context=context.repeat(1, self.beam_size,1)
            context_mask=context_mask.repeat(self.beam_size,1)

            for _ in range(self.max_length): 
                if beam.done():
                    break
                attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
                tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous()
                out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool())
                out = torch.tanh(self.dense(out))
                hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
                out = self.lsm(self.lm_head(hidden_states)).data
                if args is not None:
                    required_types = self.gdr.extract_required_api(args['docs'][i].strip().split())
                    beam_size, vocab_size = out.shape[0], out.shape[1]
                    rule_mask = torch.ones((beam_size, vocab_size)).cuda()*-1e12
                    # prior_mask = torch.ones((beam_size, vocab_size)).cuda()
                    for j in range(out.shape[0]):
                        cur_seq = input_ids[j].cpu().numpy().tolist()[1:]
                        candi = self.gdr.run(required_types, cur_seq)
                        for idx in candi: rule_mask[j][idx] = 0
                        # for idx in prior: prior_mask[j][idx] = rate1
                    out = (out + rule_mask) # * prior_mask
                beam.advance(out, args['docs'][i])
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)

            result = beam.getFinal(args['docs'][i])
            preds.append(result)            
        return preds
        

class Beam(object):
    def __init__(self, size, sos, eos, chr):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []
        self.chr = chr
        

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]
    
    def getAPISequence(self, timestep, k):
        hyp = []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k].item())
            k = self.prevKs[j][k]
        hyp = hyp[::-1]
        if self._eos in hyp: 
            hyp = hyp[:hyp.index(self._eos)]
        if 0 in hyp:
            hyp = hyp[:hyp.index(0)]
        hyp = self.chr.tokenizer.decode(hyp, clean_up_tokenization_spaces=False)
        return hyp

    def advance(self, wordLk, doc):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let self._ have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))
        
        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                apiseq = self.getAPISequence(len(self.nextYs) - 1, i)
                if (len(apiseq.split()) >= 3) and (not self.chr.check_api_name_error(apiseq.split())):
                    if self.chr.run(doc.strip(), apiseq): s *= rate
                    self.finished.append([s/(len(self.nextYs) - 1), apiseq])

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size
    
    def getFinal(self, doc):
        unfinished=[]
        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] != self._eos:
                s = self.scores[i]
                apiseq = self.getAPISequence(len(self.nextYs) - 1, i)
                if not self.chr.check_api_name_error(apiseq.split()):
                    if self.chr.run(doc.strip(), apiseq): s *= rate
                    unfinished.append([s/(len(self.nextYs) - 1), apiseq])
        self.finished += unfinished
        self.finished.sort(key=lambda a: -a[0])
        return self.finished[:min(self.size, len(self.finished))]
        
