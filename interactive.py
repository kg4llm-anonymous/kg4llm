from __future__ import absolute_import
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import math
import time
import bleu
from syn_bleu import TopNSynBleuFromMaps
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from my_model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

if not os.path.exists('./model'):
    os.makedirs('./model')
logging.basicConfig(filename = './model/output.log',
                    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


from checker import RuntimeChecker


class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx  
            apiseq=js['apiseq'].strip()
            nl=js['doc'].strip()

            examples.append(
                Example(
                        idx=idx,
                        source=nl,
                        target=apiseq,
                        ) 
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       
        

def convert_examples_to_features(examples, tokenizer, args, stage=None):
    # logger.info(f"loading examples ...")
    n = 0
    features = []
    example_index = 0
    for example in examples:
        # ----------------------------------------------------------------
        # target
        target_tokens = tokenizer.tokenize(example.target)
        if len(target_tokens) >= args.max_target_length - 2:
            target_tokens = target_tokens[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length
        # ----------------------------------------------------------------
        # source
        source_tokens = tokenizer.tokenize(example.source)
        if len(source_tokens) >= args.max_source_length-2:
            n += 1
            source_tokens = source_tokens[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
        # ----------------------------------------------------------------
        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
        example_index += 1

    # logger.info(f"finished loading")
    return features, examples


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def metrics(chr, eval_examples, goldMap, predictionMap, topn_list=[1,5,10]):
    dev_topn_bleu, dev_topn_radr, dev_topn_synbleu = None, None, None
    # cal BLEU rate
    dev_topn_bleu = bleu.topNBleuFromMaps(goldMap, predictionMap, topn_list)
    for k, v in dev_topn_bleu.items():
        logger.info('  top-{} bleu-4 = {} '.format(k, v))
    logger.info("  "+"*"*20)
    # cal legal rate
    dev_topn_radr = chr.cal_legal_rate(eval_examples, predictionMap, topn_list)
    for k, v in dev_topn_radr.items():
        logger.info('  top-{} radr = {} '.format(k, v))
    logger.info("  "+"*"*20)
    # cal topn synBLEU
    dev_topn_synbleu = TopNSynBleuFromMaps(goldMap, predictionMap, eval_examples, chr, topn_list)
    for k, v in dev_topn_synbleu.items():
        logger.info('  top-{} Synbleu-4 = {} '.format(k, v))
    logger.info("  "+"*"*20)
    # cal type1 legal rate
    dev_topn_radr = chr.cal_type1_legal_rate(predictionMap, topn_list)
    for k, v in dev_topn_radr.items():
        logger.info('  top-{} radr(t1) = {} '.format(k, v))
    logger.info("  "+"*"*20)
    # cal type2 legal rate
    dev_topn_radr = chr.cal_type2_legal_rate(eval_examples, predictionMap, topn_list)
    for k, v in dev_topn_radr.items():
        logger.info('  top-{} radr(t2) = {} '.format(k, v))
    logger.info("  "+"*"*20)
    # cal type3 legal rate
    dev_topn_radr = chr.cal_type3_legal_rate(eval_examples, predictionMap, topn_list)
    for k, v in dev_topn_radr.items():
        logger.info('  top-{} radr(t3) = {} '.format(k, v))
    logger.info("  "+"*"*20)


    return dev_topn_bleu, dev_topn_radr, dev_topn_synbleu
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default="./model", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    
    parser.add_argument("--config_name", default="microsoft/codebert-base", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=96, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    chr = RuntimeChecker(tokenizer=tokenizer)
    
    #build model
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config, tokenizer=tokenizer,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)
    while True:
        desc = input('Please input your functional description: ')
        param = input('Please input your interface parameter types (add a space between each parameter type): ')
        doc = desc + ' </s> ' + param

        eval_example = Example(idx=0, source=doc, target='void') 
        eval_features, eval_examples = convert_examples_to_features([eval_example], tokenizer, args, stage='test')

        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
        eval_data = TensorDataset(all_source_ids,all_source_mask)  

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)
        model.eval()
        p = []
        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask= batch                  
            with torch.no_grad():
                docs = [eval_examples[i].source for i in range(len(p), len(p)+source_ids.shape[0])]
                preds = model(source_ids=source_ids, source_mask=source_mask, args={'docs': docs})
                for pred in preds:
                    text = [pred[i][1] for i in range(len(pred))]
                    p.append(text)
        for i, item in enumerate(p[0]):
            print(f'{i+1}: ' + item)
            print('')

if __name__ == "__main__":
    main()
