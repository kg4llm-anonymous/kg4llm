import json
from bleu import bleu
from checker import RuntimeChecker


def computeMaps(predictionfile, goldfile):
    predictionMap = {}
    goldMap = {}
    gf = open(goldfile, 'r')
    pf = open(predictionfile, 'r')

    for row in pf:
        cols = row.strip().split('\t')
        if len(cols) == 1:
            (rid, pred) = (cols[0], '') 
        else:
            (rid, pred) = (cols[0], cols[1])
        predictionMap[rid] = [pred.strip()]

    for row in gf:
        (rid, pred) = row.split('\t') 
        if rid in predictionMap: # Only insert if the id exists for the method
            if rid not in goldMap:
                goldMap[rid] = []
        goldMap[rid].append(pred.strip())

    return (goldMap, predictionMap)


#m1 is the reference map
#m2 is the prediction map
def SynBleuFromMaps(m1, m2, eval_examples, chr):
    score = [0] * 5
    num = 0.0
    for key in m1:
        if key in m2:
            if not chr.run(eval_examples[int(key)].source, m2[key][0]):
                bl = [0] * 5
            else:
                bl = bleu(m1[key], m2[key][0])
            score = [score[i] + bl[i] for i in range(0, len(bl))]
            num += 1
    return [s * 100.0 / num for s in score]


#m1 is the reference map
#m2 is the prediction map
def TopNSynBleuFromMaps(m1, m2, eval_examples, chr, topn_list=[1,5,10]):
    score = {k:0 for k in topn_list}
    num = 0.0
    for key in m1:
        if key in m2:
            bls = []
            for j in range(len(m2[key])):
                if not chr.run(eval_examples[int(key)].source, m2[key][0]):
                    bl = 0
                else:
                    bl = bleu(m1[key], m2[key][j])[0]
                bls.append(bl)
            for k in topn_list:
                score[k] += max(bls[:min(k, len(bls))])
            num += 1
    return {k:score[k]/num*100 for k in topn_list}

# def IOUFromMaps(m1, m2):
#     score = 0.0
#     num = 0.0
#     for key in m1:
#          if key in m2:
#             rmrt1 = ' '.join([api.split('(')[0] for api in m1[key][0].split()])
#             rmrt2 = ' '.join([api.split('(')[0] for api in m2[key][0].split()])
#             if rmrt1 == rmrt2:
#                 score += 1
#             # s1 = set(rmrt1)
#             # s2 = set(rmrt2)
#             # s1 = set(m1[key][0].split())
#             # s2 = set(m2[key][0].split())
#             # score += len(s1 & s2)/len(s2 | s1)
#             num += 1
#     return score/num


if __name__ == '__main__':
    class Example(object):
        """A single training/test example."""
        def __init__(self, source):
            self.source = source

    examples = []
    with open("./codeBERT/code2nl/data/nl2api/test.top1.ver9.jsonl", 'r', encoding='utf-8') as fdoc:
        for i, doc_l in enumerate(fdoc.readlines()):
            doc_l = json.loads(doc_l.strip())
            examples.append(Example(source=doc_l['doc']))

    # idx = 'org'
    chr = RuntimeChecker(path='./codeBERT/code2nl/ent_dict.json')
    idx = 'rate1=1+rate2=1e-1'
    (goldMap, predictionMap) = computeMaps("./codeBERT/code2nl/model/test_{}.output".format(idx), "./codeBERT/code2nl/model/test.gold")
    dev_bleu = round(SynBleuFromMaps(goldMap, predictionMap, examples, chr)[0], 2)
    print(dev_bleu)

    # data_enhance=DE, rule_contrain=RC
    # ORG  : BLEU-4 52.08 | radr 76.48 | synBLEU-4 42.13
    # DE   : BLEU-4 54.07 | radr 82.20 | synBLEU-4 46.37
    # DE+RC: BLEU-4 55.75 | radr 91.31 | synBLEU-4 52.23

    # org 30.32
    # our 30.39