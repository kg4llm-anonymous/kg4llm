import json
from queue import Queue
from copy import deepcopy


def load_json(path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_ceir_type(node):
    for tp in ['class', 'interface', 'enum', 'record']:
        if node.__contains__(tp):
            return tp


def contains(d:dict, k:str):
    if (not d.__contains__(k)) or (d[k] == 0):
        return False
    else:
        return True


def list_starswith_list(l1, l2) -> bool:
    if len(l2) > len(l1): return False
    for item1, item2 in zip(l1, l2):
        if item1 != item2: return False
    return True


class RuntimeChecker:
    def __init__(self, tokenizer, path=r'ent_dict.json', mode='no_return') -> None:
        assert mode in ['normal', 'no_return']
        self.tokenizer = tokenizer
        self.last_apis = []
        self.mode = mode
        self.preprocess(path)
    
    def get_api_name(self, node):
        param = node['parameter']
        param_n = node['parameter dim']
        rt = node['return']
        rt_n = node['return dim']
        site = {
            'package':node['package'],
            'ceir':node['ceir'],
            'method':node['method'],
            'parameter':','.join(['.'.join(p.split('$')[-2:]) + '[]'*n for p, n in zip(param, param_n)]),
            'return':'.'.join(rt.split('$')[-2:]) + '[]'*rt_n,
        }
        if self.mode == 'normal':
            name = '{package}.{ceir}.{method}({parameter})->{return}'.format(**site)
        elif self.mode == 'no_return':
            name = '{package}.{ceir}.{method}({parameter})'.format(**site)
        else:
            exit()
        name = name.replace('basictype.', '')
        return name
    
    def extract_required_api(self, doc:list):
        idx = None
        for i, word in enumerate(doc):
            if word == '</s>':
                idx = i + 1
                break
        assert idx is not None
        if idx == len(doc):
            doc = []
        else:
            doc = doc[idx:]
        if (len(doc) == 1) and (doc[0] == 'void'):
            doc = []
        return doc

    def preprocess(self, ent_dict_fname):
        # 声明
        api2node = dict() # 用于记录哪些api可以调用。包括所有的构造方法，所有的public static方法，以及static成员的方法（这里需要考虑强转）
        ceir2father = dict() # 统计类的父类，用于强制类型转换的判断
        default_types = set()
        ceir2hash = dict()
        # 基本数据类型
        default_types.add('basictype$byte')
        default_types.add('basictype$float')
        default_types.add('basictype$double')
        default_types.add('basictype$int')
        default_types.add('basictype$short')
        default_types.add('basictype$long')
        default_types.add('basictype$char')
        default_types.add('basictype$boolean')
        default_types.add('basictype$void')
        default_types.add('basictype$generics')
        for item in list(default_types): # 基本数据类型没有父类，也不会导致当前可用数据类型的变化
            ceir2father[item] = []
            ceir2hash[item.split('$')[-1]] = item
        # 支持封箱/拆箱机制的包装类
        default_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/String.html$java.lang$String')
        default_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Integer.html$java.lang$Integer')
        default_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Float.html$java.lang$Float')
        default_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Double.html$java.lang$Double')
        default_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Long.html$java.lang$Long')
        default_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Short.html$java.lang$Short')
        default_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Character.html$java.lang$Character')
        default_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Boolean.html$java.lang$Boolean')
        default_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Byte.html$java.lang$Byte')

        ent_dict = load_json(ent_dict_fname)
        for v in ent_dict.values():
            if v['type'] == 'method':
                ceir_type = get_ceir_type(v) + '$' + '$'.join(v['hash'].split('$')[1:-1])
                if (v['return'] == 'None') and ('abstract' in ent_dict[ceir_type]['field']): continue # 抽象类的构造函数不可调用
                
                parameter_list = [param[0] for param in v['parameter']]
                parameter_dim_list = [param[1] for param in v['parameter']]
                return_type = ceir_type if v['return'] == 'None' else v['return'][0]
                return_dim = 0  if v['return'] == 'None' else v['return'][1]

                require_ceir = None
                if v['return'] == 'None': # 非抽象类的构造函数调用没有限制
                    require_ceir = None
                elif ('static' in v['field']) and ('public' in v['field']): # 抽象类/非抽象类的静态方法调用没有限制
                    require_ceir = None
                else: # 其他方法的调用要求当前已经存在对象
                    require_ceir = ceir_type
                node = {
                    'package': v['package'],
                    'ceir': v[ent_dict[ceir_type]['type']],
                    'method': v['method'],
                    'require ceir':require_ceir,
                    'parameter':parameter_list,
                    'parameter dim':parameter_dim_list,
                    'return':return_type,
                    'return dim':return_dim,
                }
                name = self.get_api_name(node)
                node['token'] = self.tokenizer.tokenize(name)
                api2node[name] = node
            elif v['type'] == 'member': # 静态方法可以直接用，不需要声明  #  and (get_ceir_type(v) == 'enum')
                if  ('static' in v['field']) and ('public' in v['field']):
                    default_types.add(v['return'][0])
            elif v['type'] in ('class', 'interface', 'enum', 'record'): # 记录类到其父类集合（包含自己）的映射
                ceir2hash[v['package'] + '.' + v[v['type']]] = v['hash']
                ceir2father[v['hash']] = []
                q = Queue()
                q.put(v['hash'])
                while not q.empty():
                    cur_hash = q.get()
                    ceir2father[v['hash']].append(cur_hash)
                    for item in ent_dict[cur_hash]['extend'] + ent_dict[cur_hash]['implement']:
                        q.put(item)

        self.api2node = api2node
        self.default_types = default_types
        self.ceir2father = ceir2father
        self.ceir2hash = ceir2hash
    
    def run(self, doc:str, apiseq:str) -> bool:
        apiseq = apiseq.strip().split()
        doc = doc.strip().split()
        if len(apiseq) == 0: return False
        if self.check_api_name_error(apiseq):
            return False
        required_types = self.extract_required_api(doc)
        if self.check_syntex_error(apiseq, required_types, deepcopy(self.default_types)):
            return False
        return True
    
    def check_api_name_error(self, apiseq:list) -> bool:
        # RETURN: True - 当前API序列不合法 | False - 当前API序列合法
        self.last_apis = []
        for i, api in enumerate(apiseq):
            if i == len(apiseq)-1:
                if not self.api2node.__contains__(api):
                    last_api_tokens = self.tokenizer.tokenize(api)
                    for tgt_api in self.api2node.keys(): # 最后一个API没预测完，需要找出所有可能的API并保存
                        if list_starswith_list(self.api2node[tgt_api]['token'], last_api_tokens):
                            self.last_apis.append(tgt_api)
                    if len(self.last_apis) == 0:
                        return True
            else:
                if not self.api2node.__contains__(api):
                    return True
        return False
    
    def check_syntex_error(self, apiseq, required_types, default_types:set):
        meets_types = set()

        for ceir in required_types:
            ceir_hash = self.ceir2hash[ceir]
            default_types.add(ceir_hash)
        
        if len(self.last_apis) > 0:
            apiseq = apiseq[:-1]

        for api in apiseq:
            node_item = self.api2node[api]
            require_ceir = node_item['require ceir']
            param = node_item['parameter']
            rt = node_item['return']

            if require_ceir is not None:
                meets_types.add(require_ceir)
                if require_ceir not in default_types:
                    return True

            for p in param:
                meets_types.add(p)
                if p not in default_types:
                    return True
                
            for ceir in self.ceir2father[rt]:
                default_types.add(ceir)

        required_types = set([self.ceir2hash[tp] for tp in required_types])
        if not required_types.issubset(meets_types):
            return True
        
        if len(self.last_apis) > 0:
            flag = False
            for api in self.last_apis:
                node_item = self.api2node[api]
                require_ceir = node_item['require ceir']
                param = node_item['parameter']

                if require_ceir is not None:
                    if require_ceir not in default_types:
                        continue

                for p in param:
                    if p not in default_types:
                        continue
                flag = True
                break
            self.last_apis = []
            return not flag

        return False

    def cal_legal_rate(self, eval_examples, predictionMap, topn_list=[1,5,10]):
        legs = {k:0 for k in topn_list}
        nums = {k:0.0 for k in topn_list}
        for i, pre in predictionMap.items():
            doc = eval_examples[int(i)].source
            for j in range(len(pre)):
                leg = self.run(deepcopy(doc), pre[j])
                for k in legs.keys():
                    if j + 1 <= k:
                        legs[k] += (1 if leg else 0)
                        nums[k] += 1
        return {k:legs[k]/nums[k]*100 for k in topn_list}
    
    def cal_non_accept_count2(self, added_param, apiseq):
        apiseq = apiseq.split()
        if apiseq[-1] not in self.api2node:
            apiseq = apiseq[:-1]
        ignore_types = set(['basictype$void', 'basictype$generics', 'basictype$boolean'])
        added_param = [self.ceir2hash[param] for param in list(added_param) if self.ceir2hash[param] not in ignore_types]
        returned_params = set(added_param)
        for api in apiseq:
            try:
                body, rt = api.split('->')
            except ValueError:
                print(apiseq)
                print(api)
                input()
            rt = rt.replace('[]', '')
            rt = self.ceir2hash[rt]
            if rt not in ignore_types:
                returned_params.add(rt)
            sep = body.find('(')
            cls = self.ceir2hash['.'.join(body[:sep].split('.')[:-1])]
            returned_params.discard(cls)
            params = body[sep+1:-1]
            if len(params) == 0: continue
            for p in params.split(','):
                p = p.replace('[]', '')
                p = self.ceir2hash[p]
                returned_params.discard(p)
        print(returned_params)
        return len(returned_params)
    
    def cal_non_accept_count_w_rt(self, added_param, apiseq):
        apiseq = apiseq.split()
        if apiseq[-1] not in self.api2node:
            apiseq = apiseq[:-1]
        ignore_types = set()
        ignore_types.add('basictype$byte')
        ignore_types.add('basictype$float')
        ignore_types.add('basictype$double')
        ignore_types.add('basictype$int')
        ignore_types.add('basictype$short')
        ignore_types.add('basictype$long')
        ignore_types.add('basictype$char')
        ignore_types.add('basictype$boolean')
        ignore_types.add('basictype$void')
        ignore_types.add('basictype$generics')
        ignore_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/String.html$java.lang$String')
        ignore_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Integer.html$java.lang$Integer')
        ignore_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Float.html$java.lang$Float')
        ignore_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Double.html$java.lang$Double')
        ignore_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Long.html$java.lang$Long')
        ignore_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Short.html$java.lang$Short')
        ignore_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Character.html$java.lang$Character')
        ignore_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Boolean.html$java.lang$Boolean')
        ignore_types.add('class$JDK(java.base)$https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/lang/Byte.html$java.lang$Byte')

        added_param = [self.ceir2hash[param] for param in list(added_param) if self.ceir2hash[param] not in ignore_types]
        returned_params = set(added_param)
        for api in apiseq:
            node_item = self.api2node[api]
            rt = node_item['return']
            if rt not in ignore_types:
                returned_params.add(rt)
            cls = node_item['ceir']
            returned_params.discard(cls)
            params = node_item['parameter']
            if len(params) == 0: continue
            for p in params:
                returned_params.discard(p)
        return len(returned_params)
    
    def cal_non_accept_count_wo_rt(self, added_param, apiseq):
        added_param = set([self.ceir2hash[param] for param in list(added_param)])
        apiseq = apiseq.split()
        if apiseq[-1] not in self.api2node:
            apiseq = apiseq[:-1]
        for api in apiseq:
            if api not in self.api2node: continue
            node_item = self.api2node[api]
            cls = node_item['require ceir']
            if cls is not None:
                added_param.discard(cls)
            params = node_item['parameter']
            if len(params) == 0: continue
            for p in params:
                added_param.discard(p)
        return len(added_param)
    
    def analyze(self, eval_examples, pre_dir):
        total_n = 0
        total_np = 0
        with open(pre_dir, 'r', encoding='utf-8') as pf:
            for i, pline in enumerate(pf.readlines()):
                doc = eval_examples[i]
                pre = pline.split('\t')
                if len(pre) <= 1:
                    pre = ''
                else:
                    pre = pre[1]
                # if self.run(doc, deepcopy(pre)):
                required_types = self.extract_required_api(doc.split())
                if len(required_types) > 0:
                    n_p = self.cal_non_accept_count_wo_rt(required_types, pre)/len(required_types)
                    total_np += n_p
                    total_n += 1
        return {'PRE': total_np/total_n, 'N': total_n}
    
    def check_type1_error(self, apiseq:list) -> bool:
        # RETURN: True - 当前API序列不合法 | False - 当前API序列合法
        last_apis = []
        for i, api in enumerate(apiseq):
            if i == len(apiseq)-1:
                if not self.api2node.__contains__(api):
                    last_api_tokens = self.tokenizer.tokenize(api)
                    for tgt_api in self.api2node.keys(): # 最后一个API没预测完，需要找出所有可能的API并保存
                        if list_starswith_list(self.api2node[tgt_api]['token'], last_api_tokens):
                            last_apis.append(tgt_api)
                    if len(last_apis) == 0:
                        return True
            else:
                if not self.api2node.__contains__(api):
                    return True
        return False

    def check_type2_error(self, apiseq, required_types, default_types:set):
        last_apis = []
        if len(apiseq) == 0:
            return True

        if not self.api2node.__contains__(apiseq[-1]):
            last_api_tokens = self.tokenizer.tokenize(apiseq[-1])
            for tgt_api in self.api2node.keys(): # 最后一个API没预测完，需要找出所有可能的API并保存
                if list_starswith_list(self.api2node[tgt_api]['token'], last_api_tokens):
                    last_apis.append(tgt_api)

        apiseq = [api for api in apiseq if self.api2node.__contains__(api)]

        for ceir in required_types:
            ceir_hash = self.ceir2hash[ceir]
            default_types.add(ceir_hash)

        for api in apiseq:
            node_item = self.api2node[api]
            require_ceir = node_item['require ceir']
            param = node_item['parameter']
            rt = node_item['return']

            if require_ceir is not None:
                if require_ceir not in default_types:
                    return True

            for p in param:
                if p not in default_types:
                    return True
                
            for ceir in self.ceir2father[rt]:
                default_types.add(ceir)
        
        if len(last_apis) > 0:
            flag = False
            for api in last_apis:
                node_item = self.api2node[api]
                require_ceir = node_item['require ceir']
                param = node_item['parameter']

                if require_ceir is not None:
                    if require_ceir not in default_types:
                        continue

                for p in param:
                    if p not in default_types:
                        continue
                flag = True
                break
            return not flag

        return False
    
    def check_type3_error(self, apiseq, required_types):
        if len(apiseq) == 0:
            return True
        meets_types = set()
        apiseq = [api for api in apiseq if self.api2node.__contains__(api)]

        for api in apiseq:
            node_item = self.api2node[api]
            require_ceir = node_item['require ceir']
            param = node_item['parameter']

            if require_ceir is not None:
                meets_types.add(require_ceir)

            for p in param:
                meets_types.add(p)

        required_types = set([self.ceir2hash[tp] for tp in required_types])
        return not required_types.issubset(meets_types)
    
    def cal_type1_legal_rate(self, predictionMap, topn_list=[1,5,10]):
        legs = {k:0 for k in topn_list}
        nums = {k:0.0 for k in topn_list}
        for i, pre in predictionMap.items():
            for j in range(len(pre)):
                apiseq = pre[j]
                apiseq = apiseq.strip().split()
                leg = self.check_type1_error(apiseq)
                for k in legs.keys():
                    if j + 1 <= k:
                        legs[k] += (0 if leg else 1)
                        nums[k] += 1
        return {k:legs[k]/nums[k]*100 for k in topn_list}
    
    def cal_type2_legal_rate(self, eval_examples, predictionMap, topn_list=[1,5,10]):
        legs = {k:0 for k in topn_list}
        nums = {k:0.0 for k in topn_list}
        for i, pre in predictionMap.items():
            for j in range(len(pre)):
                apiseq = pre[j]
                apiseq = apiseq.strip().split()
                doc = eval_examples[int(i)].source
                doc = doc.strip().split()
                required_types = self.extract_required_api(doc)
                leg = self.check_type2_error(deepcopy(apiseq), deepcopy(required_types), deepcopy(self.default_types))
                for k in legs.keys():
                    if j + 1 <= k:
                        legs[k] += (0 if leg else 1)
                        nums[k] += 1
        return {k:legs[k]/nums[k]*100 for k in topn_list}
    
    def cal_type3_legal_rate(self, eval_examples, predictionMap, topn_list=[1,5,10]):
        legs = {k:0 for k in topn_list}
        nums = {k:0.0 for k in topn_list}
        for i, pre in predictionMap.items():
            for j in range(len(pre)):
                apiseq = pre[j]
                apiseq = apiseq.strip().split()
                doc = eval_examples[int(i)].source
                doc = doc.strip().split()
                required_types = self.extract_required_api(doc)
                leg = self.check_type3_error(deepcopy(apiseq), deepcopy(required_types))
                for k in legs.keys():
                    if j + 1 <= k:
                        legs[k] += (0 if leg else 1)
                        nums[k] += 1
        return {k:legs[k]/nums[k]*100 for k in topn_list}

if __name__ == '__main__':
    # preseq = 'java.io.BufferedReader.readLine() java.util.regex.Pattern.matcher(java.lang.CharSequence) java.util.regex.Matcher.matches()'
    # preseq2 = 'java.io.BufferedReader.readLine() java.util.regex.Pattern.matcher(java.lang.CharSequence) java.util.regex.Matcher.matches() java.util.regex.Matcher.group(java.lang.String)'
    # doc = 'read next item from file </s> java.io.BufferedReader java.util.regex.Pattern java.util.regex.Matcher'
    # print(chr.run(doc, preseq))
    # exit()
    chr = RuntimeChecker(path='..\data\nl2api\ent_dict.json', mode='no_return')
    eval_examples = []
    with open("./codeBERT/code2nl/data/nl2api/test.top1.ver9.jsonl", 'r', encoding='utf-8') as fdoc:
        for i, doc_l in enumerate(fdoc.readlines()):
            doc_l = json.loads(doc_l.strip())
            eval_examples.append(doc_l['doc'])
    idx = 'org'
    result = chr.analyze(eval_examples, pre_dir="./model/test_{}.output".format(idx))
    print(result)

    # required_types = chr.extract_required_api(doc.split())
    # print(chr.cal_non_accept_count(required_types, preseq))
    # print(chr.run(deepcopy(doc), preseq))

    
    