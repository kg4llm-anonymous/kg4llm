import json
from queue import Queue
from copy import deepcopy
import time
from api_tree import api_tree

def load_json(path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_ceir_type(node):
    for tp in ['class', 'interface', 'enum', 'record']:
        if node.__contains__(tp):
            return tp
        

def list_starswith_list(l1, l2) -> bool:
    if len(l2) > len(l1): return False
    for item1, item2 in zip(l1, l2):
        if item1 != item2: return False
    return True

class RuntimeGuider:
    def __init__(self, tokenizer, mode='no_return') -> None:
        assert mode in ['normal', 'no_return']
        self.tokenizer = tokenizer
        self.mode = mode
        self.preprocess(r'ent_dict.json')
        self.last_apis= []
    
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
        required_apis = set([self.ceir2hash[tp] for tp in doc])
        return required_apis

    def preprocess(self, ent_dict_fname):
        api2node = dict()
        default_types = set()
        ceir2hash = dict()
        ceir2father = dict()
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
        for tp in list(default_types):
            ceir2father[tp] = [tp]
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
                if (v['return'] == 'None') and ('abstract' in ent_dict[ceir_type]['field']): continue
                
                parameter_list = [param[0] for param in v['parameter']]
                parameter_dim_list = [param[1] for param in v['parameter']]
                return_type = ceir_type if v['return'] == 'None' else v['return'][0]
                return_dim = 0  if v['return'] == 'None' else v['return'][1]

                require_ceir = []
                if v['return'] == 'None':
                    pass
                elif ('static' in v['field']) and ('public' in v['field']):
                    pass
                else:
                    require_ceir = [ceir_type]
                node = {
                    'package': v['package'],
                    'ceir': v[ent_dict[ceir_type]['type']],
                    'method': v['method'],
                    'requires':require_ceir,
                    'parameter':parameter_list,
                    'parameter dim':parameter_dim_list,
                    'return':return_type,
                    'return dim':return_dim,
                }
                name = self.get_api_name(node)
                node['token'] = self.tokenizer.tokenize(name)
                api2node[name] = node
            elif v['type'] == 'member':
                if  ('static' in v['field']) and ('public' in v['field']):
                    default_types.add(v['return'][0])
            elif v['type'] in ('class', 'interface', 'enum', 'record'):
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

        for api in self.api2node.keys():
            self.api2node[api]['requires'] = set(self.api2node[api]['parameter'] + self.api2node[api]['requires'])

        self.tree = api_tree('<s>')
        for api, node in self.api2node.items():
            self.tree.add_item(['<s>'] + node['token'], api)
    
    def extract_returned_api(self, apiseq, required_types):
        last_api = ''
        if (len(apiseq) > 0) and (apiseq[-1] not in self.api2node):
            last_api = apiseq[-1]
            apiseq = apiseq[:-1]
        returned_types = set()
        for api in apiseq:
            if api not in self.api2node:
                return last_api, returned_types, False
            for father_node in self.ceir2father[self.api2node[api]['return']]:
                returned_types.add(father_node)

        return last_api, returned_types, True

    
    def run(self, required_types:set, apiseq):
        if (len(apiseq) > 0) and ((apiseq[-1] == self.tokenizer.eos_token_id) or (apiseq[-1] == self.tokenizer.pad_token_id)):
            return [self.tokenizer.pad_token_id]
            
        apiseq = self.tokenizer.decode(apiseq, clean_up_tokenization_spaces=False)
        apiseq = apiseq.split()

        last_api, returned_types, legal_flag = self.extract_returned_api(apiseq, required_types)
        if not legal_flag: return []

        required_types = required_types | returned_types | self.default_types

        candicate_apis = []
        candicate_tokens = set()


        non_first_api_starts = False

        if (len(last_api) == 0) and (len(apiseq) > 0):
            candicate_tokens.add(self.tokenizer.eos_token)
            non_first_api_starts = True

        last_api_tokens = self.tokenizer.tokenize(last_api)

        for trt_api in self.tree.get_items(['<s>'] + last_api_tokens):
            api_node = self.api2node[trt_api]
            if (len(api_node['requires']) == 0) or (api_node['requires'].issubset(required_types)):
                candicate_apis.append(trt_api)

        for tp in candicate_apis:
            token = self.api2node[tp]['token'][len(last_api_tokens)]
            if non_first_api_starts:
                token = '\u0120' + token
            candicate_tokens.add(token)
        

        candicate_tokens = self.tokenizer.convert_tokens_to_ids(list(candicate_tokens))

        return candicate_tokens


    
    