from copy import deepcopy
class api_tree:
    def __init__(self, label):
        self.label = label
        self.items = []
        self.childs = {}
    
    def exist(self, label):
        if label == self.label:
            return True
        else:
            for node in self.data:
                if node.exist(label): return True
        return False
    
    def add_item(self, label_list, item):
        label_list = deepcopy(label_list)
        assert label_list[0] == self.label
        label_list = label_list[1:]
        self.items.append(item) # 过路就加
        if len(label_list) > 0:
            label = label_list[0]
            if label not in self.childs:
                self.childs[label] = api_tree(label)
            self.childs[label].add_item(label_list, item)
    
    def get_items(self, label_list):
        label_list = deepcopy(label_list)
        assert label_list[0] == self.label
        
        label_list = label_list[1:]
        if len(label_list) == 0:
            return self.items
        else:
            label = label_list[0]
            if label not in self.childs:
                return []
            return self.childs[label].get_items(label_list)

if __name__ == '__main__':
    seqs = [
        ['<s>', 'java', '.', 'lang'],
        ['<s>', 'java', '.', 'io'],
        ['<s>', 'j', 'dk', '.', 'open']
    ]
    datas = [
        ['java.lang.1', 'java.lang.2', 'java.lang.3', 'java.lang.4'],
        ['java.io.1', 'java.io.2', 'java.io.3'],
        ['jdk.open.1', 'jdk.open.2'],
    ]
    t = api_tree('<s>')
    for seq, data in zip(seqs, datas):
        for sub in data:
            t.add_item(seq, sub)

    print(t.get_items(['<s>', 'java']))