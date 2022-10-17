import numpy as np

class Node(object):
    def __init__(self, attribute=None):
        self.attribute = attribute
        self.childrens = {}
    
    def set_attr(self, attribute):
        self.attribute = attribute
    
    def add_child(self, input, attr):
        self.childrens[input] = attr
        
    def forward(self, input):
        return self.childrens[input]
    
    def print_tree(self, prev=""):
        prev += self.attribute
        #print(self.childrens.attribute)
        for input_, node_ in self.childrens.items():
            prev_ = prev + " - " + input_ + "\t"
            node_.print_tree(prev_)

        
class LeafNode(object):
    def __init__(self, value):
        self.value = value
    
    def forward(self):
        return self.value
    
    def print_tree(self, prev=""):
        print(prev+"("+self.value+")")


def save_learned_model(file, node):
    import pickle
    with open(file, 'wb') as outp:
        pickle.dump(node, outp, pickle.HIGHEST_PROTOCOL)
    
def load_learned_model(file):
    import pickle
    with open(file, 'rb') as inp:
        node = pickle.load(inp)
    return node