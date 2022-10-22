
import math
from platform import node
import numpy as np
from utils import *
import pandas as pd 
import random

def calculate_entropy(p):
    entropy = 0
    for i in range(len(p)):
        if p[i] == 0:
            return 0.0
        entropy += -p[i]*math.log2(p[i])
    return entropy

def calculate_majority_error(p):
    return min(p)

def calculate_gini_index(p):
    gi_ = 0
    for i in range(len(p)):
        gi_ += p[i]**2
    return 1- gi_

def IG(es, s, e):
    sum_ = 0
    sum_div = sum(s)
    for i in range(len(s)):
        sum_ += (s[i]/sum_div)*e[i]
    #print(s, e)
    return es - sum_

def calculate_prob(S, label, text=" "):
    key_ = list(label.keys())[0]
    unique_labels = label[key_]
    total_ = S.shape[0]
    prob_ = []
    for i in range(len(unique_labels)):
        prob_.append(len(S.loc[S[key_]==unique_labels[i]])/total_)
        #print("For",text,": P(",unique_labels[i],")",len(S.loc[S[key_]==unique_labels[i]]), "/", total_)
    return prob_

def calculate_weight_prob(S, label):
    key_ = list(label.keys())[0]
    unique_labels = label[key_]
    prob_ = []
    total_ = S["weights"].sum()
    #print(total_)
    for i in range(len(unique_labels)):
        p = S.loc[S[key_]==unique_labels[i]]
        prob_.append(p["weights"].sum()/total_)
    return prob_


class ID3(object):
    def __init__(self, label, attributes, max_depth=math.inf, purity_type="entropy"):
        self.label = label
        self.attributes = attributes
        self.max_depth = max_depth + 1
        ## set the default to use entropy for Information Gain calculation
        self.purity_fn = calculate_entropy
        ## set the purity functions based on requirement
        self.purity_type = purity_type
        if self.purity_type=="majority_error":
            self.purity_fn = calculate_majority_error
        if self.purity_type=="gini_index":
            self.purity_fn = calculate_gini_index
        self.root_node = None
        
            
    def load_learned(self, file_path):
        self.root_node = load_learned_model(file_path)
        print(f"-------- Loaded learned model ------------")
        
    def train(self, data):
        #print(f"-------- Training Decision tree using {self.purity_type} for max depth {self.max_depth-1} ------------")
        self.root_node = self.build_tree(data, self.attributes, self.max_depth)
        #print(f"-------- Training Done ------------")
        return self.root_node
    
    def train_weighted(self, data):
        #print(f"-------- Training Decision tree using {self.purity_type} for max depth {self.max_depth-1} ------------")
        self.root_node = self.build_tree_weighted(data, self.attributes, self.max_depth)
        #print(f"-------- Training Done ------------")
        return self.root_node
    
    def train_random_forest(self, data, feature_size):
        self.root_node = self.rand_learn_tree(data, self.attributes, self.max_depth, feature_size)
        return self.root_node

    def save_learned(self, file_path):
        save_learned_model(file_path, self.root_node)
        print(f"-------- Saved learned model at {file_path} ------------")
    
    ## Don't make any changes here
    def build_tree(self, data, attributes, max_depth):
        main_key = list(self.label.keys())[0]
        main_unique_labels = self.label[main_key]# S[label].unique()
        ## Take care of max depth condition
        #print(max_depth)
        max_depth -= 1
        if max_depth <= 0:
            output_ = main_unique_labels[0]
            max_ = -1
            for i in range(len(main_unique_labels)):
                llen = len(data.loc[data[main_key]==main_unique_labels[i]])
                if llen > max_:
                    max_ = llen
                    output_ = main_unique_labels[i]
            #print("Leaf Node (", output_,",",max_,")")
            return LeafNode(output_)
        
        ## take care when all attributes are used (empty)
        if attributes == {}:
            output_ = main_unique_labels[0]
            max_ = -1
            for i in range(len(main_unique_labels)):
                llen = len(data.loc[data[main_key]==main_unique_labels[i]])
                if llen > max_:
                    max_ = llen
                    output_ = main_unique_labels[i]
            #print("Leaf Node (", output_,",",max_,")")
            return LeafNode(output_) 
        ### calculate probability
        prob_main = calculate_prob(data, self.label)
        #print("prob_main:", prob_main)
        entropy_total = self.purity_fn(prob_main)
        #print("main entropy =", entropy_total)
        max_ig = -math.inf
        best_attr = ""
        attribute_keys = list(attributes.keys())
        ## Loop over all value of the attribute to check for the one with max information gain
        for i in range(len(attribute_keys)):
            unique_labels_ = attributes[attribute_keys[i]] #S[attributes.keys()[i]].unique()
            #print(unique_labels_)
            entropy_ = []
            length_ = []
            for j in range(len(unique_labels_)):
                data_current = data.loc[data[attribute_keys[i]]==unique_labels_[j]]
                if data_current.shape[0] == 0:
                    continue
                prob_ = calculate_prob(data_current, self.label)
                #print("prob -", attribute_keys[i] ,"-", unique_labels_[j],"-", prob_)
                entropy_current = self.purity_fn(prob_)
                #print("Entropy =", entropy_current)
                entropy_.append(entropy_current)
                length_.append(data_current.shape[0])
            ## calculate information gain
            #print("length_:", length_)
            ig_ = IG(entropy_total, length_, entropy_)
            ## selecting the best attribute
            if ig_ > max_ig:
                max_ig = ig_
                best_attr = attribute_keys[i]
            #print("IG of", attribute_keys[i], "=",ig_)
        #print("Best Attribute:", best_attr)
        #print(attribute_keys)
        root_node = Node(best_attr)
        unique_labels_new = attributes[best_attr]
        ## process the values of chosen attribute
        for j in range(len(unique_labels_new)):
            #print("processing - ",best_attr,"-",unique_labels_new[j])
            data_new = data.loc[data[best_attr]==unique_labels_new[j]]
            #print(data_new)
            divide_tree = True
            ## if the value of attribute not present in data snapshot then just select the max label in attribute for the specific value
            if len(data_new.index) == 0:
                output_ = main_unique_labels[0]
                max_ = -1
                for k in range(len(main_unique_labels)):
                    llen = data.loc[data[main_key]==main_unique_labels[k]].shape[0]
                    if llen > max_:
                        max_ = llen
                        output_ = main_unique_labels[k]
                #print("Leaf Node (", output_,", 0 )")
                leaf_ = LeafNode(output_)#Node(output_)
                root_node.add_child(unique_labels_new[j], leaf_)
                divide_tree = False
                continue
            ## else process that specific attribute value
            for k in range(len(main_unique_labels)):
                #print(main_unique_labels[k], list(self.label.keys())[0])
                temp_data = data_new.loc[data_new[main_key]==main_unique_labels[k]]
                ## add a leaf node if all labels for the specific attribute value are same
                if temp_data.shape[0] == data_new.shape[0]:
                    #print("Leaf Node at -", best_attr,"- (", main_unique_labels[k],",",temp_data.shape[0],")")
                    divide_tree = False
                    leaf_ = LeafNode(main_unique_labels[k])#Node(main_unique_labels[k])
                    root_node.add_child(unique_labels_new[j], leaf_)
                    continue
            ## else futher divide the tree
            if divide_tree:
                new_attributes = attributes.copy()
                del new_attributes[best_attr]
                node_ = self.build_tree(data_new, new_attributes, max_depth)
                if node_ != None:
                    root_node.add_child(unique_labels_new[j], node_)
        
        return root_node
    
    def build_tree_weighted(self, data, attributes, max_depth):
        main_key = list(self.label.keys())[0]
        main_unique_labels = self.label[main_key]# S[label].unique()
        ## Take care of max depth condition
        #print(max_depth)
        max_depth -= 1
        if max_depth <= 0:
            output_ = main_unique_labels[0]
            max_ = -1
            for i in range(len(main_unique_labels)):
                #print(data.loc[data[main_key]==main_unique_labels[i]])
                llen = data.loc[data[main_key]==main_unique_labels[i]]["weights"].sum()
                if llen > max_:
                    max_ = llen
                    output_ = main_unique_labels[i]
            #print("Leaf Node (", output_,",",max_,")")
            return LeafNode(output_)
        
        ## take care when all attributes are used (empty)
        if attributes == {}:
            output_ = main_unique_labels[0]
            max_ = -1
            for i in range(len(main_unique_labels)):
                llen = data.loc[data[main_key]==main_unique_labels[i]]["weights"].sum()#len(data.loc[data[main_key]==main_unique_labels[i]])
                if llen > max_:
                    max_ = llen
                    output_ = main_unique_labels[i]
            #print("Leaf Node (", output_,",",max_,")")
            return LeafNode(output_) 
        ### calculate probability
        prob_main = calculate_weight_prob(data, self.label)
        #print("prob_main:", prob_main)
        entropy_total = self.purity_fn(prob_main)
        #print("main entropy =", entropy_total)
        max_ig = -math.inf
        best_attr = ""
        attribute_keys = list(attributes.keys())
        ## Loop over all value of the attribute to check for the one with max information gain
        for i in range(len(attribute_keys)):
            unique_labels_ = attributes[attribute_keys[i]] #S[attributes.keys()[i]].unique()
            #print(unique_labels_)
            entropy_ = []
            length_ = []
            for j in range(len(unique_labels_)):
                data_current = data.loc[data[attribute_keys[i]]==unique_labels_[j]]
                if data_current.shape[0] == 0:
                    continue
                prob_ = calculate_weight_prob(data_current, self.label)
                #print("prob -", attribute_keys[i] ,"-", unique_labels_[j],"-", prob_)
                entropy_current = self.purity_fn(prob_)
                #print("Entropy =", entropy_current)
                entropy_.append(entropy_current)
                length_.append(data_current["weights"].sum())
            ## calculate information gain
            #print("length_:", length_)
            ig_ = IG(entropy_total, length_, entropy_)
            ## selecting the best attribute
            if ig_ > max_ig:
                max_ig = ig_
                best_attr = attribute_keys[i]
            #print("IG of", attribute_keys[i], "=",ig_)
        #print("Best Attribute:", best_attr)
        #print(attribute_keys)
        root_node = Node(best_attr)
        unique_labels_new = attributes[best_attr]
        ## process the values of chosen attribute
        for j in range(len(unique_labels_new)):
            #print("processing - ",best_attr,"-",unique_labels_new[j])
            data_new = data.loc[data[best_attr]==unique_labels_new[j]]
            #print(data_new)
            divide_tree = True
            ## if the value of attribute not present in data snapshot then just select the max label in attribute for the specific value
            if len(data_new.index) == 0:
                output_ = main_unique_labels[0]
                max_ = -1
                for k in range(len(main_unique_labels)):
                    #print(data.loc[data[main_key]==main_unique_labels[k]])
                    llen = data.loc[data[main_key]==main_unique_labels[i]]["weights"].sum()#data.loc[data[main_key]==main_unique_labels[k]].shape[0]
                    if llen > max_:
                        max_ = llen
                        output_ = main_unique_labels[k]
                #print("Leaf Node (", output_,", 0 )")
                leaf_ = LeafNode(output_)#Node(output_)
                root_node.add_child(unique_labels_new[j], leaf_)
                divide_tree = False
                continue
            ## else process that specific attribute value
            for k in range(len(main_unique_labels)):
                #print(main_unique_labels[k], list(self.label.keys())[0])
                temp_data = data_new.loc[data_new[main_key]==main_unique_labels[k]]
                #print(temp_data)
                ## add a leaf node if all labels for the specific attribute value are same
                if temp_data.shape[0] == data_new.shape[0]:
                    #print("Leaf Node at -", best_attr,"- (", main_unique_labels[k],",",temp_data.shape[0],")")
                    divide_tree = False
                    leaf_ = LeafNode(main_unique_labels[k])#Node(main_unique_labels[k])
                    root_node.add_child(unique_labels_new[j], leaf_)
                    continue
            ## else futher divide the tree
            if divide_tree:
                new_attributes = attributes.copy()
                del new_attributes[best_attr]
                node_ = self.build_tree_weighted(data_new, new_attributes, max_depth)
                if node_ != None:
                    root_node.add_child(unique_labels_new[j], node_)
        
        return root_node
    
    def rand_learn_tree(self, data, attributes, max_depth, feature_size):        
        main_key = list(self.label.keys())[0]
        main_unique_labels = self.label[main_key]# S[label].unique()
        #print(len(attributes.items()))
        
        ## Take care of max depth condition
        #print(max_depth)
        max_depth -= 1
        if max_depth <= 0:
            output_ = main_unique_labels[0]
            max_ = -1
            for i in range(len(main_unique_labels)):
                llen = len(data.loc[data[main_key]==main_unique_labels[i]])
                if llen > max_:
                    max_ = llen
                    output_ = main_unique_labels[i]
            #print("Leaf Node (", output_,",",max_,")")
            return LeafNode(output_)
        
        ## take care when all attributes are used (empty)
        if attributes == {}:
            output_ = main_unique_labels[0]
            max_ = -1
            for i in range(len(main_unique_labels)):
                llen = len(data.loc[data[main_key]==main_unique_labels[i]])
                if llen > max_:
                    max_ = llen
                    output_ = main_unique_labels[i]
            #print("Leaf Node (", output_,",",max_,")")
            return LeafNode(output_) 
        ## subset the attributes
        if len(attributes.items()) > feature_size:
            attributes_sample = dict(random.sample(attributes.items(), feature_size))
        else:
            attributes_sample = attributes.copy()
        ### calculate probability
        prob_main = calculate_prob(data, self.label)
        #print("prob_main:", prob_main)
        entropy_total = self.purity_fn(prob_main)
        #print("main entropy =", entropy_total)
        max_ig = -math.inf
        best_attr = ""
        attribute_keys = list(attributes_sample.keys())
        ## Loop over all value of the attribute to check for the one with max information gain
        for i in range(len(attribute_keys)):
            unique_labels_ = attributes_sample[attribute_keys[i]] #S[attributes.keys()[i]].unique()
            #print(unique_labels_)
            entropy_ = []
            length_ = []
            for j in range(len(unique_labels_)):
                data_current = data.loc[data[attribute_keys[i]]==unique_labels_[j]]
                if data_current.shape[0] == 0:
                    continue
                prob_ = calculate_prob(data_current, self.label)
                #print("prob -", attribute_keys[i] ,"-", unique_labels_[j],"-", prob_)
                entropy_current = self.purity_fn(prob_)
                #print("Entropy =", entropy_current)
                entropy_.append(entropy_current)
                length_.append(data_current.shape[0])
            ## calculate information gain
            #print("length_:", length_)
            ig_ = IG(entropy_total, length_, entropy_)
            ## selecting the best attribute
            if ig_ > max_ig:
                max_ig = ig_
                best_attr = attribute_keys[i]
            #print("IG of", attribute_keys[i], "=",ig_)
        #print("Best Attribute:", best_attr)
        #print(attribute_keys)
        root_node = Node(best_attr)
        unique_labels_new = attributes_sample[best_attr]
        ## process the values of chosen attribute
        for j in range(len(unique_labels_new)):
            #print("processing - ",best_attr,"-",unique_labels_new[j])
            data_new = data.loc[data[best_attr]==unique_labels_new[j]]
            #print(data_new)
            divide_tree = True
            ## if the value of attribute not present in data snapshot then just select the max label in attribute for the specific value
            if len(data_new.index) == 0:
                output_ = main_unique_labels[0]
                max_ = -1
                for k in range(len(main_unique_labels)):
                    llen = data.loc[data[main_key]==main_unique_labels[k]].shape[0]
                    if llen > max_:
                        max_ = llen
                        output_ = main_unique_labels[k]
                #print("Leaf Node (", output_,", 0 )")
                leaf_ = LeafNode(output_)#Node(output_)
                root_node.add_child(unique_labels_new[j], leaf_)
                divide_tree = False
                continue
            ## else process that specific attribute value
            for k in range(len(main_unique_labels)):
                #print(main_unique_labels[k], list(self.label.keys())[0])
                temp_data = data_new.loc[data_new[main_key]==main_unique_labels[k]]
                ## add a leaf node if all labels for the specific attribute value are same
                if temp_data.shape[0] == data_new.shape[0]:
                    #print("Leaf Node at -", best_attr,"- (", main_unique_labels[k],",",temp_data.shape[0],")")
                    divide_tree = False
                    leaf_ = LeafNode(main_unique_labels[k])#Node(main_unique_labels[k])
                    root_node.add_child(unique_labels_new[j], leaf_)
                    continue
            ## else futher divide the tree
            if divide_tree:
                new_attributes = attributes.copy()
                del new_attributes[best_attr]
                node_ = self.rand_learn_tree(data_new, new_attributes, max_depth, feature_size)
                if node_ != None:
                    root_node.add_child(unique_labels_new[j], node_)
        
        return root_node
                
    def run_inference(self, input_dict):
        node_ = self.root_node
        while input_dict:
            if type(node_) != LeafNode:
                current_attr = node_.attribute
                #print("Checking: ", current_attr, input_dict[node_.attribute])
                node_ = node_.forward(input_dict[node_.attribute])
            else:
                output = node_.forward()
                return output
        return None