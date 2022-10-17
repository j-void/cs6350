import numpy as np
import math
import pandas as pd
from collections import Counter

class DataLoader(object):
    def __init__(self, path, attribute_values, label_values):
        self.attribute_values = attribute_values.copy()
        self.attribute_keys = list(attribute_values.keys())
        self.label_keys = list(label_values.keys())
        self.label_out = label_values[self.label_keys[0]]
        all_keys = self.attribute_keys + self.label_keys
        self.data_dict = {key: [] for key in all_keys}
        self.path = path
        self.len = 0
        self.median_info = {}

        

    def load_data(self):
        count = 0
        with open(self.path , 'r') as f : 
            for line in f :
                terms = line.strip().split(',')
                for i, key in enumerate(self.data_dict.keys()):
                    self.data_dict[key].append(terms[i])
                count += 1
        self.len = count
        return pd.DataFrame(self.data_dict)
    
    
    def convert_binary(self, numerical_attributes):
        self.load_data()
        for attr in numerical_attributes:
            arr_ = np.array(self.data_dict[attr]).astype(int)
            median = np.median(arr_)
            self.data_dict[attr] = np.where(arr_>=median, "yes", "no").tolist()
            self.median_info[attr] = median
        return pd.DataFrame(self.data_dict)
    
    def convert_binary_test_data(self, median_dict):
        self.load_data()
        for attr in list(median_dict.keys()):
            arr_ = np.array(self.data_dict[attr]).astype(int)
            median_ = median_dict[attr]
            self.data_dict[attr] = np.where(arr_>=median_, "yes", "no").tolist()
        return pd.DataFrame(self.data_dict)
        
    def fill_missing_variables(self, value):
        import operator
        max_labels = {}
        for x, y in self.attribute_values.items():
            if value in y:
                y.remove(value)
                self.attribute_values[x] = y
                arr_ = np.array(self.data_dict[x])
                counts = dict(Counter(arr_))
                del counts[value]
                max_elem = max(counts.items(), key=operator.itemgetter(1))[0]
                #print(f"Max for {x}: {max_elem}")
                max_labels[x] = max_elem
                self.data_dict[x] = np.where(arr_ == value, max_elem, arr_)
        
        return self.attribute_values, pd.DataFrame(self.data_dict), max_labels
    
    def fill_missing_variables_test(self, value, max_labels):
        for x, y in max_labels.items():
            arr_ = np.array(self.data_dict[x])
            self.data_dict[x] = np.where(arr_ == value, y, arr_)
            #print(self.data_dict[x])
        #print(self.attribute_values)
        return pd.DataFrame(self.data_dict)
        
    
    def calculate_error(self, model):
        error = 0
        correct_index = []
        wrong_index = []
        for i in range(self.len):
            input_ = {}
            for key in self.attribute_keys:
                input_[key] = self.data_dict[key][i]
            output_y = model.run_inference(input_)
            if output_y != self.data_dict[self.label_keys[0]][i]:
                #print(output_y, self.data_dict[self.label_keys[0]][i], input_, i)
                error += 1
                wrong_index.append(i)
            else:
                correct_index.append(i)
        #print("Wrong Labels =", error, self.len)
        return error/self.len, correct_index, wrong_index
    
    def get_output_all(self, model):
        outputs = []
        for i in range(self.len):
            input_ = {}
            for key in self.attribute_keys:
                input_[key] = self.data_dict[key][i]
            output_y = model.run_inference(input_)
            outputs.append(output_y)
        return outputs
    
    def calculate_final_error(self, models, votes):
        error = 0
        for i in range(self.len):
            input_ = {}
            out_dict = dict.fromkeys(self.label_out , 0)
            for key in self.attribute_keys:
                input_[key] = self.data_dict[key][i]
            for j in range(len(models)):
                output_ = models[j].run_inference(input_)
                out_dict[output_] += votes[j]
            output_y = max(out_dict, key=out_dict.get)
            if output_y != self.data_dict[self.label_keys[0]][i]:
                #print(output_y, self.data_dict[self.label_keys[0]][i], input_, i, out_dict)
                error += 1
        #print("Wrong Labels =", error, self.len)
        return error/self.len
    
    def calculate_bagging_error(self, models):
        error = 0
        for i in range(self.len):
            input_ = {}
            out_dict = dict.fromkeys(self.label_out , 0)
            for key in self.attribute_keys:
                input_[key] = self.data_dict[key][i]
            for j in range(len(models)):
                output_ = models[j].run_inference(input_)
                out_dict[output_] += 1
            output_y = max(out_dict, key=out_dict.get)
            #print(out_dict, output_y)
            if output_y != self.data_dict[self.label_keys[0]][i]:
                #print(output_y, self.data_dict[self.label_keys[0]][i], input_, i)
                error += 1
        #print("Wrong Labels =", error, self.len)
        return error/self.len
    
    def get_output_bagging(self, models):
        outputs = []
        for i in range(self.len):
            input_ = {}
            out_dict = dict.fromkeys(self.label_out , 0)
            for key in self.attribute_keys:
                input_[key] = self.data_dict[key][i]
            for j in range(len(models)):
                output_ = models[j].run_inference(input_)
                out_dict[output_] += 1
            output_y = max(out_dict, key=out_dict.get)
            outputs.append(output_y)
        return outputs
    