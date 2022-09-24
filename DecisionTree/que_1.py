from distutils.log import error
import numpy as np
import math
import pandas as pd
from utils import *
from model import *
from data_loaders import DataLoader
    

if __name__ == "__main__":
    
    attribute_values = {
        "buying":   ["vhigh", "high", "med", "low"],
        "maint":    ["vhigh", "high", "med", "low"],
        "doors":    ["2", "3", "4", "5more"],
        "persons":  ["2", "4", "more"],
        "lug_boot": ["small", "med", "big"],
        "safety":   ["low", "med", "high"]
    }
    label_values = {"label": ["unacc", "acc", "good", "vgood"]}
    train_dataloader = DataLoader("car/train.csv", attribute_values, label_values)
    train_df = train_dataloader.load_data()
    test_dataloader = DataLoader("car/test.csv", attribute_values, label_values)
    test_df = test_dataloader.load_data()
    
    
    # id3_car = ID3(label_values, attribute_values, max_depth=1, purity_type="gini_index")
    # id3_car.train(pd.DataFrame(train_df))
    
    # id3_car.root_node.print_tree()
    
    #print(f"Train Error:  {train_dataloader.calculate_error(id3_car)}")
    # print(f"Test Error:  {test_dataloader.calculate_error(id3_car)}")
    
    
    split_types = {"entropy":0, "majority_error":0, "gini_index":0}
    for split_type in list(split_types.keys()):
        error_dict = {}
        max_depth = 6
        error_dict[split_type+"-Train"] = [] 
        error_dict[split_type+"-Test"] = []
        prev_train_error = 0
        prev_test_error = 0
        for depth in range(1,max_depth+1):
            #print(f"-------Running Train using {split_type} and depth={depth}")
            id3_car = ID3(label_values, attribute_values, max_depth=depth, purity_type=split_type)
            id3_car.train(train_df)
            
            train_error = train_dataloader.calculate_error(id3_car)
            print(f"Train Error: {train_error}")
            test_error = test_dataloader.calculate_error(id3_car)
            print(f"Test Error: {test_error}")
            if train_error == prev_train_error and test_error == prev_test_error:
                max_depth = depth
                break
            error_dict[split_type+"-Train"].append(train_error)
            error_dict[split_type+"-Test"].append(test_error)
            prev_train_error = train_error
            prev_test_error = test_error
        split_types[split_type] = pd.DataFrame(error_dict, index=list(range(1,depth+1)))
    
    print("-------------- Errors ---------------")
    for key, value in split_types.items():
        print(value)


        
            