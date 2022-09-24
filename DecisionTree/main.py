from distutils.log import error
import numpy as np
import math
import pandas as pd
from utils import *
from model import *
from data_loaders import DataLoader
import sys

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
    
    if len(sys.argv) < 3:
        print("Please pass the arguments --")
        print("$1 for max depth...")
        print("$2 for split method type - [entropy, majority_error, gini_index]")
        quit()
    
    depth_ = int(sys.argv[1])
    purity_type_ = sys.argv[2]
    
    id3_car = ID3(label_values, attribute_values, max_depth=depth_, purity_type=purity_type_)
    id3_car.train(pd.DataFrame(train_df))
    
    #id3_car.root_node.print_tree()
    
    print(f"Train Error:  {train_dataloader.calculate_error(id3_car)}")
    print(f"Test Error:  {test_dataloader.calculate_error(id3_car)}")
    


        
            