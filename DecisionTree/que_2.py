import numpy as np
import math
import pandas as pd
from utils import *
from model import *
from data_loaders import DataLoader
import sys
    

if __name__ == "__main__":

    ## preprocess the data
    #data_dict = {"buying": [], "maint": [], "doors":[], "persons":[], "lug_boot":[], "safety":[], "label":[]}
    
    attribute_values = {
        "age":      ["yes", "no"],
        "job":      ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services"],
        "marital":  ["married","divorced","single"],
        "education":["unknown","secondary","primary","tertiary"],
        "default":  ["yes", "no"],
        "balance":  ["yes", "no"],
        "housing":  ["yes", "no"],
        "loan":     ["yes", "no"],
        "contact":  ["unknown","telephone","cellular"],
        "day":      ["yes", "no"],
        "month":    ['sep', 'may', 'apr', 'jul', 'aug', 'jan', 'oct', 'jun', 'nov', 'dec', 'feb', 'mar'],
        "duration": ["yes", "no"],
        "campaign": ["yes", "no"],
        "pdays":    ["yes", "no"],
        "previous": ["yes", "no"],
        "poutcome": ["unknown","other","failure","success"],
    }
    label_values = {"y": ["yes", "no"]}
    train_dataloader = DataLoader("bank/train.csv", attribute_values, label_values)
    train_df = train_dataloader.convert_binary(["age", "balance", "day", "duration", "campaign", "pdays", "previous"])
    test_dataloader = DataLoader("bank/test.csv", attribute_values, label_values)
    test_df = test_dataloader.convert_binary_test_data(train_dataloader.median_info)
    
    split_types = {"entropy":0, "majority_error":0, "gini_index":0}
    
    
    if len(sys.argv) > 1:
        p1 = False if sys.argv[1] == "1" else True
    else:
        p1 = True
    if p1:
        ## For part (a)
        print("Running part (a) for Que-2")
        for split_type in split_types:
            error_dict = {}
            max_depth = 16
            error_dict[split_type+"-Train"] = [] 
            error_dict[split_type+"-Test"] = []
            prev_train_error = 0
            prev_test_error = 0
            for depth in range(1,max_depth+1):
                id3_bank = ID3(label_values, attribute_values, max_depth=depth, purity_type=split_type)
                id3_bank.train(train_df)
                
                train_error = train_dataloader.calculate_error(id3_bank)
                print(f"Train Error: {train_error}")
                test_error = test_dataloader.calculate_error(id3_bank)
                print(f"Test Error: {test_error}")
                if train_error == prev_train_error and test_error == prev_test_error:
                    max_depth = depth
                    break
                error_dict[split_type+"-Train"].append(train_error)
                error_dict[split_type+"-Test"].append(test_error)
                prev_train_error = train_error
                prev_test_error = test_error
            #print(len(error_dict[split_type+"-Train"]), depth+1)
            split_types[split_type] = pd.DataFrame(error_dict, index=list(range(1,depth)))
        
    else:
        print("Running part (b) for Que-2")
        ## For part (b)
        attribute_values, train_df, max_labels = train_dataloader.fill_missing_variables("unknown")
        test_df = test_dataloader.fill_missing_variables_test("unknown", max_labels)
        for split_type in split_types:
            error_dict = {}
            max_depth = 16
            error_dict[split_type+"-Train"] = [] 
            error_dict[split_type+"-Test"] = []
            prev_train_error = 0
            prev_test_error = 0
            for depth in range(1,max_depth+1):
                id3_bank = ID3(label_values, attribute_values, max_depth=depth, purity_type=split_type)
                id3_bank.train(train_df)
                
                train_error = train_dataloader.calculate_error(id3_bank)
                print(f"Train Error: {train_error}")
                test_error = test_dataloader.calculate_error(id3_bank)
                print(f"Test Error: {test_error}")
                if train_error == prev_train_error and test_error == prev_test_error:
                    max_depth = depth
                    break
                error_dict[split_type+"-Train"].append(train_error)
                error_dict[split_type+"-Test"].append(test_error)
                prev_train_error = train_error
                prev_test_error = test_error
            split_types[split_type] = pd.DataFrame(error_dict, index=list(range(1,depth)))
        
    print("-------------- Errors ---------------")
    for key, value in split_types.items():
        print(value)
    
    # id3_bank = ID3(label_values, attribute_values)
    # id3_bank.train(train_df)
    # # print("-------- printing tree output : start -------")
    # # id3_car.root_node.print_tree()
    # # print("-------- printing tree output : end -------")

    # # # ## save the learned model
    # # id3_car.save_learned("save_q2.pkl")
    
    # ## load the learned model
    # #id3_car.load_learned("save_q2.pkl")
    # depth = math.inf
    # print(f"Train Error with depth={depth+1} : {train_dataloader.calculate_error(id3_bank)}")
    # print(f"Test Error with depth={depth+1} : {test_dataloader.calculate_error(id3_bank)}")
    
    ### for part 2
    #attribute_values, train_df = train_dataloader.fill_missing_variables("unknown")
    # id3_bank = ID3(label_values, attribute_values)
    # id3_bank.train(train_df)
    # print(f"Train Error: {train_dataloader.calculate_error(id3_bank)}")
    
    
        
    
    


        
            