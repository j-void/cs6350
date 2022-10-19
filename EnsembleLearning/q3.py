import numpy as np
import math
import pandas as pd
from utils import *
from model import *
from data_loaders import DataLoader
import sys
import matplotlib.pyplot as plt
    

if __name__ == "__main__":
    
    attribute_values = {
        "x1":      ["yes", "no"],
        "x2":      ["1","2"],
        "x3":  ["1","2","3", "4"],
        "x4":   ["1","2","3"],
        "x5":  ["yes", "no"],
        "x6":  ["-1", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "x7":  ["-1", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "x8":   ["-1", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "x9":  ["-1", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "x10":  ["-1", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "x11":   ["-1", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "x12": ["yes", "no"],
        "x13": ["yes", "no"],
        "x14":    ["yes", "no"],
        "x15": ["yes", "no"],
        "x16": ["yes", "no"],
        "x17": ["yes", "no"],
        "x18": ["yes", "no"],
        "x19": ["yes", "no"],
        "x20": ["yes", "no"],
        "x21": ["yes", "no"],
        "x22": ["yes", "no"],
        "x23": ["yes", "no"],
    }
    
    label_values = {"y": ["yes", "no"]}
    
    ## processing dataset
    df = pd.read_csv('credit_card/credit_card_clients.csv')
    train_df = df.sample(n=24000, replace=False)
    train_df.to_csv("credit_card/train.csv" ,index=False)
    test_df = df.drop(train_df.index)
    test_df.to_csv("credit_card/test.csv" ,index=False)
    
    train_dataloader = DataLoader("credit_card/train.csv", attribute_values, label_values)
    train_df = train_dataloader.convert_binary(["x1", "x5", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23"])
    weights = [1/train_dataloader.len] * train_dataloader.len
    train_df["weights"] = weights
    
    test_dataloader = DataLoader("credit_card/test.csv", attribute_values, label_values)
    test_df = test_dataloader.convert_binary_test_data(train_dataloader.median_info)
    
    ## Single Tree
    print("---- Running Single Decision Tree ----")
    id3_ = ID3(label_values, attribute_values, purity_type="entropy")
    id3_.train(train_df)
    
    train_error = train_dataloader.calculate_error(id3_)
    test_error = test_dataloader.calculate_error(id3_)
    print(f"Error for single tree: train={train_error}, test={test_error}")
    error_dict = {"single_tree": {"train_error":train_error, "test_error":test_error}}
    
    print("---- Running Bagging ----")
    steps = []
    train_errors = []
    test_errors = []
    
    for j in range(500):
        model_list = []
        for i in range(j+1):
            print(f"-------- Training for {j+1} trees - t={i+1}th tree ------------")
            
            ## Resample the data distribution
            train_df = train_df.sample(frac=1, replace=True)
            
            id3_bank = ID3(label_values, attribute_values, purity_type="entropy")
            id3_bank.train(train_df)
            
            model_list.append(id3_bank)
            
        train_error= train_dataloader.calculate_bagging_error(model_list)
        test_error = test_dataloader.calculate_bagging_error(model_list)
        train_errors.append(train_error)
        test_errors.append(test_error)
        steps.append(j)
        print(f"Error with t={j+1} trees; Train={train_error}, Test={test_error}")
        
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(np.array(steps), np.array(train_errors), '-bo', label="Train Error")
    ax.plot(np.array(steps), np.array(test_errors), '-ro', label="Test Error")
    ax.legend()
    ax.set(xlabel='steps', ylabel='Error')
    fig.savefig("q3_bagging.png")
    
    error_dict["bagging"] = {"train_errors":train_errors, "test_errors":test_errors}
    
    print("---- Running Adaboost ----")
    
    vote_list = []
    model_list = []
    train_errors = []
    test_errors = []
    
    steps = []
    
    for i in range(500):
        print(f"-------- Training Decision tree for t={i+1} ------------")
        #print(train_df)
        id3_bank = ID3(label_values, attribute_values, 1, purity_type="entropy")
        id3_bank.train(train_df)
        
        #id3_bank.root_node.print_tree()
        e, ci, wi = train_dataloader.calculate_error(id3_bank)
        vote = math.log((1-e)/e)/2
        
        ## get the weights
        weights_ = train_df["weights"].to_numpy()
        ## increase the weight of wrong examples and decrease weight of correct examples
        weights_[ci] = weights_[ci] * math.exp(-vote)
        weights_[wi] = weights_[wi] * math.exp(vote)
        
        ## normalize the weights
        weights_ = weights_ / np.sum(weights_)
        train_df["weights"] = weights_
        vote_list.append(vote)
        model_list.append(id3_bank)
        
        steps.append(i)
        train_error = train_dataloader.calculate_final_error(model_list, vote_list)
        test_error = test_dataloader.calculate_final_error(model_list, vote_list)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        print(f"Error after t={i} : train={train_error}, test={test_error}")
    
    fig, ax = plt.subplots()
    ax.plot(np.array(steps), np.array(train_errors), '-bo', label="Train Error")
    ax.plot(np.array(steps), np.array(test_errors), '-ro', label="Test Error")
    ax.set(xlabel='steps', ylabel='Error Combined')
    ax.legend()
    fig.savefig("q3_adaboost.png")
    
    error_dict["adaboost"] = {"train_errors":train_errors, "test_errors":test_errors}
    
    print("---- Running Random Forest ----")
    
    import pickle
    with open('q3_out.pkl', 'wb') as f:
        pickle.dump(error_dict, f)