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
        "x1":      ["1", "0"],
        "x2":      ["1","2"],
        "x3":  ["0", "1", "2", "3", "4", "5", "6"],
        "x4":   ["0","1","2","3"],
        "x5":  ["1", "0"],
        "x6":  ["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"],
        "x7":  ["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"],
        "x8":  ["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"],
        "x9":  ["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"],
        "x10": ["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"],
        "x11": ["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"],
        "x12": ["1", "0"],
        "x13": ["1", "0"],
        "x14": ["1", "0"],
        "x15": ["1", "0"],
        "x16": ["1", "0"],
        "x17": ["1", "0"],
        "x18": ["1", "0"],
        "x19": ["1", "0"],
        "x20": ["1", "0"],
        "x21": ["1", "0"],
        "x22": ["1", "0"],
        "x23": ["1", "0"],
    }
    
    label_values = {"y": ["1", "0"]}
    
    ## processing dataset
    if False:
        print("--- Dividing data ---")
        df = pd.read_csv('credit_card/credit_card_clients.csv', header=None)
        train_df = df.sample(n=24000, replace=False)
        train_df.to_csv("credit_card/train.csv" ,index=False, header=None)
        test_df = df.drop(train_df.index)
        test_df.to_csv("credit_card/test.csv" ,index=False, header=None)
    
    train_dataloader = DataLoader("credit_card/train.csv", attribute_values, label_values)
    train_df = train_dataloader.convert_binary_01(["x1", "x5", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23"])
    weights = [1/train_dataloader.len] * train_dataloader.len
    train_df["weights"] = weights
    
    test_dataloader = DataLoader("credit_card/test.csv", attribute_values, label_values)
    test_df = test_dataloader.convert_binary_test_data_01(train_dataloader.median_info)
    
    ## Single Tree
    print("---- Running Single Decision Tree ----")
    id3_ = ID3(label_values, attribute_values, purity_type="entropy")
    id3_.train(train_df)
    
    train_error, _, _ = train_dataloader.calculate_error(id3_)
    test_error, _, _ = test_dataloader.calculate_error(id3_)
    print(f"Error for single tree: train={train_error}, test={test_error}")
    error_dict = {"single_tree": {"train_error":train_error, "test_error":test_error}}
    
    print("---- Running Bagging ----")
    steps = []
    train_errors = []
    test_errors = []
    model_list = []
    
    for j in range(500):
        
        #print(f"-------- Training for {j+1} trees - t={i+1}th tree ------------")
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
        print(f"Bagging - Error with t={j+1} trees; Train={train_error}, Test={test_error}")
        
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
        #print(f"-------- Training Decision tree for t={i+1} ------------")
        
        id3_bank = ID3(label_values, attribute_values, 1, purity_type="entropy")
        id3_bank.train_weighted(train_df)
        ## get current weights
        weights_ = train_df["weights"].to_numpy()

        e, ci, wi = train_dataloader.calculate_weighted_error(id3_bank, weights_)
        vote = math.log((1.0-e)/e)/2.0

        ## increase the weight of wrong examples and decrease weight of correct examples
        weights_[ci] = weights_[ci] * math.exp(-vote)
        weights_[wi] = weights_[wi] * math.exp(vote)

        weights_ = weights_ / np.sum(weights_)
        train_df["weights"] = weights_

        vote_list.append(vote)
        model_list.append(id3_bank)
        
        steps.append(i)
        train_error = train_dataloader.calculate_final_error(model_list, vote_list)
        test_error = test_dataloader.calculate_final_error(model_list, vote_list)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        print(f"Adaboost - Error after t={i} : train={train_error}, test={test_error}")
    
    fig, ax = plt.subplots()
    ax.plot(np.array(steps), np.array(train_errors), '-bo', label="Train Error")
    ax.plot(np.array(steps), np.array(test_errors), '-ro', label="Test Error")
    ax.set(xlabel='steps', ylabel='Error Combined')
    ax.legend()
    fig.savefig("q3_adaboost.png")
    
    error_dict["adaboost"] = {"train_errors":train_errors, "test_errors":test_errors}
    
    print("---- Running Random Forest using feature size = 12 ----")
    
    
    train_errors = []
    test_errors = []
    steps = []
    
    for i in range(500):
        #print(f"-------- Training Decision tree for f_size: {f_size} & t={i+1} ------------")
        ## Resample the data distribution
        train_df = train_df.sample(frac=1, replace=True)
        
        id3_bank = ID3(label_values, attribute_values, purity_type="entropy")
        id3_bank.train_random_forest(train_df, 12)
        
        model_list.append(id3_bank)
        
        train_error= train_dataloader.calculate_bagging_error(model_list)
        test_error= test_dataloader.calculate_bagging_error(model_list)
        print(f"Random Forest - Error with {i+1} trees; Train={train_error}, Test={test_error}")
        train_errors.append(train_error)
        test_errors.append(test_error)
        steps.append(i)
        
    error_dict["rf"] = {"train_errors":train_errors, "test_errors":test_errors}
    
    fig, ax = plt.subplots()
    ax.plot(np.array(steps), np.array(train_errors), '-bo', label="Train Error")
    ax.plot(np.array(steps), np.array(test_errors), '-ro', label="Test Error")
    ax.set(xlabel='steps', ylabel='Error Combined')
    ax.legend()
    fig.savefig("q3_rf.png")
    
    import pickle
    with open('q3_out.pkl', 'wb') as f:
        pickle.dump(error_dict, f)
    
    print(error_dict)