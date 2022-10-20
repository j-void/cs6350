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
    
    train_dataloader = DataLoader("bank-2/train.csv", attribute_values, label_values)
    train_df = train_dataloader.convert_binary(["age", "balance", "day", "duration", "campaign", "pdays", "previous"])
    weights = [1/train_dataloader.len] * train_dataloader.len
    train_df["weights"] = weights
    
    # attribute_values = {
    #     "outlook": ["sunny", "overcast", "rain"],
    #     "temperature": ["hot", "mild", "cool"],
    #     "humidity": ["high", "normal", "low"],
    #     "wind": ["strong", "weak"]
    # }
    # label_values = {"PlayTennis": ["yes", "no"]}
    # train_dataloader = DataLoader("ex.csv", attribute_values, label_values)
    # train_df = train_dataloader.load_data()
    # weights = [1/train_dataloader.len] * train_dataloader.len
    # train_df["weights"] = weights
    
    vote_list = []
    model_list = []
    train_errors = []
    test_errors = []
    train_inv_errors = []
    test_inv_errors = []
    
    test_dataloader = DataLoader("bank-2/test.csv", attribute_values, label_values)
    test_df = test_dataloader.convert_binary_test_data(train_dataloader.median_info)
    
    steps = []
    
    # a = train_df.loc[train_df["duration"]=="yes"]
    # print(len(a.loc[a["y"]=="yes"]), len(a.loc[a["y"]=="no"]))
    # e_a = len(a.loc[a["y"]=="yes"])
    # b = train_df.loc[train_df["duration"]=="no"]
    # print(len(b.loc[b["y"]=="yes"]), len(b.loc[b["y"]=="no"]))
    # e_b = len(b.loc[b["y"]=="yes"])
    # print((e_a+e_b)/train_dataloader.len)
    
    # a = train_df.loc[train_df["pdays"]=="yes"]
    # print(len(a.loc[a["y"]=="yes"]), len(a.loc[a["y"]=="no"]))
    # b = train_df.loc[train_df["pdays"]=="no"]
    # print(len(b.loc[b["y"]=="yes"]), len(b.loc[b["y"]=="no"]))
    
    # a = train_df.loc[train_df["duration"]=="yes"]
    # print(len(a.loc[a["y"]=="yes"]), len(a.loc[a["y"]=="no"]))
    # b = train_df.loc[train_df["duration"]=="no"]
    # print(len(b.loc[b["y"]=="yes"]), len(b.loc[b["y"]=="no"]))
    
    
    
    for i in range(50):
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
        #print(vote)
        steps.append(i)
        train_error = train_dataloader.calculate_final_error(model_list, vote_list)
        test_error = test_dataloader.calculate_final_error(model_list, vote_list)
        train_inv_error, _, _ = train_dataloader.calculate_error(id3_bank)
        test_inv_error, _, _ = test_dataloader.calculate_error(id3_bank)
        
        # train_errors.append(train_error)
        # test_errors.append(test_error)
        # train_inv_errors.append(train_inv_error)
        # test_inv_errors.append(test_inv_error)
        
        print(f"Train Error after t={i} : {train_error}, {train_inv_error}")
        print(f"Test Error after t={i} : {test_error}, {test_inv_error}")
    
    # import pickle
    # error_dict = {"train_errors":train_errors, "test_errors":test_errors, "train_inv_errors":train_inv_errors, "test_inv_errors":test_inv_errors}
    # with open('q2a_out.pkl', 'wb') as f:
    #     pickle.dump(error_dict, f)
    
    # fig, ax = plt.subplots()
    # ax.plot(np.array(steps), np.array(train_errors), '-bo', label="Train Error")
    # ax.plot(np.array(steps), np.array(test_errors), '-ro', label="Test Error")
    # ax.set(xlabel='steps', ylabel='Error Combined')
    # ax.legend()
    # fig.savefig("q2a_combined.png")
    # fig, ax = plt.subplots()
    # ax.plot(np.array(steps), np.array(train_inv_errors), '-bo', label="Train Error")
    # ax.plot(np.array(steps), np.array(test_inv_errors), '-ro', label="Test Error")
    # ax.set(xlabel='steps', ylabel='Error Individual')
    # ax.legend()
    # fig.savefig("q2a_individual.png")