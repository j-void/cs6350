import numpy as np
import math
import pandas as pd
from utils import *
from model import *
from data_loaders import DataLoader
import sys
    

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
    weights = [1] * train_dataloader.len
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
    # weights = [1] * train_dataloader.len
    # train_df["weights"] = weights
        
    
    test_dataloader = DataLoader("bank-2/test.csv", attribute_values, label_values)
    test_df = test_dataloader.convert_binary_test_data(train_dataloader.median_info)
    
    features = [2, 4, 6]
    features_dict = {str(i) : {} for i in features}
    
    for f_size in features:
        model_list = []
        train_errors = []
        test_errors = []
        steps = []
        for i in range(T):
            print(f"-------- Training Decision tree for t={i+1} ------------")
            ## Resample the data distribution
            train_df = train_df.sample(frac=1, replace=True)
            
            id3_bank = ID3(label_values, attribute_values, purity_type="entropy")
            id3_bank.train_random_forest(train_df, f_size)
            
            model_list.append(id3_bank)
            
            train_error= train_dataloader.calculate_bagging_error(model_list)
            test_error= test_dataloader.calculate_bagging_error(model_list)
            print(f"Error for f_size: {f_size} with {i+1} trees; Train={train_error}, Test={test_error}")
            train_errors.append(train_error)
            test_errors.append(test_error)
            steps.append(i)
        features_dict[str(f_size)] = {"train_errors":train_errors, "test_errors":test_errors, "trees":steps}
    
    import pickle
    with open('q2b_out.pkl', 'wb') as f:
        pickle.dump(features_dict, f)
        
    import matplotlib.pyplot as plt
    for key, value in features_dict.items():
        fig, ax = plt.subplots()
        ax.plot(np.array(value["steps"]), np.array(value["train_errors"]), '-bo', label="Train Error")
        ax.plot(np.array(value["steps"]), np.array(value["test_errors"]), '-ro', label="Test Error")
        ax.set(xlabel='Trees', ylabel='Error_f'+key)
        fig.savefig("q2d_f"+key+".png")