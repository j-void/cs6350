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
    
    repeat_ = 3
    
    single_result = pd.DataFrame() #np.empty((test_dataloader.len, repeat_), dtype=object)
    bag_result = pd.DataFrame()
    
    for j in range(repeat_):
        model_list = []
        for i in range(2):
            print(f"-------- Training Decision tree for repeat={j+1} and t={i+1} ------------")
            
            ## Resample the data distribution
            train_df = train_df.sample(n=1000, replace=False)
            
            id3_bank = ID3(label_values, attribute_values, purity_type="entropy")
            id3_bank.train(train_df)
            
            model_list.append(id3_bank)
        output = test_dataloader.get_output_all(model_list[0])
        single_result[str(j)] = output
        output_b = test_dataloader.get_output_bagging(model_list)
        bag_result[str(j)] = output_b
    
    bias_single = []
    variance_single = []
    bias_bag = []
    variance_bag = []
    for i in range(test_dataloader.len):
        out_dict = dict.fromkeys(test_dataloader.label_out , 0)
        #print(len(single_result.iloc[i][single_result.iloc[i]=="no"]))
        for key in out_dict.keys():
            out_dict[key] = len(single_result.iloc[i][single_result.iloc[i]==key])
        avg_output = max(out_dict, key=out_dict.get)
        if test_dataloader.data_dict[test_dataloader.label_keys[0]][i] == avg_output:
            bias_single.append(0)
        else:
            bias_single.append(1)
        var = len(single_result.iloc[i][single_result.iloc[i]!= avg_output])/(repeat_-1)
        #print(var, avg_output, single_result.iloc[i])
        variance_single.append(var)
        
        for key in out_dict.keys():
            out_dict[key] = len(bag_result.iloc[i][bag_result.iloc[i]==key])
        avg_output = max(out_dict, key=out_dict.get)
        if test_dataloader.data_dict[test_dataloader.label_keys[0]][i] == avg_output:
            bias_bag.append(0)
        else:
            bias_bag.append(1)
        var = len(bag_result.iloc[i][bag_result.iloc[i]!= avg_output])/(repeat_-1)
        variance_bag.append(var)
        
    bias_single_avg = np.average(bias_single)
    variance_single_avg = np.average(variance_single)
    gse_single = bias_single_avg + variance_single_avg
    bias_bag_avg = np.average(bias_bag)
    variance_bag_avg = np.average(variance_bag)
    gse_bag = bias_bag_avg + variance_bag_avg
    print(f"Average Bias for single trees={bias_single_avg}")
    print(f"Average Variance for single trees={variance_single_avg}")
    print(f"Average Bias for bagged trees={bias_bag_avg}")
    print(f"Average Variance for bagged trees={variance_bag_avg}")
    print(f"General Squared Error for single trees={gse_single}")
    print(f"General Squared Error for bagged trees={gse_bag}")
    
    import pickle
    error_dict = {"bias_single":bias_single, "variance_single":variance_single, "bias_bag":bias_bag, "variance_bag":variance_bag,\
        "bias_single_avg":bias_single_avg, "variance_single_avg":variance_single_avg, "gse_single":gse_single,\
            "bias_bag_avg":bias_bag_avg, "variance_bag_avg":variance_bag_avg, "gse_bag":gse_bag}
    with open('q2c_out.pkl', 'wb') as f:
        pickle.dump(error_dict, f)
        
            
        