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
    single_result = pd.DataFrame()
    rf_result = pd.DataFrame()
    
    N = 5
    for f_size in features:
        model_list = []
        train_errors = []
        test_errors = []
        steps = []
        for i in range(N):
            print(f"-------- Training Decision tree for f_size: {f_size} & t={i+1} ------------")
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
        output = test_dataloader.get_output_all(model_list[0])
        single_result[str(f_size)] = output
        output_b = test_dataloader.get_output_bagging(model_list)
        rf_result[str(f_size)] = output_b
        
        bias_single = []
        variance_single = []
        bias_rf = []
        variance_rf = []
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
            var = len(single_result.iloc[i][single_result.iloc[i]!= avg_output])/(N-1)
            #print(var, avg_output, single_result.iloc[i])
            variance_single.append(var)
            
            for key in out_dict.keys():
                out_dict[key] = len(rf_result.iloc[i][rf_result.iloc[i]==key])
            avg_output = max(out_dict, key=out_dict.get)
            if test_dataloader.data_dict[test_dataloader.label_keys[0]][i] == avg_output:
                bias_rf.append(0)
            else:
                bias_rf.append(1)
            var = len(rf_result.iloc[i][rf_result.iloc[i]!= avg_output])/(N-1)
            variance_rf.append(var)
        bias_single_avg = np.average(bias_single)
        variance_single_avg = np.average(variance_single)
        gse_single = bias_single_avg + variance_single_avg
        bias_rf_avg = np.average(bias_rf)
        variance_rf_avg = np.average(variance_rf)
        gse_rf = bias_rf_avg + variance_rf_avg
        print(f"Average Bias for single trees={bias_single_avg}")
        print(f"Average Variance for single trees={variance_single_avg}")
        print(f"Average Bias for random forest ={bias_rf_avg}")
        print(f"Average Variance for random forest ={variance_rf_avg}")
        print(f"General Squared Error for single trees={gse_single}")
        print(f"General Squared Error for random forest ={gse_rf}")
        features_dict[str(f_size)]["bias_single_avg"]=bias_single_avg
        features_dict[str(f_size)]["variance_single_avg"]=variance_single_avg
        features_dict[str(f_size)]["bias_rf_avg"]=bias_rf_avg
        features_dict[str(f_size)]["variance_rf_avg"]=variance_rf_avg
        features_dict[str(f_size)]["gse_single"]=gse_single
        features_dict[str(f_size)]["gse_rf"]=gse_rf
        
    
    
    
    import pickle
    with open('q2de_out.pkl', 'wb') as f:
        pickle.dump(features_dict, f)
        
    import matplotlib.pyplot as plt
    for key, value in features_dict.items():
        fig, ax = plt.subplots()
        ax.plot(np.array(value["trees"]), np.array(value["train_errors"]), '-bo', label="Train Error")
        ax.plot(np.array(value["trees"]), np.array(value["test_errors"]), '-ro', label="Test Error")
        ax.set(xlabel='Trees', ylabel='Error_f'+key)
        ax.legend()
        fig.savefig("q2de_f"+key+".png")