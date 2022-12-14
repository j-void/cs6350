import math
import numpy as np
import pandas as pd 
from scipy import optimize

def cost(y, x, w):
    jw = 0
    for i in range(len(y)):
        x_ = np.append(x[i], 1)
        y_out = np.sign(np.dot(w.transpose(),x_))
        if y[i]!=y_out:
            jw += 1
    return jw/len(y)


def load_data(path):
    y = []
    x = []
    with open(path , 'r') as f : 
        for line in f :
            terms = line.strip().split(',')
            x.append(terms[:-1])
            y.append(terms[-1])
    return np.array(x).astype(float), np.array(y).astype(float)

def load_data_cmp(path):
    data = []
    with open(path , 'r') as f : 
        for line in f :
            terms = line.strip().split(',')
            data.append(terms)
    return np.array(data).astype(float)

if __name__ == "__main__":
    ## load the data
    x_test, y_test = load_data("bank-note/test.csv")
    train_data = load_data_cmp("bank-note/train.csv")
    train_data[:,-1][train_data[:,-1]==0] = -1
    y_test[y_test==0] = -1
    
    ## intialize the weight vector
    w_init = np.zeros((train_data[:,:-1].shape[1]), dtype=float)
    C_list = [100/873, 500/873, 700/873]
    gammas = [0.1,0.5,1,5,100]
    train_data_new = train_data.copy()
    x_train, y_train = train_data_new[:,:-1], train_data_new[:,-1]
    
    import pickle
    print('Please run hw4_q3b.py if not done already')
    print("Loading the results obtained from previous run of hw4_q3b.py")
    saved_output = pickle.load( open( "q3b_out.pkl", "rb" ) )

    
    for ci, C in enumerate(C_list):
        for gi, g in enumerate(gammas):
            alphas = saved_output[str(ci)][str(g)]["alpha"]
            print(f"Length of support vectors for C={C} and gamma={g} is {np.where(alphas>0)[0].shape[0]}")
            
    for i in range(len(gammas)-1):
        alphas_1 = saved_output["1"][str(gammas[i])]["alpha"]
        alphas_2 = saved_output["1"][str(gammas[i+1])]["alpha"]
        svm_idx1 = np.where(alphas_1>0)[0]
        svm_idx2 = np.where(alphas_2>0)[0]
        same_idx = np.in1d(svm_idx1, svm_idx2)
        print(f'Number of overlapped support vectors between gamma=({gammas[i]}, {np.where(alphas_1>0)[0].shape[0]}) and gamma=({gammas[i+1]}, {np.where(alphas_2>0)[0].shape[0]}) is {np.sum(same_idx)}')
        
        
    
    
        