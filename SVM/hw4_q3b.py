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

def cost_c(y, yt, xt, x, c, g, b):
    jw = 0
    for i in range(len(y)):
        dist = np.linalg.norm(xt-x[i], axis=1)
        kernel = np.exp(-(dist**2/g))
        y_out = np.sum(np.einsum('i,i,i->i', c, yt, kernel))
        if y[i]!=np.sign(y_out):
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
    
    out_dict = {}
    
    for ci, C in enumerate(C_list):
        out_dict[str(ci)] = {}
        for gi, g in enumerate(gammas):
            w = w_init
            alpha_init = np.zeros((y_train.shape[0]), dtype=float)
            def fun(alpha):
                dist = np.linalg.norm(x_train-x_train[:, np.newaxis], axis=2)
                X = np.exp(-(dist**2/g))
                A = np.einsum("i,j->ij", y_train*alpha, y_train*alpha)
                AX = np.einsum("ij,ji->i", A, X)
                
                return 0.5*np.sum(AX) - np.sum(alpha)
            
            cons=({'type': 'eq',
                'fun': lambda a: np.dot(a.transpose(),y_train)})
            
            bnds = ((0, C),) * alpha_init.shape[0]
            
            res = optimize.minimize(fun, alpha_init, method='SLSQP', bounds=bnds, constraints=cons)
            for i in range(len(y_train)):
                w += res.x[i]*y_train[i]*x_train[i]
                
            idx = (res.x>0) & (res.x<C)
            yb = y_train[idx]
            xb = x_train[idx]
            b = np.mean(yb - np.einsum('i,ji->j', w, xb))
            w = np.append(w, b)

            #train_cost = cost(y_train, x_train, w)
            train_cost = cost_c(y_train, y_train, x_train, x_train, res.x, g, b)
            #print(f"Final weights={w} for C={C} and gamma={g}")
            #test_cost = cost(y_test, x_test, w)
            test_cost = cost_c(y_test, y_train, x_train, x_test, res.x, g, b)
            print(f"Error for C={C} and gamma={g} : Train={train_cost}; Test={test_cost}")
            print(f"Length of support vectors={np.where(res.x>0)[0].shape[0]}")
            out_dict[str(ci)][str(g)] = {"train_cost":train_cost, "test_cost":test_cost, "w":w, "alpha":res.x}
    
    import pickle
    print('Saving the output in q3b_out.pkl')
    with open('q3b_out.pkl', 'wb') as f:
        pickle.dump(out_dict, f)
    
        