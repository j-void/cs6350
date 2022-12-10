import numpy as np
import math

def weight_init(size, type_="randn"):
    if type_ == "zero":
        return np.zeros((size))
    else:
        return np.random.randn(size)

class NN(object):
    def __init__(self, nh1, nh2, inp, wt_type="randn"):
        '''
        params
        ------
        nh2 = no. of neurons in 2nd hidden layer (layer2 - see diagram from hw)
        nh1 = no. of neurons in 1nd hidden layer (layer1 - see diagram from hw)
        inp = length of input x (layer0 - see diagram from hw)
        '''
        self.nh2 = nh2
        self.nh1 = nh1
        self.inp = inp

        self.weights  = []
        wt = {}
        for i in range(nh1):
            wt[str(i+1)] = weight_init(inp+1, type_=wt_type) #np.random.randn(inp+1)#np.zeros((inp+1)) # +1 for bias
        self.weights.append(wt)
        
        wt = {}
        for i in range(nh2):
            wt[str(i+1)] = weight_init(nh1+1, type_=wt_type)#np.random.randn(nh1+1)#np.zeros((nh1+1)) # +1 for bias
        self.weights.append(wt)
        
        wt = {"1":weight_init(nh2+1, type_=wt_type)} #{"1":np.zeros((nh2+1))} # +1 for bias
        self.weights.append(wt)

        assert len(self.weights) == 3
        self.z_list = []
        
    def sigmoid(self, x):
        return 1/(1 + math.exp(-x))
    
    def set_weights(self, wts):
        assert len(wts) == 3
        self.weights = wts
    
    def del_sigma(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def forward(self, x):
        x_prev = np.append(1, x)
        self.z_list.append(x_prev)
        ### Layer 1
        #for i in len()
        #z_ = [self.sigmoid(np.einsum('i,i', x_prev, value))]
        for iw, wt in enumerate(self.weights):
            if iw < len(self.weights)-1:
                z_ = np.ones((len(wt)+1))
                for i, (key, value) in enumerate(wt.items()):
                    z_[i+1] = self.sigmoid(np.einsum('i,i', x_prev, value))
            else:
                ## for last layer - just predicting the $y$ output
                z_ = np.einsum('i,i', x_prev, wt["1"])
            self.z_list.append(z_)
            x_prev = z_
        return self.z_list[-1]
    
    def backward(self, y, ys):
        dldy = y-ys
        dweights = []

        dydw = {}
        dldw = {}
        dws = []
        for i in range(self.nh2+1):
            dydw[str(i)+"13"] = self.z_list[-2][i]
            dws.append(dldy*dydw[str(i)+"13"])
        dldw["1"] = np.array(dws)
        dweights.append(dldw)
        
        dldz = {}
        dydz = {}
        value = self.weights[-1]["1"]
        for i in range(self.nh2+1):
            if i!=0:
                dydz[str(i)+"2"] = value[i]
            else:
                dydz[str(i)+"2"] = 0
            dldz[str(i)+"2"] = dldy*dydz[str(i)+"2"]
        #print(dldz)
        dldw = {}
        for j in range(self.nh2):
            dws = []
            for i in range(self.nh1+1):
                value= self.z_list[-3]
                if i!=0:
                    dydw[str(i)+str(j+1)+"2"] = dydz[str(j+1)+"2"]*self.del_sigma(np.einsum('i,i', value, self.weights[-2][str(j+1)]))*value[i]
                else:
                    dydw[str(i)+str(j+1)+"2"] = dydz[str(j+1)+"2"]*self.del_sigma(np.einsum('i,i', value, self.weights[-2][str(j+1)]))*1
                #dldw[str(i)++str(j+1)+"2"] = dldy*dydw[str(i)+str(j+1)+"2"]
                dws.append(dldy*dydw[str(i)+str(j+1)+"2"])
            dldw[str(j+1)] = np.array(dws)
        dweights.append(dldw)
            
        for i in range(self.nh1+1):
            dydz[str(i)+"1"] = 0
            for j in range(len(self.weights[-2])):
                value = self.weights[-2][str(j+1)]
                #print(self.weights[-2][str(j+1)])
                if i!=0 and (j+1)!= 0:
                    #print("dz/dz:", self.del_sigma(np.einsum('i,i', self.z_list[-3], value))*value[i], dydz[str(j+1)+"2"])
                    dydz[str(i)+"1"] += dydz[str(j+1)+"2"]*self.del_sigma(np.einsum('i,i', self.z_list[-3], value))*value[i]
                else:
                    dydz[str(i)+"1"] = 0
            #print("dy/dz",dydz[str(i)+"1"])
            dldz[str(i)+"1"] = dldy*dydz[str(i)+"1"]
        #print(dldz)
        dldw = {}
        for j in range(self.nh1):
            dws = []
            #print(self.nh1)
            for i in range(self.inp+1):
                value= self.z_list[-4]
                #if i!=0:
                #print(value.shape, i, self.weights[-3][str(j+1)].shape)
                dydw[str(i)+str(j+1)+"1"] = dydz[str(j+1)+"1"]*self.del_sigma(np.einsum('i,i', value, self.weights[-3][str(j+1)]))*value[i]
                #print(dydz[str(j+1)+"1"])
                #else:
                #    dydw[str(i)+str(j+1)+"1"] = dydz[str(j+1)+"2"]*self.del_sigma(np.einsum('i,i', value, self.weights[-3][str(j+1)]))*1
                dws.append(dldy*dydw[str(i)+str(j+1)+"1"])
            dldw[str(j+1)] = np.array(dws)
        #print(dldw)
        dweights.append(dldw)
        dweights.reverse()
        return dweights
    
    def update_weights(self, dw, alpha):
        for iw, w in enumerate(self.weights):
            for key, value in w.items():
                self.weights[iw][key] = self.weights[iw][key] - alpha*dw[iw][key]
        
        

if __name__ == "__main__":
    weights = []
    wt = {"1":np.array([-1, -2, -3]), "2":np.array([1, 2, 3])}
    weights.append(wt)
    wt = {"1":np.array([-1, -2, -3]), "2":np.array([1, 2, 3])}
    weights.append(wt)
    wt = {"1":np.array([-1, 2, -1.5])}
    weights.append(wt)
    nn = NN(2, 2, 2)
    nn.set_weights(weights)
    #print(nn.weights)
    ys = 1
    y = nn.forward(np.array([2, 3]))
    #print(y)
    #print(nn.z_list)
    weits = nn.backward(y, ys)
    print(weits)
    
    
