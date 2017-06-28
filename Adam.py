# -*- coding: utf-8 -*-
import numpy as np
import pandas

coef_dic = 1.0  # coefficient dicrement weights in all steps
coef_lear = 0.7 #coef_learning
coef_1 = 0.75   # coefficient for Adam
coef_2 = 0.75
alfa = 1/10.0



def sigmoid(x, alfa = 1/10.0, deriv = False):
    if (deriv == False):
         return 1.0/(1.0+np.exp(-x*alfa))
    return x*(1.0-x)

class NN:
    "neiral network direct propagation(feedforward)"
    def __init__(self, size_in, size_hidden, size_out):
        #initialization
        self.size_in = size_in
        self.size_hidden = size_hidden# example: size = np.array([100,150])
        self.size_out = size_out
        #hidden layers
        
        self.weights = []
        self.inc_g = [] # increment in all previos iteration learning
        self.inc_f = [] # increment in current iteration
        self.inc_m = [] # increment in current iteration
        self.delta_hidden = []
        self.bias_h = []
        self.train_values_h = []

        self.weights.append(np.random.random([size_in,self.size_hidden[0]]))
        self.inc_g.append(np.zeros([size_in,self.size_hidden[0]]))
        self.inc_f.append(np.zeros([size_in,self.size_hidden[0]]))
        self.inc_m.append(np.zeros([size_in,self.size_hidden[0]]))
        
        self.delta_hidden.append( np.zeros(self.size_hidden))  #error for one neiron, and after local gragient
        self.bias_h.append(np.random.random_sample())
        self.train_values_h.append(np.zeros(self.size_hidden[0]))#last calculate out of neiron
        
        for i in range(1,len(self.size_hidden)):
            self.weights.append(np.random.random([size_hidden[i-1],self.size_hidden[i]]))
            self.inc_g.append(np.zeros([size_hidden[i-1],self.size_hidden[i]]))
            self.inc_f.append(np.zeros([size_hidden[i-1],self.size_hidden[i]]))
            self.inc_m.append(np.zeros([size_hidden[i-1],self.size_hidden[i]]))
            
            self.delta_hidden.append( np.zeros(self.size_hidden))  #error for one neiron, and after local gragient
            self.bias_h.append(np.random.random_sample())
            self.train_values_h.append(np.zeros(self.size_hidden[i]))#last calculate out of neiron
            
        #out layer
        self.weights_out = np.random.random([self.size_hidden[-1], size_out])
        self.inc_g_out = np.zeros([self.size_hidden[-1], size_out])
        self.inc_f_out = np.zeros([self.size_hidden[-1], size_out])
        self.inc_m_out = np.zeros([self.size_hidden[-1], size_out])
        
        self.delta_out = np.zeros(size_out)
        self.bias_out = np.random.random_sample()
        self.train_values_out = np.zeros(size_out)
        
        self.E_out = np.zeros(size_out)# Error in last repeat for out neiral network

    def predict(self,X): # X - input example
    #prediction out of NN
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0

        self.train_values_h[0] = np.dot(self.weights[0].T,X)
        self.train_values_h[0] += self.bias_h[0]
        for i in range(0,self.size_hidden[0] ):
            self.train_values_h[0][i] = sigmoid(self.train_values_h[0][i])

        for j in range(1,len(self.size_hidden)):
            self.train_values_h[j] = np.dot(self.weights[j].T,self.train_values_h[j-1] )
            self.train_values_h[j] += self.bias_h[j]
            for i in range(0,self.size_hidden[j]):
                self.train_values_h[j][i] = sigmoid(self.train_values_h[j][i])
        self.train_values_out = np.dot(self.weights_out.T,self.train_values_h[-1])
 
        self.train_values_out += self.bias_out
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])

        return self.train_values_out
        
    def learning(self, X,y, coef=0.1,t=1): # X - input example, y - expected yield, coef - coefficient of learning
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0
        
        self.train_values_h[0] = np.dot(self.weights[0].T,X)
        self.train_values_h[0] += self.bias_h[0]
        for i in range(0,self.size_hidden[0] ):
            self.train_values_h[0][i] = sigmoid(self.train_values_h[0][i])

        for j in range(1,len(self.size_hidden)):
            self.train_values_h[j] = np.dot(self.weights[j].T,self.train_values_h[j-1] )
            self.train_values_h[j] += self.bias_h[j]
            for i in range(0,self.size_hidden[j]):
                self.train_values_h[j][i] = sigmoid(self.train_values_h[j][i])
        
        self.train_values_out = np.dot(self.weights_out.T,self.train_values_h[-1])
 
        self.train_values_out += self.bias_out
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])
      
        self.delta_out = -(y - self.train_values_out)
        self.E_out = sum(self.delta_out*self.delta_out/2)
        self.delta_out = self.delta_out  * sigmoid(self.train_values_out, deriv = True)
#        print self.E_out
        self.delta_hidden[-1] = np.dot(self.delta_out,self.weights_out.T)
        self.delta_hidden[-1] = self.delta_hidden[-1]  * sigmoid(self.train_values_h[-1], deriv = True)
        for j in range(len(self.size_hidden)-2 ,-1 ,-1):
            self.delta_hidden[j] = np.dot(self.delta_hidden[j+1],self.weights[j+1].T)
            self.delta_hidden[j] = self.delta_hidden[j]  * sigmoid(self.train_values_h[j], deriv = True)
        
        self.inc_f_out = np.dot(self.train_values_h[-1][:,None],self.delta_out[:,None].T)        
        self.inc_m_out = coef_1 * self.inc_m_out + (1.0 - coef_1) * self.inc_f_out
        self.inc_g_out = coef_2 * self.inc_g_out + (1.0 - coef_2) * self.inc_f_out**2
        self.weights_out = self.weights_out*coef_dic - coef * (self.inc_m_out/(1-coef_1**(t+1)))/(np.sqrt(self.inc_g_out/(1-coef_2**(t+1)))+1e-6)

        if (len(self.size_hidden)>1):
            for i in range(len(self.size_hidden)-1,0,-1):
                #self.inc_c[i] = np.dot(self.train_values_h[i-1][:,None],self.delta_hidden[i][:,None].T)
                #self.inc_w[i] = self.inc_w[i] * coef_A + (self.inc_c[i])**2
                #self.weights[i] = self.weights[i]*coef_dic - coef * self.inc_c[i] / (1e-6 + np.sqrt(self.inc_w[i]))
                
                self.inc_f[i] = np.dot(self.train_values_h[i-1][:,None],self.delta_hidden[i][:,None].T)
                self.inc_m[i] = coef_1 * self.inc_m[i] + (1.0 - coef_1) * self.inc_f[i]
                self.inc_g[i] = coef_2 * self.inc_g[i] + (1.0 - coef_2) * self.inc_f[i]**2
                self.weights[i] = self.weights[i]*coef_dic - coef * (self.inc_m[i]/(1-coef_1**(t+1)))/(np.sqrt(self.inc_g[i]/(1-coef_2**(t+1)))+1e-6)
     
#        self.inc_c[0] = np.dot(X[:,None],self.delta_hidden[0][:,None].T)
#        self.inc_w[0] = self.inc_w[0] * coef_A + (self.inc_c[0])**2
#        self.weights[0] = self.weights[0]*coef_dic - coef * self.inc_c[0] /(1e-6 + np.sqrt(self.inc_w[0]))

        self.inc_f[0] = np.dot(X[:,None],self.delta_hidden[0][:,None].T)
        self.inc_m[0] = coef_1 * self.inc_m[0] + (1.0 - coef_1) * self.inc_f[0]
        self.inc_g[0] = coef_2 * self.inc_g[0] + (1.0 - coef_2) * self.inc_f[0]**2
        self.weights[0] = self.weights[0]*coef_dic - coef * (self.inc_m[0]/(1-coef_1**(t+1)))/(np.sqrt(self.inc_g[0]/(1-coef_2**(t+1)))+1e-6)
        
        return self.E_out
'''
data = pandas.read_csv('train.csv')
y_train = data["label"] # 42000

X_train = data.drop("label", axis = 1) # 42000 * 784
X_train = X_train.as_matrix()
#X_train[X_train<100] = 0   #normalization
#X_train[X_train>1] = 1
X_train = X_train/250.0
from time import time
t = time()


MyNN = NN( 784,[100],10)
y_temp = np.zeros(10)
lear = 0
step_c = 0
for i in np.random.randint(42000, size = 30000):
    y_temp[y_train[i]] = 1
    MyNN.learning(X = X_train[i],y = y_temp,coef =0.7,t = step_c)
    if ( np.argmax(MyNN.predict(X = X_train[i])) == y_train[i]):
        lear +=1
        if(lear == 500):
            print "step_c=", step_c 
            break
    else:
        lear -=1
    step_c+=1
    y_temp[y_train[i]] = 0

#for i in np.random.randint(42000, size = 2000):
#    y_temp[y_train[i]] = 1
#    MyNN.learning(X = X_train[i],y = y_temp,coef =0.3)
#    step_c+=1
#   y_temp[y_train[i]] = 0


for i in range(50,60):
    print MyNN.predict(X_train[i])
    print y_train[i]    

#import matplotlib.pyplot as plt

acc = 0.0 
num = np.zeros(10)
for i in np.random.randint(42000, size = 1000):
   if(np.argmax(MyNN.predict(X_train[i])) == y_train[i]):
    acc+=1
    num[y_train[i]]+=1
#   else:
#        imgplot = plt.imsave( '%d' % i ,np.reshape(X_train[i],(28,28)))

print "accurance:", acc/1000
print num
print "time"
print time()-t
'''