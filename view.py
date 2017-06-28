# -*- coding: utf-8 -*-
import numpy as np
import pandas
from time import time
from matplotlib import pyplot as plt
plt.figure(1)
#module = __import__('foo')
#func = getattr(module, 'bar')
#func()
#np.random.seed(10)

packets = ['SGD' ,'Nesterov', 'Adagrad','RMSProp', 'Adadelta', 'Adam']
#packets = ['Adam']
for packet in packets:
    print packet
    module = __import__(packet)
    NN = getattr(module, 'NN')
    data = pandas.read_csv('train.csv')
    y_train = data["label"]
    X_train = data.drop("label", axis = 1)
    X_train = X_train.as_matrix()
    X_train[X_train<100] = 0 
    X_train[X_train>1] = 1
    t = time()

    mapX = np.arange(0.0, 100.0)
    mapY = np.zeros(100)
    numY = 0 
    y_temp = np.zeros(10)
    MyNN = NN( 784,[100],10)

    j = 0
    for i in np.random.randint(42000, size = 29999):
        y_temp[y_train[i]] = 1
        MyNN.learning(X = X_train[i],y = y_temp,coef =0.8)
        if(j%300 == 0 ):
            acc =0.0
            for i in np.random.randint(42000, size = 50):
                if(np.argmax(MyNN.predict(X_train[i])) == y_train[i]):
                    acc+=1
            mapY[numY] = acc/50.0        
            numY = numY+1
        j = j+1
        y_temp[y_train[i]] = 0

        
    print "time"
    print time()-t
    
    plt.figure(1) 
    
    plt.axis([0,100, 0,1])
    plt.plot(mapX,mapY,'b+')
    #print mapY
    #plt.show()
    #print str(packet)
    #plt.savefig(str(packet).split('\'')[1])
    plt.savefig(str(packet))
    plt.clf()
    
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