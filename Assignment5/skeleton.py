import numpy as np
import random
import pandas as pd

from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


def logistic_z(z): 
    return 1.0/(1.0+np.exp(-z))

def logistic_wx(w,x): 
    return logistic_z(np.inner(w,x))

def classify(w,x):
    x=np.hstack(([1],x))
    return 0 if (logistic_wx(w,x)<0.5) else 1
#x_train = [number_of_samples,number_of_features] = number_of_samples x \in R^number_of_features

def stochast_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in range(niter):
        if(len(index_lst)==0):
            index_lst=random.sample(range(num_n), k=num_n)
        xy_index = index_lst.pop()
        x=x_train[xy_index,:]
        y=y_train[xy_index]
        for i in range(dim):
            update_grad = 1 ### something needs to be done here
            w[i] = w[i] + learn_rate ### something needs to be done here
    return w

def batch_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in range(niter):
        for i in range(dim):
            update_grad=0.0
            for n in range(num_n):
                update_grad+=(-logistic_wx(w,x_train[n])+y_train[n])# something needs to be done here
            w[i] = w[i] + learn_rate * update_grad/num_n
    return w


def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=10):
    plt.figure()
    #train data
    data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
    ax=data.plot(kind='scatter',x='x',y='y',c='lab')

    #train weights
    w=training_method(xtrain,ytrain,learn_rate,niter)
    error=[]
    y_est=[]
    for i in range(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)
    data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    data_test.plot(kind='scatter',x='x',y='y',c='lab',ax=ax,cmap=cm.coolwarm)
    print("error=",np.mean(error))
    return w

def task_1_plot():
    w1 = np.arange(-6, 7, step=0.1)
    w2 = np.arange(-6, 7, step=0.1)
    X, Y = np.meshgrid(w1, w2)
    
    loss_func = lambda w: ((logistic_wx(w, [1,0])-1)**2 + logistic_wx(w, [1,0])**2 + (logistic_wx(w, [1,1]) - 1)**2)**2
    Z = []
    minimum = 100
    for i in range(len(X)):
        Z.append([])
        for j in range(len(Y)):
            val = loss_func([X[i,j], Y[i,j]])
            if val < minimum:
                minimum = val
                print("min:", minimum)
                print("weighs:", X[i,j], Y[i,j])
            Z[i].append(val)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')   
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False, linewidth = 0)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # From the graph, we observe that the minimum of the function is about 0.25, which is for the weights
    #  w1 = -2.13162820728e-14, w2 =  6.9 when the step size is 0.1
    
    
task_1_plot()   
