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
    w1 = np.arange(-6, 6, step=0.1)
    w2 = np.arange(-6, 6, step=0.1)
    X, Y = np.meshgrid(w1, w2)
    
    Z = []
    minimum = 100
    for i in range(len(X)):
        Z.append([])
        for j in range(len(Y)):
            val = simple_loss_func([X[i,j], Y[i,j]])
            if val < minimum:
                minimum = val
                print("min:", minimum)
                print("weigths:", X[i,j], Y[i,j])
            Z[i].append(val)
    plot_surface(X, Y, Z, heatmap=True)
    
    # From the graph, we observe that the minimum of the function is about 0.005, which is for the weights
    # 5.9 and -2.9, but there is a whole area that is low, so all weights in that region is good.
    
def plot_surface(X, Y, Z, heatmap=False):
    if heatmap:
        plt.pcolormesh(X, Y, Z)
        plt.colorbar()
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.gca(projection='3d')   
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False, linewidth = 0)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

def simple_loss_func(w):
    return (logistic_wx(w, [1,0])-1)**2 + logistic_wx(w, [0,1])**2 + (logistic_wx(w, [1,1]) - 1)**2
def simple_loss_gradient(w):
    # The term in the middle should be dropped
    w1_deriv = 2*(logistic_wx(w, [1,0]) - 1)*logistic_wx_deriv(w,[1,0])[0] + 2*(logistic_wx(w, [1,1]) - 1)*logistic_wx_deriv(w, [1,1])[0]
    w2_deriv = 2*logistic_wx(w, [0,1])*logistic_wx_deriv(w, [0,1])[1]      + 2*(logistic_wx(w, [1,1]) - 1)*logistic_wx_deriv(w, [1,1])[1]
    return [w1_deriv, w2_deriv]

def logistic_wx_deriv(w, x):
    # the derivative is defined as e^-w*x/(1+e^-w*x)[x1, x2] where * is dot product (w^t x)
    constant = np.exp(-np.inner(w, x)) / (1+ np.exp(- np.inner(w,x)))**2
    return np.multiply(constant, x)
    

def gradient_descent(num_iters = 1000, phi=0.1, print_interval=100, weights = None):
    # The start weights should probably be random floats instead of ints
    if not weights:
        weights = [12*np.random.random()-6, 12*np.random.random()-6]
    
    for t in range(num_iters):
        if t % print_interval-1 == 0: print("\n====================\nIteration %s, weights : %s, loss_func: %s" % (t, str(weights), simple_loss_func(weights)))
        for i in range(len(weights)):
            delta = -phi * simple_loss_gradient(weights)[i]
            # We only allow weights from [-6,6]
            if -6 < weights[i] + delta < 6: 
                weights[i] += delta 
    return weights


def main():
    learning_rates = [10**i for i in range(-2, 4)]
    print(learning_rates)
    results = []
    for phi in learning_rates:
        # Set starting weights so that we can reproduce results
        weights = gradient_descent(num_iters= 10000, phi= phi, weights = [3, 3], print_interval=10000)
        results.append(simple_loss_func(weights))
    print(results)
    plt.plot(learning_rates, results)
    plt.xscale('log')
    plt.ylabel('Loss func')
    plt.xlabel('learning rate')
    plt.show()

main()

#gradient_descent(num_iters=1000, print_interval=1000)

#task_1_plot()   
