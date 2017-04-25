import numpy as np
import random
import pandas as pd

from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import time

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
            update_grad = (logistic_wx(w,x) - y)*loss_func_deriv(w,x)*x ### something needs to be done here
            w[i] -= learn_rate*update_grad[i] ### something needs to be done here
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
                x = x_train[n]
                y = y_train[n]
                update_grad += (logistic_wx(w,x) - y) * loss_func_deriv(w, x)*x
            w[i] -=  learn_rate * update_grad[i]/num_n
    return w


def loss_func_deriv(w, x):
    return logistic_wx(w, x)*(1 - logistic_wx(w,x))



def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=10, plot=True):
    #train data
    data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
    
    #train weights
    w=training_method(xtrain,ytrain,learn_rate,niter)
    error=[]
    y_est=[]
    for i in range(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)

    data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    if plot:
        ax=data.plot(kind='scatter',x='x',y='y',c='lab')
        data_test.plot(kind='scatter',x='x',y='y',c='lab',ax=ax,cmap=cm.coolwarm)
        plt.show()
    print("error=",np.mean(error))
    
    return w, np.mean(error)

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
    plot_surface(X, Y, Z, heatmap=False)
    
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
        
        for i in range(len(weights)):
            delta = -phi * simple_loss_gradient(weights)[i]
            # We only allow weights from [-6,6]
            if -6 < weights[i] + delta < 6: 
                weights[i] += delta 
            if t % print_interval == 0: print("\n====================\nIteration %s, weights : %s, loss_func: %s" % (t, str(weights), simple_loss_func(weights)))
    return weights




def main_1():
    learning_rates = [10**i for i in range(-4, 4)]
    print(learning_rates)
    results = []
    weight_list = []
    for phi in learning_rates:
        # Set starting weights so that we can reproduce results
        weights = gradient_descent(num_iters= 1000, phi= phi, weights = [3, 3])
        weight_list.append(weights)
        results.append(simple_loss_func(weights))
    print(results)
    print("Lowest error: ", min(results))
    print(weight_list)
    plt.plot(learning_rates, results)
    plt.xscale('log')
    plt.ylabel('Loss func')
    plt.xlabel('learning rate')
    plt.show()

#main()

#main()
#task_1_plot()

def extract_feats_and_class(datalines):

    x = np.array([line[:2] for line in datalines], dtype=np.float64)
    y = np.array([line[-1] for line in datalines], dtype=np.float64)
    return x, y


def main(nonsep_dataset = False):
    if nonsep_dataset:
        train_datalines = np.loadtxt("data/data_big_nonsep_train.csv", delimiter='\t')
        test_datalines = np.loadtxt("data/data_big_nonsep_test.csv")
    else:
        train_datalines = np.loadtxt("data/data_big_separable_train.csv", delimiter='\t')
        test_datalines = np.loadtxt("data/data_big_separable_test.csv")
    
    
    
    x_train, y_train = extract_feats_and_class(train_datalines)
    x_test, y_test = extract_feats_and_class(test_datalines)
    train_and_plot(x_train, y_train, x_test, y_test, stochast_train_w, niter=1000)

    n_iter = [10, 25, 50, 100, 250, 500, 1000]
    train_methods = [ stochast_train_w, batch_train_w]
    results = []
    for func in train_methods:
        results.append([])
        for iters in n_iter:
            
            weights, error = train_and_plot(x_train, y_train, x_test, y_test, func, niter= iters, plot=False)
            results[len(results) - 1].append(error)
    plt.plot(n_iter, results[0], 'r', n_iter, results[1], 'b')
    plt.show()


def timing():
    train_datalines = np.loadtxt("data/data_big_separable_train.csv", delimiter='\t')
    test_datalines = np.loadtxt("data/data_big_separable_test.csv")

    x_train, y_train = extract_feats_and_class(train_datalines)
    x_test, y_test = extract_feats_and_class(test_datalines)

    n_iter = [10, 20, 50, 100, 200, 500, 1000, 2000]
    errors = []
    times = []
    for iter in n_iter:
        start = time.time()
        w, e = train_and_plot(x_train, y_train, x_test, y_test, stochast_train_w, niter= iter, plot=False)
        end = time.time()
        errors.append(e)
        times.append(end - start)

    a = plt.plot(n_iter, times, label="Execution time")
    b = plt.plot(n_iter, errors, label = "Error rate")
    plt.legend()
    plt.show()

task_1_plot()