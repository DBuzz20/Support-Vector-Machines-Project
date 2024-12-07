import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from statistics import mean

#solvers.options['abstol'] = 1e-15
#solvers.options['reltol'] = 1e-15
solvers.options['show_progress'] = False

#hyperparams----------------------------------------------------------------------------
gamma=2
C=1
eps=1e-5

#DATA EXTRACTION------------------------------------------------------------
def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


cwd = os.getcwd()

X_all_labels, y_all_labels = load_mnist(cwd, kind='train')

"""
We are only interested in the items with label 1, 5 and 7.
Only a subset of 1000 samples per class will be used.
"""
indexLabel1 = np.where((y_all_labels==1))
xLabel1 =  X_all_labels[indexLabel1][:1000,:].astype('float64')
yLabel1 = y_all_labels[indexLabel1][:1000].astype('float64')

indexLabel5 = np.where((y_all_labels==5))
xLabel5 =  X_all_labels[indexLabel5][:1000,:].astype('float64')
yLabel5 = y_all_labels[indexLabel5][:1000].astype('float64')

indexLabel7 = np.where((y_all_labels==7))
xLabel7 =  X_all_labels[indexLabel7][:1000,:].astype('float64')
yLabel7 = y_all_labels[indexLabel7][:1000].astype('float64')

X_data = np.concatenate((xLabel1, xLabel5), axis=0)
Y_data = np.concatenate((yLabel1, yLabel5), axis=0)

#DATA adjustment and normalization-----------------------------------------------------
def binary_class(y):
    for i in range(len(y)):
        if y[i]==1:
            y[i]=1
        else:
            y[i]=-1
    return y

x_train, x_test, y_train, y_test  = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 1895533)

y_train=binary_class(y_train)
y_test=binary_class(y_test)

#data normalization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#functions definition----------------------------------------------------------
def pol_ker(x1,x2,gamma):
    K=(np.dot(x1,x2.T)+1)**gamma
    return K


def prediction(alfa,x1,x2,Y,gamma,epsilon,C):
    #RIVEDI
    return


def init_M(alfa, y_train, epsilon, C, Kernel):
    P = y_train.shape[0]
    Y = y_train*np.eye(P)
    Q = np.dot(np.dot(Y, Kernel), Y)
    
    fun = -(np.dot(Q, alfa) - 1) * y_train  #= - (np.dot(Q, alpha) - np.ones(P).reshape(-1, 1)) * y_train
    
    S_a = np.where(
        np.logical_or(
            np.logical_and(alfa < C-epsilon, y_train==-1), np.logical_and(alfa > epsilon ,y_train == 1)))[0]
    
    M = np.min(fun[S_a])
        
    return M


def init_m(alfa, y_train, epsilon, C, K):
    P = y_train.shape[0]
    Y = y_train*np.eye(P)
    Q = np.dot(np.dot(Y, K), Y)
    
    fun = -(np.dot(Q, alfa) - 1) * y_train  #= - (np.dot(Q, alpha) - np.ones(P).reshape(-1, 1)) * y_train
    
    R_a = np.where(
        np.logical_or(
            np.logical_and(alfa < C-epsilon, y_train==1), np.logical_and(alfa > epsilon ,y_train == -1)))[0]
    
    m = np.max(fun[R_a])
    
    return m

def main(x_train,x_test,y_train,y_test,gamma,C):
    P = y_train.shape[0]
    Kernel = pol_ker(x_train,x_train,gamma)
    Y_train = y_train * np.eye(P)
    Q_0 = np.dot(np.dot(Y_train, Kernel), Y_train)
    #Matrix definition to solve the QP
    #Objective Function
    Q = matrix(Q_0)
    e = matrix((-np.ones(P)).reshape(-1, 1))

    #Inequality constraints
    G = matrix(np.concatenate((np.eye(P), -np.eye(P)))) #vincoli
    h = matrix(np.concatenate((C*np.ones((P, 1)), np.zeros((P, 1))))) #termini noti

    #Equality constraints
    A = matrix(y_train.T)#vincoli #transposed
    b = matrix(np.array([0.])) #termini noti #np.zeros(1)

    start = time.time()
    opt = solvers.qp(Q, e, G, h, A, b, solver="cvxopt")
    run_time = time.time() - start
    
    alfa_star = np.array(opt['x'])
    n_it = opt["iterations"]
    fun_opt = (0.5 * (alfa_star.T @ Q_0 @ alfa_star) - np.sum(np.ones(P) * alfa_star))[0][0]
    
    pred_train = prediction(alfa_star,x_train,x_train,y_train,gamma,eps,C) 
    acc_train = np.sum(pred_train.ravel() == y_train.ravel())/y_train.size 

    pred_test = prediction(alfa_star,x_train,x_test,y_train,gamma,eps,C) 
    acc_test = np.sum(pred_test.ravel() == y_test.ravel())/y_test.size
    
    M = init_M(alfa_star, y_train, eps, C, Kernel)
    m = init_m(alfa_star, y_train, eps, C, Kernel) 
    
    #printing routine
    print("C value: ",C)
    print("Gamma values: ",gamma)
    
    print("Accuracy on Training set: ",acc_train)
    print("Accuracy on test set: ",acc_test)
    
    print("Time spent in optimization: ",run_time)
    print("Number of iterations: ",n_it)
    print("Starting objective function value: 0")
    print("Optimal objective function value: ",fun_opt)
    print("KKT violation: ",m-M)
    
    return