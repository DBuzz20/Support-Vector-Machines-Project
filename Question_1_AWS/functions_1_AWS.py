import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from statistics import mean
import time

#solvers.options['abstol'] = 1e-15
#solvers.options['reltol'] = 1e-15
solvers.options['show_progress'] = False


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

#functions definition----------------------------------------------------------
def binary_class(y):
    for i in range(len(y)):
        if y[i]==1:
            y[i]=1
        else:
            y[i]=-1
    return y

def pol_ker(x1,x2,gamma):
    K=(x1 @ x2.T + 1) ** gamma
    return K

def prediction(alfa,x1,x2,y,gamma,C,eps):
    svList = []
    #find all the support vectors
    for sv in range(len(alfa)):
        if eps < alfa[sv] < C - eps:
            svList.append(sv)
            
    #b = (1 / len(svList)) * (np.sum((y[sv] - np.dot((alfa * y).T, pol_ker(x1, x2[sv], gamma).reshape(-1, 1)) for sv in svList)))
    b = np.mean([y[sv] - np.sum(alfa * y * pol_ker(x1, x2[sv], gamma)) for sv in svList])

    
    #Compute prediction
    Ker = pol_ker(x1, x2, gamma)
    pred = np.sign(np.dot((alfa * y).T, Ker) + b)

    return pred
"""

def prediction(alfa,x1,x2,y,gamma,C,eps):
    sv=0 #baseline
    for i in range(len(alfa)):
        if alfa[i]>=eps and alfa[i]<=C-eps:
            sv=i
            break
    K=pol_ker(x1,x2,gamma)   
    Kb=pol_ker(x1,x1[sv].reshape(1,x1.shape[1]),gamma)
    pred=np.dot((alfa*y.reshape(-1,1)).T,K)+y[sv]-np.dot((alfa*y.reshape(-1,1)).T,Kb)
    pred=np.sign(pred)
    return pred
"""
def init_M(alfa, y_train, eps, C, Ker,P):
    Y = y_train*np.eye(P)
    Q = np.dot(np.dot(Y, Ker), Y)
    
    fun = -(np.dot(Q, alfa) - 1) * y_train  #= - (np.dot(Q, alpha) - np.ones(P).reshape(-1, 1)) * y_train
    S_a = np.where(
        np.logical_or(
            np.logical_and(alfa < C-eps, y_train==-1), np.logical_and(alfa > eps ,y_train == 1)))[0]
    M = np.min(fun[S_a])

    return M


def init_m(alfa, y_train, eps, C, Ker,P):
    Y = y_train*np.eye(P)
    Q = np.dot(np.dot(Y, Ker), Y)
    
    fun = -(np.dot(Q, alfa) - 1) * y_train  #= - (np.dot(Q, alpha) - np.ones(P).reshape(-1, 1)) * y_train
    R_a = np.where(
        np.logical_or(
            np.logical_and(alfa < C-eps, y_train==1), np.logical_and(alfa > eps ,y_train == -1)))[0]
    m = np.max(fun[R_a])
    
    return m

def train(x_train,y_train,gamma,C,P):
    Kernel = pol_ker(x_train,x_train,gamma)
    Y_train = y_train * np.eye(P)
    Q_0 = np.dot(np.dot(Y_train, Kernel), Y_train)
    #Matrix definition to solve the QP
    #Objective Function
    Q = matrix(Q_0)
    e = matrix((-np.ones(P)).reshape(-1, 1))

    #Inequality constraints ( Gx <= h )
    G = matrix(np.concatenate((np.eye(P), -np.eye(P)))) #vincoli
    h = matrix(np.concatenate((np.ones((P, 1)) * C, np.zeros((P, 1))))) #termini noti

    #Equality constraints ( A x = b )
    A = matrix(y_train.reshape(1, -1))#vincoli #transposed
    b = matrix(np.array([0.])) #termini noti #np.zeros(1)

    start = time.time()
    opt = solvers.qp(Q, e, G, h, A, b, solver="cvxopt")
    run_time = time.time() - start
    
    alfa_star = np.array(opt['x']) 
    print("EXITING TRAIN")
    return alfa_star,run_time,opt,Kernel,Q_0
    

def printing_routine(x_train,x_test,y_train,y_test,gamma,C,eps,run_time,opt,P,Kernel,Q_0,alfa_star):
    print(alfa_star)
    status= opt['status']
    fun_optimum=opt['primal objective']
    n_it = opt["iterations"]
    fun_opt = (0.5 * (alfa_star.T @ Q_0 @ alfa_star) - np.sum(np.ones(P) * alfa_star))[0][0]
    
    pred_train = prediction(alfa_star,x_train,x_train,y_train,gamma,eps,C)
    acc_train = np.sum(pred_train.ravel() == y_train.ravel())/y_train.size

    pred_test = prediction(alfa_star,x_train,x_test,y_train,gamma,eps,C)
    acc_test = np.sum(pred_test.ravel() == y_test.ravel())/y_test.size
    
    #CM_train = confusion_matrix(y_train.ravel(), pred_train.ravel())
    #CM_test = confusion_matrix(y_test.ravel(), pred_test.ravel())
    
    M = init_M(alfa_star, y_train, eps, C, Kernel,P)
    m = init_m(alfa_star, y_train, eps, C, Kernel,P)
    
    #printing routine
    print("C value: ",C)
    print("Gamma values: ",gamma)
    print()
    print("Accuracy on Training set: %2f" %acc_train)
    print("Accuracy on test set: %2f" %acc_test)
    print()
    print("Time spent in optimization: ",run_time)
    print("Solver status: ",status)
    print("Number of iterations: ",n_it)
    #tieni solo uno
    print("Optimal objective function value: ",fun_opt)   
    print("Optimal objective function value: ",fun_optimum)
    print("max KKT violation: ",M-m)
    
    #TrainCM=ConfusionMatrixDisplay(confusion_matrix=CM_train, display_labels=["T","F"]).plot()
    #TestCM=ConfusionMatrixDisplay(confusion_matrix=CM_test, display_labels=["T","F"]).plot()
    #TrainCM.show()
    #TestCM.show()
    
    return


#parametri tipo [C,gamma]-----------------------------------------------------------
params=[np.array([1,2,3,4,5,10,15,20,25,50,100]),np.arange(2,10,step=1)]
#-----------------------------------------------------------------------------------

def grid_search(x_train,y_train,eps, params): #avrei usato tutto il db x e y, ma uso x/y_train perche sono gia scalati
    kf = KFold(n_splits=5, random_state=1895533, shuffle=True)
    
    best_acc = float("inf")
    
    avg_acc_list=[]
    
    for C in params[0]:
        for gamma in params[1]:
            acc_train = 0
            acc_test = 0
            print("Current hyperparameters => C: ",C,"\tgamma: ",gamma)
            
            for train_index, val_index in kf.split(x_train):
                x_train_fold, x_test_fold = x_train[train_index], x_train[val_index]
                y_train_fold, y_test_fold = y_train[train_index], y_train[val_index]
                
                alfa_star = train(x_train_fold, y_train_fold, gamma, C)[0]
                
                pred_train = prediction(alfa_star,x_train_fold,x_train_fold,y_train_fold,gamma,C,eps) 
                acc_train += np.sum(pred_train.ravel() == y_train_fold.ravel())/y_train_fold.size

                pred_test = prediction(alfa_star,x_train_fold,x_test_fold,y_train_fold,gamma,C,eps) 
                acc_test += np.sum(pred_test.ravel() == y_test_fold.ravel())/y_test_fold.size
            
            avg_acc_train = acc_train / kf.get_n_splits()
            avg_acc_test = acc_test / kf.get_n_splits()
            
            avg_acc_list.append([avg_acc_train, avg_acc_test])
            
            if avg_acc_test >= best_acc:
                print("BETTER PARAMS FOUND:")
                print("C = ",C)
                print("gamma = ",gamma)
                best_acc = avg_acc_test
                best_params = [C, gamma]
                
    print("List of average accuracy = ", avg_acc_list)
    print(best_params) 
    print(best_acc)
    
    return
            
            