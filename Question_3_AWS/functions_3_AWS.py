import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import time
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


gamma=2
C=1
q=2
eps=1e-5
tol=1e-16

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

x_data=np.concatenate((xLabel1,xLabel5))
y_data=np.concatenate((yLabel1,yLabel5))


def binary_class(y):
    for i in range(len(y)):
        if y[i]==1:
            y[i]=1
        else:
            y[i]=-1
    return y
            
def pol_ker(x1, x2, gamma):
    k=(x1 @ x2.T +1)**gamma
    return k

def prediction(alfa,x1,x2,y,gamma,C,eps):
    SV=0 
    for i in range(len(alfa)):
        if alfa[i]>=eps and alfa[i]<=C-eps:
            SV+=1
    if SV==0:
        print("No SV found")
        b=0
    else:      
        Kb=pol_ker(x1,x1[SV].reshape(1,x1.shape[1]),gamma)
        b=np.mean(y[SV] - ((alfa*y.reshape(-1,1)).T @ Kb))
    
    K=pol_ker(x1,x2,gamma)
    pred=((alfa*y.reshape(-1,1)).T @ K ) + b
    pred=np.sign(pred)

    return pred

"""
def get_M(alfa, y, C,eps,grad, q):
    grad = grad.reshape((len(alfa), 1))
    S = np.union1d(np.where((alfa <= C-eps) & (y<0))[0], np.where((alfa >= eps) & (y >0))[0])[0]
    M_grad=-grad[S]* y[S]
    M = np.min(M_grad[S])
    
    q2=np.argsort(M_grad.ravel())[0:int(q/2)]
    
    q2_ind=np.array(S, dtype = int)[q2]
        
    return M,q2_ind

def get_m(alfa, y, C,eps,grad, q):
    grad = grad.reshape((len(alfa), 1))
    R = np.union1d(np.where((alfa <= C-eps) & (y>0))[0], np.where((alfa >= eps) & (y <0))[0])[0]
    
    m_grad = -grad[R] * y[R]
    m = np.max(m_grad)
    
    q1=np.argsort(-m_grad.ravel())[0:int(q/2)]
    
    q1_ind=np.array(R, dtype = int)[q1]
    
    return m,q1_ind
"""

def get_M(alfa, y, C,eps,grad, q):
    grad = grad.reshape((len(alfa), 1))
    indici = np.where(np.logical_or(np.logical_and(alfa <= C-eps, y==-1), np.logical_and(alfa >= eps ,y == 1)))[0]
    M_grad=-grad[indici] * y[indici]
    M = np.min(M_grad)
    q2=np.argsort(M_grad.ravel())[0:int(q/2)]
    
    q2_ind=np.array(indici, dtype = int)[q2]
    
    
    return M,q2_ind

def get_m(alfa, y, C,eps,grad, q):
    grad = grad.reshape((len(alfa), 1))
    
    indici = np.where(np.logical_or(np.logical_and(alfa <= C-eps, y==1), np.logical_and(alfa >= eps ,y == -1)))[0]
    m_grad = -grad[indici] * y[indici]
    m = np.max(m_grad)
    
    q1=np.argsort(-m_grad.ravel())[0:int(q/2)]
    
    
    q1_ind=np.array(indici, dtype = int)[q1]
    
    
    return m,q1_ind

def train(x_train,y_train,C,eps,q):
    P=y_train.shape[0]
    alfa=np.zeros(P).reshape(-1,1)
    gradient=np.ones(P).reshape(-1,1)
    m, i = get_m(alfa, y_train, C, eps, gradient, q)
    M, j = get_M(alfa, y_train, C, eps, gradient, q)
    kernel=pol_ker(x_train,x_train,gamma)
    Y_train=y_train*np.eye(P)
    Q_0=Y_train@kernel@Y_train
    n_it=0
    
    start=time.time()
    
    while m - M >= tol:
        W_ind=np.concatenate((i,j))
        Q_tmp=Q_0[W_ind][:,W_ind]
        
        #direction
        fdd=np.array([y_train[i],-y_train[j]]).reshape(-1,1)
        d_i=fdd[0]
        d_j=fdd[1]
        
        #linesearch
        t_star=-(gradient[W_ind].T @ fdd)/(fdd.T@Q_tmp@fdd)
        if d_i>0:
            max_i=(C-alfa[i])*d_i
        else:
            max_i=alfa[i]*np.abs(d_i)
            
        if d_j>0:
            max_j=(C-alfa[j])*d_j
        else:
            max_j=alfa[j]*np.abs(d_j)
            
        t_max=min(max_i,max_j)
        
        t_star=min(t_max,t_star)
        
        #alfa_new only has the W_indexes
        alfa_new=alfa[W_ind]+t_star*fdd
        diff=alfa_new-alfa[W_ind]
        
        gradient=gradient + (Q_0[:,W_ind] @ diff)
        
        alfa[W_ind]=alfa_new
        
        m, i = get_m(alfa, y_train, C, eps, gradient, q)
        M , j = get_M(alfa, y_train, C, eps,gradient, q)
        
        n_it += 1
        
    run_time=time.time()-start
    
    kkt_viol=m-M
    obj_fun_opt = (0.5 * (alfa.T @ Q_0 @ alfa) - np.sum(np.ones(P) * alfa))[0][0]
    
    return alfa,run_time,kernel,kkt_viol,n_it,obj_fun_opt

def printing_routine(x_train,x_test,y_train,y_test,gamma,C,eps,run_time,kernel,alfa,kkt_viol,n_it,obj_fun_opt):
    P=y_train.shape[0]
    Y_train=y_train*np.eye(P)
    pred_train = prediction(alfa,x_train,x_train,y_train,gamma,C,eps) 
    acc_train = np.sum(pred_train.ravel() == y_train.ravel())/y_train.size 

    pred_test = prediction(alfa,x_train,x_test,y_train,gamma,C,eps) 
    acc_test = np.sum(pred_test.ravel() == y_test.ravel())/y_test.size
    
    'We calculate Q only to compute the final value of the objective function'
    Q=np.dot(np.dot(Y_train,kernel),Y_train)
    FOB=1/2*(np.dot(np.dot(alfa.T,Q),alfa))-np.dot(np.ones((1,len(alfa))),alfa)
    
    print("C value: ",C)
    print("Gamma values: ",gamma)
    print("q value:",q)
    print()
    print("Accuracy on Training set: %.3f" %acc_train)
    print("Accuracy on test set: %.3f" %acc_test)
    print()
    print("Time spent in optimization: ",run_time)
    print("Number of iterations: ",n_it)
    print("Optimal objective function value 1: ",obj_fun_opt)
    print("Optimal objective function value 2: ",FOB)
    print("max KKT violation: ",kkt_viol)
    
    cm = confusion_matrix(y_test.ravel(), pred_test.ravel()) 
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[True,False])
    disp.plot()
    plt.show()
        
