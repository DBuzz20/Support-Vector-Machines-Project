import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import time
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


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


def get_M(alfa, y, eps, C,grad, q):
    grad = grad.reshape((len(alfa), 1))
    S = np.union1d(np.where((alfa <= C-eps) & (y<0))[0], np.where((alfa >= eps) & (y >0))[0])
    #print(len(alfa))
    #print(len(grad))
    #print(len(S))
    #print(len(y))
    S = S[S < len(grad)]
    M_grad=-grad[S]* y[S]
    #print(len(M_grad))
    M = np.min(M_grad)
    
    q2=np.argsort(M_grad.ravel())[0:int(q/2)]
    
    q2_ind=np.array(S, dtype = int)[q2]
        
    return M,q2_ind

def get_m(alfa, y, eps, C,grad, q):
    grad = grad.reshape((len(alfa), 1))
    R = np.union1d(np.where((alfa <= C-eps) & (y>0))[0], np.where((alfa >= eps) & (y <0))[0])
    R = R[R < len(grad)]
    m_grad = -grad[R] * y[R]
    m = np.max(m_grad)
    
    q1=np.argsort(-m_grad.ravel())[0:int(q/2)]
    
    q1_ind=np.array(R, dtype = int)[q1]
    
    return m,q1_ind

def init_Q(buff, work, not_w):
    Q_work=[]
    for i in work:
        Q_work.append(buff['{}'.format(i)])
    
    Q_work=np.array(Q_work)
    
    index=np.arange(Q_work.shape[0])
    Q_w = Q_work[np.ix_(index, work)]
    
    
    Q_notw = Q_work[np.ix_(index, not_w)]
  
    return Q_work,Q_w, Q_notw

def train(X_train,y_train,gamma,epsilon,C,q,tol):
    solvers.options['abstol'] = 1e-15
    solvers.options['reltol'] = 1e-15
    solvers.options['feastol']= 1e-15
    solvers.options['show_progress'] = False
    
    index_array = np.arange(X_train.shape[0])
    y_train=y_train.reshape(len(y_train),1)
    K=pol_ker(X_train,X_train,gamma)
    Y_train=y_train*np.eye(len(y_train))
    
    alfa=np.zeros((X_train.shape[0], 1))
    grad = -np.ones((len(alfa), 1))
    
    m, m_ind = get_m(alfa, y_train, epsilon, C,grad,q)
    M , M_ind = get_M(alfa, y_train, epsilon, C,grad,q)
    buffer={}
    
    start=time.time()
    cont = 0
    while (m - M) >= tol:
        
        w = np.sort(np.concatenate((m_ind, M_ind)))
        
        for i in w:
            if i not in buffer.keys():
                Ke=pol_ker(X_train[i],X_train,gamma)
                colonna=y_train[i]*np.dot(Ke,Y_train)#calcolo colonna
            
                buffer['{}'.format(i)]=colonna
        
        not_w = np.delete(index_array, w)
        
        Q_workers,Q_w,Q_notw = init_Q(buffer, w, not_w)
        
        not_var = alfa[not_w]
        
        
        P=matrix(Q_w)
        
        
    
        e=matrix(np.dot(Q_notw, not_var)- np.ones((len(w),1)))
        A=matrix(y_train[w].T)
        b=matrix(-np.dot(y_train[not_w].T, not_var))
        G=matrix(np.concatenate((np.eye(len(w)),-np.eye(len(w)))))
        h=matrix(np.concatenate((C*np.ones((len(w),1)),np.zeros((len(w),1)))))
    
    
    
        sol = solvers.qp(P,e, G, h, A, b)
    
    
        alfa_star = np.array(sol['x'])
        cont += sol['iterations']
        
        grad = grad + np.dot(Q_workers.T, (alfa_star - alfa[w]))
        alfa[w] = alfa_star
        m, m_ind = get_m(alfa, y_train, epsilon, C,grad,q)
        M , M_ind = get_M(alfa, y_train, epsilon, C,grad,q)
        
    end = time.time()
    run_time= end - start
    
    return alfa, run_time, M, m, cont,K

def printing_routine(X_train,X_test,y_train,y_test,gamma,epsilon,C,q,alfa,run_time,M, m,cont,K):
    y_train=y_train.reshape(len(y_train),1)
    Y_train=y_train*np.eye(len(y_train))
    
    #we have calculated the entire Q only to compute the FOB as requested in the instructions
    Q=np.dot(np.dot(Y_train,K),Y_train)
    FOB=1/2*(np.dot(np.dot(alfa.T,Q),alfa))-np.dot(np.ones((1,len(alfa))),alfa)
    
    
    pred_train = prediction(alfa,X_train,X_train,y_train,gamma,epsilon,C) 
    acc_train = np.sum(pred_train.ravel() == y_train.ravel())/y_train.size 

    pred_test = prediction(alfa,X_train,X_test,y_train,gamma,epsilon,C) 
    acc_test = np.sum(pred_test.ravel() == y_test.ravel())/y_test.size 
    
    #print(len(buffer))
    print('Test Accuracy:' ,acc_test)
    print('Training Accuracy:', acc_train)
    
    print('Initial value of the objective function :',0)
    print('Final value of the objective function:', FOB)
    print('Value chosen for C:' ,C)
    print('Value chosen for gamma:' ,gamma)
    print('Time to optimize:', run_time)
    print('Number of workers chosen:', q)
    print('Number of iterations', cont)
    print('KKT Violation:', M-m)
    
    
    print('\n')
    
    cm = confusion_matrix(y_test.ravel(), pred_test.ravel()) 
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,5])
    disp.plot()
    plt.show()
    
    return
    
