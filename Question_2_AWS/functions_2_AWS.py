import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import time
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
solvers.options['abstol'] = 1e-15
solvers.options['reltol'] = 1e-15
solvers.options['feastol'] = 1e-15
solvers.options['show_progress'] = False

gamma=2
C=1
q=66
eps = 1e-9
tol = 1e-10


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
    K=pol_ker(x1,x2,gamma)
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
    
    pred=((alfa*y.reshape(-1,1)).T @ K ) + b
    pred=np.sign(pred)

    return pred



def get_M(alfa, y, eps, C,grad,q):
    grad = grad.reshape((len(alfa), 1))
    S = np.union1d(np.where((alfa <= C-eps) & (y<0))[0], np.where((alfa >= eps) & (y >0))[0])
    M_grad=-grad[S] * y[S]
    M = np.min(M_grad)
    
    q2=np.argsort(M_grad.ravel())[0:int(q/2)]
    q2_i=np.array(S, dtype = int)[q2]
    return M,q2_i

def get_m(alfa, y, eps, C,grad,q):
    grad = grad.reshape((len(alfa), 1))
    R = np.union1d(np.where((alfa <= C-eps) & (y>0))[0], np.where((alfa >= eps) & (y <0))[0])
    m_grad = -grad[R] * y[R]
    m = np.max(m_grad)
    
    q1=np.argsort(-m_grad.ravel())[0:int(q/2)]
    q1_i=np.array(R, dtype = int)[q1]
    return m,q1_i


def split_Q(memory, ws, not_ws):
    Q_tot=[]
    for i in ws:
        Q_tot.append(memory[i])
        
    Q_tot=np.array(Q_tot)
    index=np.arange(Q_tot.shape[0])
    Q_w = Q_tot[np.ix_(index, ws)]
    Q_nw = Q_tot[np.ix_(index, not_ws)]
    return Q_tot,Q_w, Q_nw


def train(x_train,x_test,y_train,y_test,gamma,eps,C,q,tol):
    P=y_train.shape[0]
    tot_index = np.arange(P)
    y_train=y_train.reshape(P,1)
    K=pol_ker(x_train,x_train,gamma)
    Y_train=y_train*np.eye(P)
    
    alfa=np.zeros((P, 1))
    grad = -np.ones((P, 1))
    
    m, i = get_m(alfa, y_train, eps, C,grad,q)
    M , j = get_M(alfa, y_train, eps, C,grad,q)
    
    memory={}
    
    start=time.time()
    n_it = 0
    while (m - M) >= tol:
        
        W_ind = np.sort(np.concatenate((i, j)))
        
        for i in W_ind:
            if i not in memory.keys():
                K_tmp=pol_ker(x_train[i],x_train,gamma)
                #column associated with i
                col=y_train[i]*(K_tmp@Y_train)
                #save the association i->col
                memory[i]=col
        
        W_not = np.delete(tot_index, W_ind)
        
        Q_tot,Q_w,Q_nw = split_Q(memory, W_ind, W_not)
        
        Q=matrix(Q_w)
        e=matrix((Q_nw @ alfa[W_not])- np.ones((len(W_ind),1)))
        
        G=matrix(np.concatenate((np.eye(len(W_ind)),-np.eye(len(W_ind)))))
        h=matrix(np.concatenate((C*np.ones((len(W_ind),1)),np.zeros((len(W_ind),1)))))
        
        A=matrix(y_train[W_ind].T)
        b=matrix(-(y_train[W_not].T @ alfa[W_not]))
    
        opt = solvers.qp(Q,e, G, h, A, b)
    
        alfa_star = np.array(opt['x'])
        n_it += opt['iterations']
        
        diff = alfa_star - alfa[W_ind]
        grad = grad + (Q_tot.T @ diff)
        alfa[W_ind] = alfa_star
        
        m, i = get_m(alfa, y_train, eps, C,grad,q)
        M , j = get_M(alfa, y_train, eps, C,grad,q)
        #print(m)
        #print(M)  
    run_time = time.time() - start
    
    status= opt['status']

    #we have calculated the entire Q only to compute the obj_fun_val
    Q_complete=((Y_train@K)@Y_train)
    obj_fun_val=1/2*((alfa.T @ Q_complete @ alfa))-(np.ones((1,len(alfa))) @ alfa)

    pred_train = prediction(alfa,x_train,x_train,y_train,gamma,C,eps) 
    acc_train = np.sum(pred_train.ravel() == y_train.ravel())/y_train.size 

    pred_test = prediction(alfa,x_train,x_test,y_train,gamma,C,eps) 
    acc_test = np.sum(pred_test.ravel() == y_test.ravel())/y_test.size
    
    return alfa,run_time, acc_test,acc_train, obj_fun_val,n_it,M,m,pred_test,status


def printing_routine(y_test,M,m,run_time, acc_test,acc_train, obj_fun_val,n_it,pred_test,status):
    kkt_viol=m-M
    print("C value: ",C)
    print("Gamma values: ",gamma)
    print("q value:",q)
    print()
    print("Accuracy on Training set: %.3f" %acc_train)
    print("Accuracy on test set: %.3f" %acc_test)
    print()
    print("Time spent in optimization: ",run_time)
    print("Solver status: ",status)
    print("Number of iterations: ",n_it)
    print("Optimal objective function value: ",obj_fun_val[0][0])
    print('max KKT Violation:',kkt_viol)
    
    cm = confusion_matrix(y_test.ravel(), pred_test.ravel()) 
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[True,False])
    disp.plot()
    plt.show() 
  

#parametri tipo [q]-----------------------------------------------------------
params = np.arange(4, 150, step=5)

def optimum_q(params, x_train, y_train, eps,gamma,C,tol):
    kf = KFold(n_splits=5, random_state=1895533, shuffle=True)
    
    best_acc = -float("inf")
    
    avg_acc_list=[]
    
    for q in params:
        acc_train_tot = 0
        acc_test_tot = 0
        print("current q: ", q)

        for train_index, val_index in kf.split(x_train):
            x_train_fold, x_test_fold =x_train[train_index], x_train[val_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[val_index]
            
            alfa_fold= train(x_train_fold,x_test_fold,y_train_fold,y_test_fold,gamma,eps,C,q,tol)[0]
            
            pred_train = prediction(alfa_fold,x_train_fold,x_train_fold,y_train_fold,gamma,C,eps) 
            acc_train_tot += np.sum(pred_train.ravel() == y_train_fold.ravel())/y_train_fold.size 

            pred_test = prediction(alfa_fold,x_train_fold,x_test_fold,y_train_fold,gamma,C,eps) 
            acc_test_tot += np.sum(pred_test.ravel() == y_test_fold.ravel())/y_test_fold.size
        
        avg_acc_train = acc_train_tot / kf.get_n_splits()
        avg_acc_test = acc_test_tot / kf.get_n_splits()
        
        avg_acc_list.append([avg_acc_train, avg_acc_test])
        print(avg_acc_train)
        print(avg_acc_test)
            
        if avg_acc_test > best_acc:
            print("BETTER PARAMS FOUND:")
            print("q = ",q)
            best_acc = avg_acc_test
            best_params = [q]
            print(best_acc)
                       
    print("List of average accuracy = ", avg_acc_list)
    print(best_params) 
    print(best_acc)
    
    return
            


