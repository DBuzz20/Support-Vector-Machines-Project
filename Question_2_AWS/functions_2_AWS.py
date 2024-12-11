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
"""
solvers.options['abstol'] = 1e-15
solvers.options['reltol'] = 1e-15
"""
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

#functions definition---------------------------------------------
def binary_class(y):
    for i in range(len(y)):
        if y[i]==1:
            y[i]=1
        else:
            y[i]=-1
    return y


def pol_ker(X, xp, gamma):
    K=(np.dot(X,xp.T)+1)**gamma
    return K


def prediction(alfa,X,x,Y,gamma,epsilon,C):
    K=pol_ker(X,x,gamma)
    sv=0
    for i in range(len(alfa)):
        if alfa[i]>=epsilon and alfa[i]<=C-epsilon:
            sv=i
            break    
    Kb=pol_ker(X,X[sv].reshape(1,X.shape[1]),gamma)
    pred=np.dot((alfa*Y.reshape(-1,1)).T,K)+Y[sv]-np.dot((alfa*Y.reshape(-1,1)).T,Kb)
    pred=np.sign(pred)
    return pred


def init_M(alfa, y, epsilon, C,grad, K,q):
    grad = grad.reshape((len(alfa), 1))
    S = np.where(np.logical_or(np.logical_and(alfa <= C-epsilon, y==-1), np.logical_and(alfa >= epsilon ,y == 1)))[0]
    M_grad=-grad[S] * y[S]
    M = np.min(M_grad)
    q2=np.argsort(M_grad.ravel())[0:int(q/2)]
    
    q2_ind=np.array(S, dtype = int)[q2]
    
    return M,q2_ind


def init_m(alfa, y, epsilon, C,grad, K,q):
    grad = grad.reshape((len(alfa), 1))
    
    R = np.where(np.logical_or(np.logical_and(alfa <= C-epsilon, y==1), np.logical_and(alfa >= epsilon ,y == -1)))[0]
    m_grad = -grad[R] * y[R]
    m = np.max(m_grad)
    
    q1=np.argsort(-m_grad.ravel())[0:int(q/2)]
    
    q1_ind=np.array(R, dtype = int)[q1]
    
    return m,q1_ind


def init_Q(buffer, workers, not_workers):
    Q_workers=[]
    for i in workers:
        Q_workers.append(buffer['{}'.format(i)])
    
    Q_workers=np.array(Q_workers)
    
    index=np.arange(Q_workers.shape[0])
    Q_w = Q_workers[np.ix_(index, workers)]
    Q_notw = Q_workers[np.ix_(index, not_workers)]
  
    return Q_workers,Q_w, Q_notw


def get_Q(Q, workers, not_workers):
    
    Q_w = Q[np.ix_(workers, workers)]
    Q_notw = Q[np.ix_(workers, not_workers)]
    
    return Q_w, Q_notw

def init_train(x_train,y_train,gamma,C,eps,q):
    index_array = np.arange(x_train.shape[0])
    y_train=y_train.reshape(len(y_train),1)
    K=pol_ker(x_train,x_train,gamma)
    Y_train=y_train*np.eye(len(y_train))
    
    alfa=np.zeros((x_train.shape[0], 1))
    grad = -np.ones((len(alfa), 1))
    
    m, m_ind = init_m(alfa, y_train, eps, C,grad, K,q)
    M , M_ind = init_M(alfa, y_train, eps, C,grad, K,q)
    
    return index_array,Y_train,y_train,m,m_ind,M,M_ind,alfa,K,grad

"""
def training_Q(x_train,x_test,y_train,y_test,gamma,eps,C,q,tol):
    
    index_array,Y_train,y_train,m,m_ind,M,M_ind,alfa,K,grad=init_train(x_train,y_train,gamma,C,eps,q)
    
    'I compute Q only once before the cycle' 
    Q=np.dot(np.dot(Y_train,K),Y_train)
    
    start=time.time()
    cont = 0
    while (m - M) >= tol:
        w = np.sort(np.concatenate((m_ind, M_ind)))
        
        not_w = np.delete(index_array, w)

        'It takes Q calculated before and takes only the necessary elements'
        Q_w,Q_notw = get_Q(Q, w, not_w)
        
        not_var = alfa[not_w]
        
        P=matrix(Q_w)
        e=matrix(np.dot(Q_notw, not_var)- np.ones((len(w),1)))
        
        A=matrix(y_train[w].T)
        b=matrix(-np.dot(y_train[not_w].T, not_var))
        
        G=matrix(np.concatenate((np.eye(len(w)),-np.eye(len(w)))))
        h=matrix(np.concatenate((C*np.ones((len(w),1)),np.zeros((len(w),1)))))
    
        opt = solvers.qp(P,e, G, h, A, b)
    
        alfa_star = np.array(opt['x'])
        cont += opt['iterations']
        
        grad = grad + np.dot(Q[w].T, (alfa_star - alfa[w]))
        alfa[w] = alfa_star
        m, m_ind = init_m(alfa, y_train, eps, C,grad, K,q)
        M , M_ind = init_M(alfa, y_train, eps, C,grad, K,q)
        fun_optimum=opt['primal objective']
        status= opt['status']
    end = time.time()
    
    run_time=end-start
    
    obj_val_star=1/2*(np.dot(np.dot(alfa.T,Q),alfa))-np.dot(np.ones((1,len(alfa))),alfa)
    
    pred_train = prediction(alfa,x_train,x_train,y_train,gamma,eps,C) 
    acc_train = np.sum(pred_train.ravel() == y_train.ravel())/y_train.size 

    pred_test = prediction(alfa,x_train,x_test,y_train,gamma,eps,C) 
    acc_test = np.sum(pred_test.ravel() == y_test.ravel())/y_test.size 
    
    print('Value chosen for C:' ,C)
    print('Value chosen for gamma:' ,gamma)
    print('Number of workers chosen:', q)
    print()
    print("Accuracy on Training set: %.2f" %acc_train)
    print("Accuracy on test set: %.2f" %acc_test)
    print()
    print("Time spent in optimization: ",run_time)
    print("Solver status: ",status)
    print('Number of iterations: ', cont)
    print()
    print("Final objective function value: ",fun_optimum)
    print('KKT Violation:', M-m)
    print('\n')
    
    cm = confusion_matrix(y_test.ravel(), pred_test.ravel()) 
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[True,False])
    disp.plot()
    plt.show()
    
    return alfa,run_time
"""

def train(x_train,x_test,y_train,y_test,gamma,eps,C,q,tol):
    index_array,Y_train,y_train,m,m_ind,M,M_ind,alfa,K,grad=init_train(x_train,y_train,gamma,C,eps,q)
    
    buffer={}
    
    start=time.time()
    cont = 0
    while (m - M) >= tol:
        
        w = np.sort(np.concatenate((m_ind, M_ind)))
        
        for i in w:
            if i not in buffer.keys():
                Ke=pol_ker(x_train[i],x_train,gamma)
                colonna=y_train[i]*np.dot(Ke,Y_train)#calcolo colonna
            
                buffer['{}'.format(i)]=colonna
        
        #ALTERNATIVA:
        #buffer=np.dot(np.dot(Y_train_fold,K),Y_train_fold)
        
        not_w = np.delete(index_array, w)
        
        Q_workers,Q_w,Q_notw = init_Q(buffer, w, not_w)
        
        not_var = alfa[not_w]
        
        P=matrix(Q_w)
        e=matrix(np.dot(Q_notw, not_var)- np.ones((len(w),1)))
        
        A=matrix(y_train[w].T)
        b=matrix(-np.dot(y_train[not_w].T, not_var))
        
        G=matrix(np.concatenate((np.eye(len(w)),-np.eye(len(w)))))
        h=matrix(np.concatenate((C*np.ones((len(w),1)),np.zeros((len(w),1)))))

        opt = solvers.qp(P,e, G, h, A, b)
    
        alfa_star = np.array(opt['x'])
        cont += opt['iterations']
        
        grad = grad + np.dot(Q_workers.T, (alfa_star - alfa[w]))
        alfa[w] = alfa_star
        m, m_ind = init_m(alfa, y_train, eps, C,grad, K,q)
        M , M_ind = init_M(alfa, y_train, eps, C,grad, K,q)
        status= opt['status']
        fun_optimum=opt['primal objective']
    end = time.time()
    
    run_time=end-start
    return alfa,x_train,y_train,x_test,y_test,Y_train,K,M,m,run_time,status,fun_optimum,gamma,eps,C,q,cont

def printing_routine(alfa,x_train,y_train,x_test,y_test,Y_train,K,M,m,run_time,status,fun_optimum,gamma,eps,C,q,cont):   
    #we have calculated the entire Q only to compute the FOB as requested in the instructions
    Q=np.dot(np.dot(Y_train,K),Y_train)
    
    pred_train = prediction(alfa,x_train,x_train,y_train,gamma,eps,C) 
    acc_train = np.sum(pred_train.ravel() == y_train.ravel())/y_train.size 

    pred_test = prediction(alfa,x_train,x_test,y_train,gamma,eps,C) 
    acc_test = np.sum(pred_test.ravel() == y_test.ravel())/y_test.size 
    
    #print(len(buffer))
    print('Value chosen for C:' ,C)
    print('Value chosen for gamma:' ,gamma)
    print('Number of workers chosen:', q)
    print()
    print("Accuracy on Training set: %.2f" %acc_train)
    print("Accuracy on test set: %.2f" %acc_test)
    print()
    print("Time spent in optimization: ",run_time)
    print("Solver status: ",status)
    print('Number of iterations: ', cont)
    print("Final objective function value: ",fun_optimum)
    print('max KKT Violation:', M-m)
    print('\n')
    
    cm = confusion_matrix(y_test.ravel(), pred_test.ravel()) 
    
    CM = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[True,False])
    CM.plot()
    plt.show()
    
    return alfa,run_time


params=np.arange(4,40,step=2)

#params only contains q
def grid_search(x_train, y_train, eps,gamma,C,tol,params):
    kf = KFold(n_splits=5,random_state=1895533, shuffle=True)
    
    best_acc = -float("inf")
    avg_acc_list = []
    
    #y_train=y_train.reshape(len(y_train),1)
    
    for q in params:
        acc_train_tot=0
        acc_test_tot=0
            
        print("Current hyperparameters => q: ",q)

        for train_index, val_index in kf.split(x_train):
            x_train_fold, x_test_fold = x_train[train_index], x_train[val_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[val_index]

            alfa=train(x_train_fold,x_test_fold,y_train_fold,y_test_fold,gamma,eps,C,q,tol)[0]
            """
            while (m - M) >= tol and cont<= 5000:
                #print(m-M)
                w = np.sort(np.concatenate((m_ind, M_ind)))

                not_w = np.delete(index_array, w)

                Q_w,Q_notw = init_Q(Q, w, not_w)

                not_var = alfa[not_w]

                P=matrix(Q_w)

                e=matrix(np.dot(Q_notw, not_var)- np.ones((len(w),1)))
                y=matrix(y_train_fold[w].T)
                zero_ug=matrix(-np.dot(y_train_fold[not_w].T, not_var))
                plus_minus_one=matrix(np.concatenate((np.eye(len(w)),-np.eye(len(w)))))
                C_zero=matrix(np.concatenate((C*np.ones((len(w),1)),np.zeros((len(w),1)))))

                sol = solvers.qp(P,e, plus_minus_one, C_zero, y, zero_ug)

                alfa_star = np.array(sol['x'])
                cont += sol['iterations']

                grad = grad + np.dot(Q[w].T, (alfa_star - alfa[w]))
                alfa[w] = alfa_star
                m, m_ind = init_m(alfa, y_train_fold, eps, C,grad, K,q)
                M , M_ind = init_M(alfa, y_train_fold, eps, C,grad, K,q)
            """ 
            pred_train = prediction(alfa,x_train_fold,x_train_fold,y_train_fold,gamma,C,eps) 
            acc_train_tot += np.sum(pred_train.ravel() == y_train_fold.ravel())/y_train_fold.size
            pred_test = prediction(alfa,x_train_fold,x_test_fold,y_train_fold,gamma,eps,C) 
            acc_test_tot += np.sum(pred_test.ravel() == y_test_fold.ravel())/y_test_fold.size 
            
        avg_acc_train = acc_train_tot / kf.get_n_splits()
        avg_acc_test = acc_test_tot / kf.get_n_splits()
        
        avg_acc_list.append([avg_acc_train, avg_acc_test])

        if avg_acc_test > best_acc:
            print("BETTER PARAMS FOUND:")
            print("q = ",q)
            best_acc = avg_acc_test
            best_param = q
            print(best_acc)
            
    print("List of average accuracy = ", avg_acc_list)        
    print("Best q: ", best_param)  
    print("Highest Accuracy: ", best_acc)
    


def workers_selection(workers_list,x_train,x_test,y_train,gamma,epsilon,C,tol):
    mean_times = []
    for w in workers_list:
        times = []
        for i in range(10):
        
            tim = train(x_train,x_test,y_train,gamma,epsilon,C,w,tol)[1]
            times.append(tim)
        
        mean_time =mean(times)
        mean_times.append(mean_time)
    
    print(mean_times)
