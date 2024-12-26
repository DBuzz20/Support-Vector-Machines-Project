

import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import time
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from statistics import mean
solvers.options['abstol'] = 1e-15
solvers.options['reltol'] = 1e-15
solvers.options['show_progress'] = False


gamma=2
C=1
eps=1e-5
tol=1e-12
q=80

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



def get_M(alfa, y, eps, C,grad, K,q):
    grad = grad.reshape((len(alfa), 1))
    S = np.union1d(np.where((alfa <= C-eps) & (y<0))[0], np.where((alfa >= eps) & (y >0))[0])
    M_grad=-grad[S] * y[S]
    M = np.min(M_grad)
    q2=np.argsort(M_grad.ravel())[0:int(q/2)]
    
    q2_i=np.array(S, dtype = int)[q2]
    
    return M,q2_i

def get_m(alfa, y, eps, C,grad, K,q):
    grad = grad.reshape((len(alfa), 1))
    R = np.union1d(np.where((alfa <= C-eps) & (y>0))[0], np.where((alfa >= eps) & (y <0))[0])
    m_grad = -grad[R] * y[R]
    m = np.max(m_grad)
    
    q1=np.argsort(-m_grad.ravel())[0:int(q/2)]
    
    q1_i=np.array(R, dtype = int)[q1]
    
    return m,q1_i
#sia get M che m hanno K dentro ma non lo usano (una è segnato l'altra no)
def init_Q(buffer, workers, not_workers):
    Q_workers=[]
    for i in workers:
        Q_workers.append(buffer['{}'.format(i)])
    
    
    Q_workers=np.array(Q_workers)
    
    index=np.arange(Q_workers.shape[0])
    Q_w = Q_workers[np.ix_(index, workers)]
    
    
    Q_notw = Q_workers[np.ix_(index, not_workers)]
  
    return Q_workers,Q_w, Q_notw



def training(x_train,x_test,y_train,y_test,gamma,eps,C,q,tol):
    index_array = np.arange(x_train.shape[0])
    y_train=y_train.reshape(len(y_train),1)
    K=pol_ker(x_train,x_train,gamma)
    Y_train=y_train*np.eye(len(y_train))
    
    alfa=np.zeros((x_train.shape[0], 1))
    grad = -np.ones((len(alfa), 1))
    
    m, m_i = get_m(alfa, y_train, eps, C,grad, K,q)
    M , M_i = get_M(alfa, y_train, eps, C,grad, K,q)
    buffer={}
    
    start=time.time()
    n_it = 0
    while (m - M) >= tol:
        
        w = np.sort(np.concatenate((m_i, M_i)))
        
        for i in w:
            if i not in buffer.keys():
                Ke=pol_ker(x_train[i],x_train,gamma)
                colonna=y_train[i]*(Ke@Y_train)#calcolo colonna
            
                buffer['{}'.format(i)]=colonna
        
        not_w = np.delete(index_array, w)
        
        Q_workers,Q_w,Q_notw = init_Q(buffer, w, not_w)
        
        not_var = alfa[not_w]
        
        
        P=matrix(Q_w)
        
        
    
        e=matrix((Q_notw @ not_var)- np.ones((len(w),1)))
        A=matrix(y_train[w].T)
        b=matrix(-(y_train[not_w].T @ not_var))
        G=matrix(np.concatenate((np.eye(len(w)),-np.eye(len(w)))))
        h=matrix(np.concatenate((C*np.ones((len(w),1)),np.zeros((len(w),1)))))
    
    
    
        sol = solvers.qp(P,e, G, h, A, b)
    
    
        alfa_star = np.array(sol['x'])
        n_it += sol['iterations']
        
        grad = grad + Q_workers.T@ (alfa_star - alfa[w])
        alfa[w] = alfa_star
        m, m_i = get_m(alfa, y_train, eps, C,grad, K,q)
        M , M_i = get_M(alfa, y_train, eps, C,grad, K,q)
    end = time.time()
    
    
    #we have calculated the entire Q only to compute the FOB as requested in the instructions
    Q=((Y_train@K)@Y_train)
    FO=1/2*(((alfa.T@Q)@alfa))-(np.ones((1,len(alfa)))@alfa)
    
    
    pred_train = prediction(alfa,x_train,x_train,y_train,gamma,eps,C) 
    acc_train = np.sum(pred_train.ravel() == y_train.ravel())/y_train.size 

    pred_test = prediction(alfa,x_train,x_test,y_train,gamma,eps,C) 
    acc_test = np.sum(pred_test.ravel() == y_test.ravel())/y_test.size 
    run_time= end-start
    #print(len(buffer))
    
    
    return alfa,run_time, acc_test,acc_train, FO,n_it,M,m,pred_test,pred_train


def printing_routine(M,m,run_time, acc_test,acc_train, FO,n_it,pred_test,pred_train):
    print("C value: ",C)
    print("Gamma values: ",gamma)
    print("q value:",q)
    print()
    print("Accuracy on Training set: %.3f" %acc_train)
    print("Accuracy on test set: %.3f" %acc_test)
    print()
    print("Time spent in optimization: ",run_time)
    print("Number of iterations: ",n_it)
    print("Optimal objective function value: ",FO)
    print('max KKT Violation:', M-m)
    
    
    cm = confusion_matrix(y_test.ravel(), pred_test.ravel()) 
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,5])
    disp.plot()
    plt.show() 
  

def cross_val(q_list, x_train, x_test, y_train, y_test, eps,gamma,C,tol):
    
    kf = KFold(n_splits=5)
    best_acc_valid = 0
    acc_list = []
    tot_iter = len(q_list)
    num = 1
    y_train=y_train.reshape(len(y_train),1)
    
    for q in q_list:
            
        print ("iteration",num, "over", tot_iter )

        for i, j in kf.split(x_train):
            x_train_cv, x_valid = x_train[i], x_train[j]
            y_train_cv, y_valid = y_train[i], y_train[j]
            
            x_train_cv,x_valid=normalization(x_train_cv,x_valid)
            index_array = np.arange(x_train_cv.shape[0])
            y_train_cv=y_train_cv.reshape(len(y_train_cv),1)
            K=pol_ker(x_train_cv,x_train_cv,gamma)
            Y_train_cv=y_train_cv*np.eye(len(y_train_cv))
            
            alfa=np.zeros((x_train_cv.shape[0], 1))
            grad = -np.ones((len(alfa), 1))


            Q=(Y_train_cv@K)@Y_train_cv

            m, m_ind = get_m(alfa, y_train_cv, eps, C,grad, K,q)
            M , M_ind = get_M(alfa, y_train_cv, eps, C,grad, K,q)

            start = time.time()
            cont = 0
            while (m - M) >= tol and cont<= 5000:
                #print(m-M)
                w = np.sort(np.concatenate((m_ind, M_ind)))

                not_w = np.delete(index_array, w)

                Q_w,Q_notw = init_Q(Q, w, not_w)

                not_var = alfa[not_w]


                Qw_m=matrix(Q_w)



                e=matrix(np.dot(Q_notw, not_var)- np.ones((len(w),1)))
                y=matrix(y_train_cv[w].T)
                zero_ug=matrix(-np.dot(y_train_cv[not_w].T, not_var))
                plus_minus_one=matrix(np.concatenate((np.eye(len(w)),-np.eye(len(w)))))
                C_zero=matrix(np.concatenate((C*np.ones((len(w),1)),np.zeros((len(w),1)))))

                sol = solvers.qp(Qw_m,e, plus_minus_one, C_zero, y, zero_ug)


                alfa_star = np.array(sol['x'])
                cont += sol['iterations']

                grad = grad + (Q[w].T@ (alfa_star - alfa[w]))
                alfa[w] = alfa_star
                m, m_ind = get_m(alfa, y_train_cv, eps, C,grad, K,q)
                M , M_ind = get_M(alfa, y_train_cv, eps, C,grad, K,q)
            end = time.time()
            prediction_valid = prediction(alfa,x_train_cv,x_valid,y_train_cv,gamma,eps,C) 
            Accuracy_validation = np.sum(prediction_valid.ravel() == y_valid.ravel())/y_valid.size 
            print(Accuracy_validation)
            acc_list.append(Accuracy_validation)
            

        num += 1


        
        print(mean(acc_list), q)
        if mean(acc_list) > best_acc_valid:
            best_acc_valid = mean(acc_list)
            best_q = q


            acc_list = []

    print("Best q: ", best_q)
    
    print("Best Median Accuracy: ", best_acc_valid)


def workers_selection(workers_list,x_train,x_test,y_train,gamma,epsilon,C,tol):
    
    mean_times = []

    for w in workers_list:
    
        times = []
    
        for i in range(10):
        
            alfa,tim = training_buffer(x_train,x_test,y_train,gamma,epsilon,C,w,tol)
            times.append(tim)
        
        mean_time =mean(times)
        mean_times.append(mean_time)
    
    print(mean_times)

def get_Q(Q, workers, not_workers):
    
    Q_w = Q[np.ix_(workers, workers)]
    Q_notw = Q[np.ix_(workers, not_workers)]
    
    return Q_w, Q_notw


"""
def training_Q(x_train,x_test,y_train,gamma,epsilon,C,q,tol):
    x_train,x_test=normalization(x_train,x_test)
    index_array = np.arange(x_train.shape[0])
    y_train=y_train.reshape(len(y_train),1)
    K=pol_ker(x_train,x_train,gamma)
    Y_train=y_train*np.eye(len(y_train))
    
    
    alfa=np.zeros((x_train.shape[0], 1))
    grad = -np.ones((len(alfa), 1))
    
    'I compute Q only once before the cycle' 
    Q=np.dot(np.dot(Y_train,K),Y_train)
    
    m, m_ind = get_m(alfa, y_train, epsilon, C,grad, K,q)
    M , M_ind = get_M(alfa, y_train, epsilon, C,grad, K,q)
    
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
    
    
    
        sol = solvers.qp(P,e, G, h, A, b)
    
    
        alfa_star = np.array(sol['x'])
        cont += sol['iterations']
        
        grad = grad + (Q[w].T @ (alfa_star - alfa[w]))
        alfa[w] = alfa_star
        m, m_ind = get_m(alfa, y_train, epsilon, C,grad, K,q)
        M , M_ind = get_M(alfa, y_train, epsilon, C,grad, K,q)
    end = time.time()
    
    FOB=1/2*(((alfa.T@Q)@alfa))-(np.ones((1,len(alfa)))@alfa)
    
    
    pred_train = prediction(alfa,x_train,x_train,y_train,gamma,epsilon,C) 
    acc_train = np.sum(pred_train.ravel() == y_train.ravel())/y_train.size 

    pred_test = prediction(alfa,x_train,x_test,y_train,gamma,epsilon,C) 
    acc_test = np.sum(pred_test.ravel() == y_test.ravel())/y_test.size 
    
    print('Test Accuracy:' ,acc_test)
    print('Training Accuracy:', acc_train)
    print('Initial value of the objective function :',0)
    #print('Final value of the objective function:', float(final_value))
    #print('M(alfa) =', M)
    #print('m(alfa) =', m)
    print('KKT Violation:', M-m)
    print('Value chosen for C:' ,C)
    print('Value chosen for gamma:' ,gamma)
    print('Time to optimize:', end-start)
    print('Number of workers chosen:', q)
    print('Number of iterations', cont)
    print('KKT Violation:', M-m)
    print('Final value of the objective function',FOB)
    
    print('\n')
    
    cm = confusion_matrix(y_test.ravel(), pred_test.ravel()) 
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,5])
    disp.plot()
    plt.show()
    
    
    return alfa,end-start
"""
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 1895533)

y_train=binary_class(y_train)
y_test=binary_class(y_test)


