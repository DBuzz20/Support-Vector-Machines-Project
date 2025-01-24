import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import time
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score

solvers.options['abstol'] = 1e-15
solvers.options['reltol'] = 1e-15
solvers.options['show_progress'] = False

gamma=2
C=1
eps=1e-9

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

x_data=np.concatenate((xLabel1,xLabel5,xLabel7))
y_data=np.concatenate((yLabel1,yLabel5,yLabel7))


def multi_bin_class(y,label):
    res=np.zeros(len(y))
    for i in range(len(y)):
        if y[i]==label:
            res[i]=1
        else:
            res[i]=-1
    return res

def get_accuracy(y):
    res = np.zeros((len(y),1))
    for i in range(len(y)):
        if y[i] == 1:
            res[i] = 0
        elif y[i] == 5:
            res[i] = 1
        elif y[i] == 7:
            res[i] = 2
    return res

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
    #print("number of SV:" ,SV)

    return pred

def get_predicted_label(pred):
    return pred.argmax(axis=0)

def get_M(alfa, y, eps, C, K):
    Y = np.eye(len(y))*y
    Q = (Y @ K)@ Y
    M_grad = -(Q @ alfa - 1) * y
    #S = np.where(np.logical_or(np.logical_and(alfa <= C-epsilon, y==-1), np.logical_and(alfa >= epsilon ,y == 1)))
    S = np.union1d(np.where((alfa <= C-eps) & (y<0))[0], np.where((alfa >= eps) & (y >0))[0])
    M = np.min(M_grad[S])
        
    return M

def get_m(alfa, y, eps, C, K):
    Y = np.eye(len(y))*y
    Q = (Y @ K) @Y
    m_grad = -(Q @ alfa - 1) * y
    #R = np.where(np.logical_or(np.logical_and(alfa <= C-epsilon, y==1), np.logical_and(alfa >= epsilon ,y == -1)))
    R = np.union1d(np.where((alfa <= C-eps) & (y>0))[0], np.where((alfa >= eps) & (y <0))[0])
    m = np.max(m_grad[R])
    
    return m


def OAA(x_train, y_train, gamma, C, eps,label):
    y_train_multi = multi_bin_class(y_train, label)
    y_train_multi=y_train_multi.reshape(len(y_train),1)
    Y_train = y_train_multi*np.eye(len(y_train))
    P=Y_train.shape[0]
    K = pol_ker(x_train, x_train, gamma)
    
    Q = matrix(Y_train @ K @ Y_train)
    e = matrix(-np.ones(P))
    
    G = matrix(np.concatenate(((np.eye(P)), -np.eye(P))))
    h = matrix(np.concatenate((C*np.ones((P,1)), np.zeros((P,1)))))
    
    A = matrix(y_train_multi.T)
    b = matrix(np.zeros(1))
    
    start = time.time()
    opt = solvers.qp(Q,e, G, h, A, b)
    run_time= time.time() - start
    
    alfa_star = np.array(opt['x'])
    
    pred_train = prediction(alfa_star,x_train,x_train,y_train_multi,gamma,C,eps)
    
    iterations=opt['iterations']
    M = get_M(alfa_star, y_train_multi, eps, C, K)
    m = get_m(alfa_star, y_train_multi, eps, C, K)
    kkt_viol=m-M
    obj_fun_val=1/2*(alfa_star.T @ Q @ alfa_star)-np.ones((1,len(alfa_star))) @ alfa_star
    
    print('-----------------------------------------------------------------------------------')
    print('RESULTS OF LABEL',label,'AGAINST ALL')
    print()
    print('Time to optimize:', run_time)
    print('KKT Violation:',kkt_viol)
    print('Objective function value:',obj_fun_val[0][0])
    print('Number of iterations:', iterations)
    
    return pred_train, alfa_star, y_train_multi, run_time,iterations


def train(x_train,x_test,y_train,y_test, gamma,C,eps):
    
    pred_train1, alfa_star1, y_train1 , time1,n_it1 = OAA(x_train, y_train, gamma, C, eps, 1)
    pred_train5, alfa_star5, y_train5 , time5,n_it5 = OAA(x_train, y_train, gamma, C, eps, 5)
    pred_train7, alfa_star7, y_train7, time7,n_it7 = OAA(x_train, y_train, gamma, C, eps, 7)
    
    time_tot=time1+time5+time7
    it_tot=n_it1+n_it5+n_it7
    
    tot_pred_train = np.concatenate((pred_train1, pred_train5, pred_train7))
    
    #get predicted label for train set
    pred_label_train = get_predicted_label(tot_pred_train)
    
    y_train_acc = get_accuracy(y_train)
    acc_train = accuracy_score(y_train_acc.ravel(), pred_label_train.ravel())
    
    pred_test1 = prediction(alfa_star1,x_train,x_test,y_train1,gamma,C,eps)
    pred_test5 = prediction(alfa_star5,x_train,x_test,y_train5,gamma,C,eps)
    pred_test7 = prediction(alfa_star7,x_train,x_test,y_train7,gamma,C,eps)
    
    tot_pred_test = np.concatenate((pred_test1, pred_test5, pred_test7))
    
    #get predicted labels for test set
    pred_label_test = get_predicted_label(tot_pred_test)
    
    y_test_acc = get_accuracy(y_test)
    acc_test = accuracy_score(y_test_acc.ravel(), pred_label_test.ravel())
    
    print('===================================================================================')
    print('Value chosen for C:' ,C)
    print('Value chosen for gamma:' ,gamma)
    print('Multiclass strategy: OAA')
    print()
    print('Train accuracy: %.3f'%acc_train)
    print('Test accuracy: %.3f'%acc_test)
    print()
    print('Total training time:', time_tot)
    print('Total number of iterations:',it_tot)
    
    cm = confusion_matrix(y_test_acc.ravel(), pred_label_test.ravel())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,5,7])
    disp.plot()
    plt.show()
    return acc_test
    

params=[np.array([1,2,3,4,5,10,15,20,25,50,100]),np.arange(2,8,step=1)]

#computationally very heavy (1 min per fold - 5 min per [C,gamma] pair)
def grid_search(x_train, y_train,params,eps):
    kf = KFold(n_splits=5)
    best_acc = 0
    avg_acc_list=[]
    
    for C in params[0]:
        for gamma in params[1]:
            
            acc_test_tot=0
            print("Current hyperparameters => C: ",C,"\tgamma: ",gamma)
            for train_index, test_index in kf.split(x_train):
                
                x_train_fold, x_test_fold = x_train[train_index], x_train[test_index]
                y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
                
                acc_test_tot += train(x_train_fold,x_test_fold,y_train_fold,y_test_fold,gamma,C, eps)
                
            avg_acc_test=acc_test_tot/kf.get_n_splits()    
            
            avg_acc_list.append([avg_acc_test])
            print(avg_acc_test)
            
            if avg_acc_test > best_acc:
                print('BETTER PARAMS FOUND:')
                print('C= ',C)
                print('gamma= ',gamma)
                best_acc = avg_acc_test
                best_params=[C,gamma]
                print(best_acc)
                
    print("List of average accuracy = ", avg_acc_list)
    print(best_params) 
    print(best_acc)
    
    return