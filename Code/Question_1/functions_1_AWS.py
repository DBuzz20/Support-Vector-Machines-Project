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

x_data=np.concatenate((xLabel1,xLabel5))
y_data=np.concatenate((yLabel1,yLabel5))

solvers.options['abstol'] = 1e-15
solvers.options['reltol'] = 1e-15
solvers.options['feastol']= 1e-15
solvers.options['show_progress'] = False


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
    #print("number of SV:" ,SV)

    return pred

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

            
def train(x_train,y_train,gamma,C):
    
    k=pol_ker(x_train,x_train,gamma)
    Y_train=np.diag(y_train)
    P=Y_train.shape[0]
    y_train=y_train.reshape(len(y_train),1)
    
    #Matrix definition to solve the QP
    #Objective Function
    Q = matrix((Y_train @ k) @ Y_train)
    e = matrix(-np.ones(P))
    
    #Inequality constraints ( Gx <= h )
    G = matrix(np.concatenate(((np.eye(P)), -np.eye(P)))) #vincoli
    h = matrix(np.concatenate((C*np.ones((P,1)), np.zeros((P,1))))) #termini noti
    
    #Equality constraints ( A x = b )
    A = matrix(y_train.T) #vincoli
    b = matrix(0, tc = 'd') #termini noti
    
    start = time.time()
    opt = solvers.qp(Q,e, G, h, A, b)
    run_time = time.time() - start
    
    alfa_star = np.array(opt['x'])
    
    return alfa_star,run_time,opt,k

  
def printing_routine(x_train,x_test,y_train,y_test,gamma,C,eps,run_time,opt,kernel,alfa_star):
    status= opt['status']
    fun_optimum=opt['primal objective']
    n_it = opt["iterations"]
    
    y_train=y_train.reshape(len(y_train),1)  
    
    pred_train = prediction(alfa_star,x_train,x_train,y_train,gamma,C,eps) 
    acc_train = np.sum(pred_train.ravel() == y_train.ravel())/y_train.size 

    pred_test = prediction(alfa_star,x_train,x_test,y_train,gamma,C,eps) 
    acc_test = np.sum(pred_test.ravel() == y_test.ravel())/y_test.size 
    
    M = get_M(alfa_star, y_train, eps, C, kernel)
    m = get_m(alfa_star, y_train, eps, C, kernel)
    
    #printing routine
    print("C value: ",C)
    print("Gamma values: ",gamma)
    print()
    print("Accuracy on Training set: %.3f" %acc_train)
    print("Accuracy on test set: %.3f" %acc_test)
    print()
    print("Time spent in optimization: ",run_time)
    print("Solver status: ",status)
    print("Number of iterations: ",n_it)
    print("Optimal objective function value: ",fun_optimum)
    print("max KKT violation: ",m-M)
    
    cm = confusion_matrix(y_test.ravel(), pred_test.flatten())
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[True,False])
    disp.plot()
    plt.show()
    


#parametri tipo [C,gamma]-----------------------------------------------------------
params=[np.array([1,2,3,4,5,10,15,20,25,50,100]),np.arange(2,8,step=1)]
""" [1, 2]
0.99375 """

params_C=[np.array([1,2,3,4,5,10,15,20,25,30,40,50,60,75,90,100]),np.array([2])]

params_gamma=[np.array([]),np.arange(2,20,step=1)]
#-----------------------------------------------------------------------------------
    
def grid_search(x_train,y_train,eps, params): #avrei usato tutto il db x e y, ma uso x/y_train perche sono gia scalati
    kf = KFold(n_splits=5, random_state=1895533, shuffle=True)
    
    best_acc = -float("inf")
    
    avg_acc_list=[]
    
    for C in params[0]:
        for gamma in params[1]:
            acc_train_tot = 0
            acc_test_tot = 0
            print("Current hyperparameters => C: ",C,"\tgamma: ",gamma)
            
            for train_index, val_index in kf.split(x_train):
                x_train_fold, x_test_fold = x_train[train_index], x_train[val_index]
                y_train_fold, y_test_fold = y_train[train_index], y_train[val_index]
                
                alfa_star = train(x_train_fold, y_train_fold, gamma, C)[0]
                
                pred_train = prediction(alfa_star,x_train_fold,x_train_fold,y_train_fold,gamma,C,eps) 
                acc_train_tot += np.sum(pred_train.ravel() == y_train_fold.ravel())/y_train_fold.size

                pred_test = prediction(alfa_star,x_train_fold,x_test_fold,y_train_fold,gamma,C,eps) 
                acc_test_tot += np.sum(pred_test.ravel() == y_test_fold.ravel())/y_test_fold.size
            
            avg_acc_train = acc_train_tot / kf.get_n_splits()
            avg_acc_test = acc_test_tot / kf.get_n_splits()
            
            avg_acc_list.append([avg_acc_train, avg_acc_test])
            print(avg_acc_train)
            print(avg_acc_test)
            
            if avg_acc_test > best_acc:
                print("BETTER PARAMS FOUND:")
                print("C = ",C)
                print("gamma = ",gamma)
                best_acc = avg_acc_test
                best_params = [C, gamma]
                print(best_acc)
                       
    print("List of average accuracy = ", avg_acc_list)
    print(best_params) 
    print(best_acc)
    
    return