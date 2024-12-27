import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import time
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score


gamma=2
C=1
eps=1e-10
epsB=1e-5

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


def multi_binary_class(y,label):
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

def train(X_train, y_train, gamma, C, eps, label ):
    solvers.options['abstol'] = 1e-15
    solvers.options['reltol'] = 1e-15
    solvers.options['feastol']= 1e-15
    solvers.options['show_progress'] = False
    y_train_converted = multi_binary_class(y_train, label)
    y_train_converted=y_train_converted.reshape(len(y_train),1)
    Y_train = y_train_converted*np.eye(len(y_train))
    K = pol_ker(X_train, X_train, gamma)
    P = matrix(np.dot(np.dot(Y_train, K), Y_train))
    q = matrix(-np.ones(Y_train.shape[0]))
    disug = np.eye(Y_train.shape[0])
    G = matrix(np.concatenate((disug, -disug)))
    h = matrix(np.concatenate((C*np.ones((Y_train.shape[0],1)), np.zeros((Y_train.shape[0],1)))))
    A = matrix(y_train_converted.T)
    b = matrix(np.zeros(1))
    
    
    start = time.time()
    sol = solvers.qp(P,q, G, h, A, b)
    end = time.time()
    print('Time to optimize of {} against all:'.format(label), end-start)
    opt_time = end-start
    
    alfa_star = np.array(sol['x'])
    
    M = get_M(alfa_star, y_train_converted, eps, C, K)
    m = get_m(alfa_star, y_train_converted, eps, C, K)
    
    pred_train = prediction(alfa_star,X_train,X_train,y_train_converted,gamma,eps,C)
    
    print('KKT Violation of {} against all:'.format(label), M-m)
    
    FOB=1/2*(np.dot(np.dot(alfa_star.T,P),alfa_star))-np.dot(np.ones((1,len(alfa_star))),alfa_star)
    print('FOB of {} against all:'.format(label),FOB )
    iterations=sol['iterations']
    #print('Number of solver iterations:', sol['iterations'])
    return pred_train, alfa_star, y_train_converted, opt_time,iterations


def multi_class(x_train,x_test,y_train,y_test, gamma, C,epsB):
    pred_train_1, alfa_star_1, y_train_converted_1 , run_time_1,n_it1 = train(x_train, y_train, gamma, C, eps, 1)
    
    pred_train_5, alfa_star_5, y_train_converted_5 , run_time_5,n_it5 = train(x_train, y_train, gamma, C, eps, 5)
    
    pred_train_7, alfa_star_7, y_train_converted_7, run_time_7,n_it7 = train(x_train, y_train, gamma, C, eps, 7)

    time_tot=run_time_1+run_time_5+run_time_7
    it_tot=n_it1+n_it5+n_it7
     
    pred_train = np.concatenate((pred_train_1, pred_train_5, pred_train_7))
    
    class_train = pred_train.argmax(axis = 0)
    
    y_train_acc = get_accuracy(y_train)
    
    acc_train = accuracy_score(y_train_acc.ravel(), class_train.ravel())

    print('Train accuracy: ', acc_train)
    
    
    pred_test_1 = prediction(alfa_star_1,x_train,x_test,y_train_converted_1,gamma,eps,C)
    
    pred_test_5 = prediction(alfa_star_5,x_train,x_test,y_train_converted_5,gamma,eps,C)

    pred_test_7 = prediction(alfa_star_7,x_train,x_test,y_train_converted_7,gamma,eps,C)
    
    
    pred_test = np.concatenate((pred_test_1, pred_test_5, pred_test_7))
    
    
    class_test = pred_test.argmax(axis = 0)
    
    
    y_test_acc = get_accuracy(y_test)
    
    
    acc_test = accuracy_score(y_test_acc.ravel(), class_test.ravel())
    
    print('Test accuracy: ', acc_test)
    print('Value chosen for C:' ,C)
    print('Value chosen for gamma:' ,gamma)
    
    cm = confusion_matrix(y_test_acc.ravel(), class_test.ravel())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,5,7])
    disp.plot()
    plt.show()
    return acc_test