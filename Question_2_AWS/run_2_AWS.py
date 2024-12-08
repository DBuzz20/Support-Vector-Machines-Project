from functions_2_AWS import *

#hyperparams----------------------------------------------------------------------------
gamma=2
C=10
eps=1e-4

tol=1e-12
q=80
#DIFFERENZIAZZIONE TRA EPS(PER M E m) E EPS PER CLASSIFIER (?)

#DATA adjustment and normalization-----------------------------------------------------
x_train, x_test, y_train, y_test  = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 1895533)

y_train=binary_class(y_train)
y_test=binary_class(y_test)

#normalization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#---------------------------------------------------------------------------------------

alfa,x_train,y_train,x_test,y_test,Y_train,K,M,m,run_time,status,fun_optimum,gamma,eps,C,q,cont=train(x_train,x_test,y_train,y_test,gamma,eps,C,q,tol)
printing_routine(alfa,x_train,y_train,x_test,y_test,Y_train,K,M,m,run_time,status,fun_optimum,gamma,eps,C,q,cont)