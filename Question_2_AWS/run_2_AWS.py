from functions_2_AWS import *

gamma=2
C=1
eps=1e-5
tol=1e-15
q=5

#DATA adjustment and normalization-----------------------------------------------------
x_train, x_test, y_train, y_test  = train_test_split(x_data, y_data, test_size = 0.2, random_state = 1895533)

y_train=binary_class(y_train)
y_test=binary_class(y_test)
#normalization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

alfa, run_time, M, m, cont=train(x_train,y_train,gamma,eps,C,q,tol)
#---------------------------------------------------------------------------------------


#alfa,x_train,y_train,x_test,y_test,Y_train,K,M,m,run_time,status,fun_optimum,gamma,eps,C,q,cont=train(x_train,x_test,y_train,y_test,gamma,eps,C,q,tol)
printing_routine(alfa,x_train,y_train,x_test,y_test,M,m,run_time,gamma,eps,C,q,cont)

#grid_search(x_train, y_train, eps,gamma,C,tol,params)