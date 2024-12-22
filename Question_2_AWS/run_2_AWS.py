from functions_2_AWS import *

#DATA adjustment and normalization-----------------------------------------------------
x_train, x_test, y_train, y_test  = train_test_split(x_data, y_data, test_size = 0.2, random_state = 1895533)

y_train=binary_class(y_train)
y_test=binary_class(y_test)

#normalization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#---------------------------------------------------------------------------------------

"""
alfa,x_train,y_train,x_test,y_test,Y_train,K,M,m,run_time,status,fun_optimum,gamma,eps,C,q,cont=train(x_train,x_test,y_train,y_test,gamma,eps,C,q,tol)
printing_routine(alfa,x_train,y_train,x_test,y_test,Y_train,K,M,m,run_time,status,fun_optimum,gamma,eps,C,q,cont)
"""
#grid_search(x_train, y_train, eps,gamma,C,tol,params)