from functions_3_AWS import *

#DATA adjustment and normalization-----------------------------------------------------
x_train, x_test, y_train, y_test  = train_test_split(x_data, y_data, test_size = 0.2, random_state = 1895533)

y_train=binary_class(y_train)
y_test=binary_class(y_test)

#normalization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#---------------------------------------------------------------------------------------

alfa,run_time,kernel,kkt_viol,n_it,obj_fun_opt=train(x_train,y_train,C,eps,q)

printing_routine(x_train,x_test,y_train,y_test,gamma,C,eps,run_time,kernel,alfa,kkt_viol,n_it,obj_fun_opt)

#grid_search(x_train, y_train, eps,gamma,C,tol,params)