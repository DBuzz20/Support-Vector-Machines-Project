from functions_2_AWS import *


#DATA adjustment and normalization-----------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 1895533)

y_train=binary_class(y_train)
y_test=binary_class(y_test)

#normalization
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#---------------------------------------------------------------------------------------

alfa,run_time, acc_test,acc_train, obj_fun_val,cont,M,m,pred_test,status= training(x_train,x_test,y_train,y_test,gamma,eps,C,q,tol)
printing_routine(y_test,M,m,run_time , acc_test,acc_train, obj_fun_val,cont,pred_test,status)

#cross_val(q_list, x_train, y_train, eps,gamma,C,tol)