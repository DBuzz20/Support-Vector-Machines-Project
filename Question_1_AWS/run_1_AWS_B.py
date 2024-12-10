from functions_1_AWS_B import *

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
alfa_star,run_time,opt,Kernel,Q_0=train(x_train,y_train,gamma,C,P)

printing_routine(x_train,x_test,y_train,y_test,gamma,C,eps,run_time,opt,Kernel,alfa_star)

"""
grid_search(x_train,y_train,eps,params)
