from functions_4_AWS import *

#DATA adjustment and normalization-----------------------------------------------------
x_train, x_test, y_train, y_test  = train_test_split(x_data, y_data, test_size = 0.2, random_state = 1895533)

#normalization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#---------------------------------------------------------------------------------------

multi_class(x_train,x_test,y_train,y_test,gamma, C, eps)

#printing_routine(x_train,x_test,y_train,y_test,gamma,C,eps,run_time,opt,kernel,alfa_star)