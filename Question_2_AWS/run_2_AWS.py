#!/usr/bin/env python
# coding: utf-8

# In[62]:


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

alfa,run_time, acc_test,acc_train, FO,cont,M,m,pred_test,pred_train= training(x_train,x_test,y_train,y_test,gamma,eps,C,q,tol)
printing_routine(M,m,run_time , acc_test,acc_train, FO,cont,pred_test,pred_train)