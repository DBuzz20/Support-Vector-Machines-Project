#!/usr/bin/env python
# coding: utf-8

# In[62]:


from functions_2_AWS import *

x_train, x_test, y_train, y_test  = train_test_split(x_data, y_data, test_size = 0.2, random_state = 1895533)

y_train=binary_class(y_train)
y_test=binary_class(y_test)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print('Performances when we calculate the entire Q before the training process')
print('CLOSE THE IMAGE TO SEE THE ALTERNATIVE METHOD IMPLEMENTED')
#training_Q(x_train,x_test,y_train,gamma,eps,C,q,tol)

print('Performances when we use the buffer without computing the enitire Q')
training_buffer(x_train,x_test,y_train,gamma,eps,C,q,tol)