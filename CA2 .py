#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import os


# In[2]:


os.chdir('C:/Users/User/Desktop/MAI/CA2')


# In[3]:


data = pd.read_excel("C:/Users/User/Desktop/MAI/CA2/salary.xlsx")
data.head()


# ### Model 1 

# We will first build a SLR model to predict salary (y) using years (x) as the predictor.
# Suppose any new staff who is employed by the company will earn a minimal salary of $50/hour. This means that when  x = 0 ,  y ̂=a=50 .  Then, in the SLR model, we will only need to determine slope b.

# In[77]:


b = 0 # Starting value of x
rate = 0.001 # Set learning rate (0.001)
precision = 0.0001 # Stop algorithm when absolute difference between 2 consecutive x-values is less than precision
diff = 1 # difference between 2 consecutive iterates
max_iter = 500 # set maximum number of iterations
iter = 1 # iterations counter
E = lambda b: np.mean((data["salary"] - (50 + b*data["years"]))**2)
deriv = lambda b: np.mean(2*data["years"]*(-data["salary"] + (50 + b*data["years"]))) # derivative of b

# Now Gradient Descent

while diff > precision and iter < max_iter:
    b_new = b - rate * deriv(b)
    print("Iteration ", iter, ": b-value is: ", b_new,"E(b) is: ", E(b_new) )
    diff = abs(b_new - b)
    iter = iter + 1
    b = b_new
    
print("The local minimum occurs at: ", b)


# ### Model 2

# Now we apply the SLR model where both intercept a and slope b are to be determined, when predicting salary (y) using years (x) as the predictor.

# In[19]:


import numpy as np

next_a = 60.7 # Initial starting point
next_b = 2.2 # Initial starting point
alpha = 0.001 # Learning rate
epsilon = 0.0001 # Stopping criterion constant
max_iters = 500 # Maximum number of iterations

# Partial derivatives and function
partialf_a = lambda a,b: np.mean(2*(-data["salary"] + a + b*data["years"]))
partialf_b = lambda a,b: np.mean((2*data["years"])*(-data["salary"] + a + b*data["years"]))
func = lambda a,b: np.mean((data["salary"] - (a + b*data["years"]))**2)

# Initial value of function at the starting point
next_func = func(next_a,next_b) 

#for loop to loop through the algorithm
for n in range(max_iters):
    current_a = next_a #indicate the current point
    current_b = next_b
    current_func = next_func #value of the function at the current point
    next_a = current_a-alpha*partialf_a(current_a,current_b) # update of x (find the next x value)
    next_b = current_b-alpha*partialf_b(current_a,current_b) # update of y (find the next y value)
    next_func = func(next_a,next_b) #find the value of the function at the next point
    change_func = abs(next_func-current_func) # stopping criterion: values of function converge
    print("Iteration",n+1,": a = ",next_a,", b = ",next_b,", E(a,b) = ",next_func) #print result
    if change_func<epsilon:
        break #break the for lop and iteration will end


# ### Model 3

# In[32]:


import numpy as np

next_a = 57.7 # Initial starting point
next_b = 2.2 # Initial starting point
next_c = 6.5
alpha = 0.001   # Learning rate
epsilon = 0.001 # Stopping criterion constant
max_iters = 500 # Maximum number of iterations

# Partial derivatives and function
partialf_a = lambda a,b,c: np.mean(2*(-data["salary"] + a + b*data["years"] + c*data["gender"]))
partialf_b = lambda a,b,c: np.mean((2*data["years"])*(-data["salary"] + a + b*data["years"] + c*data["gender"]))
partialf_c = lambda a,b,c: np.mean((2*data["gender"])*(-data["salary"] + a + b*data["years"] + c*data["gender"]))
func = lambda a,b,c: np.mean((data["salary"] - (a + b*data["years"] + c*data["gender"] ))**2) # original full expression of the function

# Initial value of function at the starting point
next_func = func(next_a,next_b, next_c) 

#for loop to loop through the algorithm
for n in range(max_iters):
    current_a = next_a #indicate the current point
    current_b = next_b
    current_c = next_c
    current_func = next_func #value of the function at the current point
    next_a = current_a-alpha*partialf_a(current_a,current_b, current_c) # update of x (find the next x value)
    next_b = current_b-alpha*partialf_b(current_a,current_b, current_c) # update of y (find the next y value)
    next_c = current_c-alpha*partialf_c(current_a,current_b, current_c)
    next_func = func(next_a,next_b, next_c) #find the value of the function at the next point
    change_func = abs(next_func-current_func) # stopping criterion: values of function converge
    print("Iteration",n+1,": a = ",next_a,", b = ",next_b,", c= ", next_c, ", E(a,b,c) = ",next_func) #print result
    if change_func<epsilon:
        break #break the for lop and iteration will end


# In[5]:


plt.figure(figsize=(15,8))
sns.lineplot(x="years", y=50+2.973*data["years"], data=data, label='Model 1')
sns.lineplot(x="years", y=60.698+2.171*data['years'], data=data, label='Model 2')
# sns.lineplot(x="years", y=57.7+(2.16*data['years'])+(6.5*data['gender']),hue="gender", data=data) 
sns.lineplot(x="years", y=57.7+(2.16*(data['years'])),data=data, label ='Model 3-female')
sns.lineplot(x="years", y=57.7+(2.16*data['years'])+6.5, data=data, label ='Model 3-male')
sns.scatterplot(x="years", y="salary", hue="gender", data=data)
plt.title("Regression Lines of the 3 Models In Predicting Salary")
# sns.lineplot(x="years", y=57.7+(2.16*data['years'])+(6.5*(data['gender']==0)), data=data)
# sns.lineplot(x="years", y=57.7+(2.16*data['years'])+(6.5*(data['gender']==1)), data=data)


# In[ ]:




