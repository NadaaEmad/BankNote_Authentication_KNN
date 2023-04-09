#!/usr/bin/env python
# coding: utf-8

# In[1106]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[1107]:


data = data = pd.read_csv('BankNote_Authentication.csv')
data.head()


# In[1108]:


# x = data[['variance', 'skewness', 'curtosis', 'entropy']]
# y = data['class']
# # x


# In[1109]:


#Normalizing features
def mean_and_std(x):
    means = []
    stds = []
    for i in x:
        std = x[i].std()
        means.append(x[i].mean())
        stds.append(std)
#         x[i] = (x[i] - x[i].mean()) / x[i].std()       
    return means, stds


# In[1110]:


def normalize(x, mean, std):
    for i,j,k in zip(x, mean, std):
        x[i] = (x[i] - j) / k
#         x.append(x[i])
        pd.concat([x, x[i]])
    return x


# In[1111]:


x_train, x_test, y_train, y_test = train_test_split(data.drop(columns=['class']), data['class'], test_size=0.3)


# In[1112]:


means, stds = mean_and_std(x_train)
# print(means)
# print(stds)


# In[1113]:


x_train = normalize(x_train, means, stds).to_numpy()

x_test = normalize(x_test, means, stds).to_numpy()

# x_train


# In[1114]:


#Euclidean Distance
def eucledian(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist


# In[1115]:


def most_frequent(arr):
    List = arr.tolist()
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency > counter):
            counter = curr_frequency
            num = i
    return num


# In[1116]:


def predict(x_train, y , x_input, k):
    op_labels = []
     
    #Loop through the Datapoints to be classified
    for item in x_input: #x_test
         
        #Array to store distances
        point_dist = []
         
        #Loop through each training Data
        for j in range(len(x_train)): 
            distances = eucledian(np.array(x_train[j,:]) , item) 
            #Calculating the distance
            point_dist.append(distances) 
        point_dist = np.array(point_dist) 
         
        #Sorting the array while preserving the index
        #Keeping the first K datapoints
        dist = np.argsort(point_dist)[:k] 
         
        #Labels of the K datapoints from above
        labels = y[dist] #0 or 1
         
        #Majority voting
        lab = most_frequent(labels)
        op_labels.append(lab)
 
    return op_labels


# In[1117]:


k=1
for i in range(9):
    y_pred = predict(x_train,y_train.to_numpy(),x_test , k)
    
    print("k value: ", k)
    correct = np.sum(np.equal(y_test.to_numpy(), y_pred))
    accuracy = correct / len(y_pred)
    print("Number of correctly classified instances : ", correct)
    print("Total number of instances : ", len(y_pred))
    print("Accuracy: ", accuracy)
    print("--------------------------------------------------")
    k +=1
    


# In[ ]:




