import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.random import seed
from numpy.random import rand

p_train = pd.read_csv('train.txt', header=None, sep=',', dtype='float64')
train_arr = p_train.values
len_train = train_arr[:, 0].size

class_1 = []
class_2 = []


for i in range(len_train):
    if train_arr[i, 2] == 1:
        class_1.extend([train_arr[i, 0:2]])
    else:
        class_2.extend([train_arr[i, 0:2]])

class_1 = np.array(class_1)
class_2 = np.array(class_2)

x1 = class_1[:, 0]
y1 = class_1[:, 1]
x2 = class_2[:, 0]
y2 = class_2[:, 1]

plt.scatter(x1, y1, color='red', marker='+',label='class_1')
plt.scatter(x2, y2, color='green', marker='*',label='class_2')
plt.legend()
plt.show()



p_test = pd.read_csv('test.txt', header=None, sep=',', dtype='float64')
test_arr = p_test.values
len_test = test_arr[:, 0].size

testpoint = []

distance_1  =np.zeros((len_test, len_train))
for i in range(len_test):
    x3 = test_arr[i, 0]
    y3 = test_arr[i, 1]
    testpoint.extend([x3,y3])
    for j in range(len_train):
        distance_1[i,j] = ((x3-train_arr[j,0])*(x3-train_arr[j,0])) +((y3-train_arr[j,1])*(y3-train_arr[j,1]))
    
testpoint = np.array(testpoint)
testpoint = testpoint.reshape(-2,2)
 
index = np.zeros((len_test, len_train))
index = np.argsort(distance_1)
distance_1 = np.sort(distance_1,axis=1)         


k  = int(input('enter the value of k : '))
dist1  =np.zeros((len_test, k))
index1  =np.zeros((len_test, k))
predict  =np.zeros((len_test, k))

class_1_test = []
class_2_test = []
count_class_1 = 0
count_class_2 = 0

trainpoint = []


for i in range(len_test):
     for l in range(k):
            dist1[i,l] = distance_1[i,l]
            index1[i,l] = index[i,l]      
            predict[i,l] = train_arr[index[i,l],2]
            trainpoint.extend([train_arr[index[i,l], 0:3]])
            if( predict[i,l] == 1):
               count_class_1  = count_class_1+1
            if( predict[i,l] == 2):
               count_class_2  = count_class_2+1
            
     if(count_class_1>count_class_2):
        class_1_test.extend(test_arr[i,:])
        
            
     else:
        class_2_test.extend(test_arr[i,:])
     count_class_1 = 0
     count_class_2 = 0 


trainpoint = np.array(trainpoint)

class_1_test = np.array(class_1_test)
class_2_test = np.array(class_2_test)
class_1_test = class_1_test.reshape(-2,2)
class_2_test = class_2_test.reshape(-2,2)

x1 = class_1_test[:, 0]
y1 = class_1_test[:, 1]
x2 = class_2_test[:, 0]
y2 = class_2_test[:, 1]

plt.scatter(x1, y1, color='red', marker='+',label='class_1')
plt.scatter(x2, y2, color='green', marker='*',label='class_2')
plt.legend()
plt.show()
             
k=0
count_1 = 0
count_2 = 0
f = open('predict.txt', 'a')
for i in range(9):
    f.write("test point : " + str(testpoint[i,:]) + "\n")
    
    f.write("distance " + str(trainpoint[k,0:2]) + "   class : " + str(trainpoint[k,2]) + "\n")
    f.write("distance " + str(trainpoint[k+1,0:2]) + "   class : " + str(trainpoint[k+1,2]) + "\n")
    f.write("distance " + str(trainpoint[k+2,0:2]) + "   class : " + str(trainpoint[k+2,2]) + "\n")
    if(trainpoint[k,2] ==1):
        
        count_1  = count_1+ 1
    if(trainpoint[k+1,2] ==1):
       count_1  = count_1+ 1
    if(trainpoint[k+2,2] ==1):
        count_1  = count_1+ 1
    if(trainpoint[k,2] ==2):
        count_2 = count_2 + 1
    if(trainpoint[k+1,2] ==2):
        count_2 = count_2 + 1
    if(trainpoint[k+2,2] ==2):
        count_2 = count_2 + 1
    if(count_1>count_2):
        f.write("predicted class : 1 " + "\n")
    else:
        f.write("predicted class : 2 " + "\n")
        
    k=k+3          
    count_1 = 0
    count_2 = 0   
       
f.close()

            

   



