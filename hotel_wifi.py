import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
from sklearn import preprocessing , cross_validation , svm
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import neural_network
from sklearn.preprocessing import StandardScaler

ann = neural_network.MLPClassifier(solver = 'lbfgs' , shuffle = True , hidden_layer_sizes = (7,7,7,7,7) , activation = 'relu' , verbose =False , random_state = 0 ,tol = 0 ,warm_start = True , max_iter = 100000) 
# hidden_layer_sizes = <tuple>  whose ith entry represnt the no of units in ith hidden layer
scaler = StandardScaler()

df1 = pd.read_csv("D:/AI/datasets/hotelwifi_train.csv")
df2 = pd.read_csv("D:/AI/datasets/hotelwifi_test.csv")
df1=shuffle(df1)
print(df1.head(6))

#x[:,0].astype('str') # for typecasting a column 
x1 = np.array(df1.drop(['ROOM','ID'],1))
y1 = np.array(df1['ROOM'],dtype = float)
scaler.fit(x1)
x2 = np.array(df2.drop(['ID'],1))

x1_train , x1_test , y1_train , y1_test = cross_validation.train_test_split(x1 , y1 , test_size = 0.2)

#clf = LogisticRegression()

ann.fit(x1_train , y1_train)
accuracy = (ann.score(x1_test , y1_test))*100
print(accuracy)
'''
#print(x2[0,0])
y = clf.predict(x2)
x2=np.hstack((np.array(df2['ID'])[:, np.newaxis], x2))   #append a column array in the begining

for i in range(len(y)):
	print(x2[i,0] , y[i])
'''

