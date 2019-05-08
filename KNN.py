import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

df = pd.read_csv('teleCust1000t.csv')
x = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values
y = df['custcat'].values

x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))
msk = np.random.rand(len(df)) <0.8
x_train = x[msk]
x_test = x[~msk]
y_train = y[msk]
y_test = y[~msk]

Ks =10
accarr = np.zeros((Ks-1))
for n in range(1,Ks): 
    nei = KNeighborsClassifier(n_neighbors= n).fit(x_train,y_train)
    yhat = nei.predict(x_test)
    accarr[n-1] = metrics.accuracy_score(y_test, yhat)
print(accarr.argmax()+1)
plt.plot(range(1,Ks),accarr,color = 'blue')
plt.xlabel(' K value')
plt.ylabel('accuracy')
plt.show()

