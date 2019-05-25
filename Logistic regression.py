import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import preprocessing 
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_similarity_score
df = pd.read_csv("ChurnData.csv")
y = df['churn'].astype(int)
x = np.asarray(df[['tenure','age','address','income','ed','employ','equip','voice','pager','internet']])
#print(x[0:5])
x = preprocessing.StandardScaler().fit(x).transform(x)
#print(x[0:5])
msk = np.random.rand(len(df))<0.8
train_x = x[msk]
test_x = x[~msk]

train_y = y[msk]
test_y = y[~msk]

LR = LogisticRegression( C = 0.01, solver='liblinear').fit(train_x,train_y)
y_predicted = LR.predict(test_x)
print (jaccard_similarity_score(test_y,y_predicted) )