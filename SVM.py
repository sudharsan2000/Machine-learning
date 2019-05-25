import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import preprocessing 
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import jaccard_similarity_score
df = pd.read_csv("cell_samples.csv")
y = np.asarray( df['Class'].astype(int) )

df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]
df['BareNuc'] = df['BareNuc'].astype('int')
print(df.dtypes)
x = np.asarray(df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc','BlandChrom', 'NormNucl', 'Mit']])
y = np.asarray( df['Class'].astype(int) )

msk = np.random.rand(len(df))<0.8
train_x = x[msk]
test_x = x[~msk]

train_y = y[msk]
test_y = y[~msk]

clf = svm.SVC(gamma='scale', kernel='rbf')
clf.fit(train_x,train_y)

y_predicted = clf.predict(test_x)

from sklearn.metrics import f1_score
print(f1_score(test_y, y_predicted, average='weighted')) 