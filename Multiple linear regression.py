import matplotlib.pyplot as plt
import pandas as pd
#import pylab as pl
import numpy as np
from sklearn import linear_model

df = pd.read_csv("FuelConsumptionCo2.csv")
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
# viz.hist()

msk = np.random.rand(len(df))<0.8
train = df[msk]
test = df[~msk]
regr = linear_model.LinearRegression()
allx = np.asanyarray(cdf[['FUELCONSUMPTION_COMB','ENGINESIZE','CYLINDERS']])
trainx = np.asanyarray(train[['FUELCONSUMPTION_COMB','ENGINESIZE','CYLINDERS']])
trainy = np.asanyarray(train[['CO2EMISSIONS']])

testx = np.asanyarray(test[['FUELCONSUMPTION_COMB','ENGINESIZE','CYLINDERS']])
testy = np.asanyarray(test[['CO2EMISSIONS']])
regr.fit(trainx,trainy)
pred = regr.predict(testx)
print(regr.coef_)
#print(" The mean error is: %.2f" % np.mean( np.absolute(testy-pred) )) 


    
ax1=plt.subplot(3,3,4)
plt.scatter(train.FUELCONSUMPTION_COMB,train.CO2EMISSIONS,color = 'red')
plt.plot(trainx, regr.coef_*trainx + regr.intercept_[0],color = 'black')
plt.xlabel('TRAIN FUELCONSUMPTION_COMB')
plt.ylabel('TRAIN Emission')


ax1=plt.subplot(3,3,6)
plt.scatter(test.FUELCONSUMPTION_COMB,test.CO2EMISSIONS,color = 'green')
plt.plot(trainx, np.sum(regr.coef_*trainx,axis=1) + regr.intercept_[0],color = 'black')
plt.xlabel('TEST FUELCONSUMPTION_COMB')
plt.ylabel('TEST Emission')


plt.subplot(332)
plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,color = 'blue')
plt.plot(allx, np.sum(regr.coef_ * allx , axis=1) + regr.intercept_[0],color = 'black')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('Emission')


plt.subplot(338)
plt.plot(trainx, regr.coef_[0][0]*trainx + regr.intercept_[0],color = 'black')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('Emission')
plt.show()