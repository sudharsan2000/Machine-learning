import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("FuelConsumptionCo2.csv")
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
# viz.hist()

msk = np.random.rand(len(df))<0.8
train = df[msk]
test = df[~msk]

trainx = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
trainy = np.asanyarray(train[['CO2EMISSIONS']])

testx = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
testy = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree = 2)
trainx = poly.fit_transform(trainx)
testx = poly.fit_transform(testx)

regr = linear_model.LinearRegression()
regr.fit(trainx,trainy)

pred = regr.predict(testx)
print(" The mean error is: %.2f" % np.mean( np.absolute(testy-pred) )) 

xx = np.arange(0,25,0.1)

plt.scatter(train.FUELCONSUMPTION_COMB,train.CO2EMISSIONS,color = 'blue')
plt.plot(xx, regr.coef_[0][2]*np.power(xx,2) + regr.coef_[0][1]*xx + regr.intercept_[0],color = 'black')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('Emission')


plt.show()