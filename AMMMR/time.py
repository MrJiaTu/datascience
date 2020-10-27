import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import Holt
from sklearn.metrics import r2_score
from math import sqrt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import ExponentialSmoothing
df = pd.read_csv("sale.csv")
print(df.head())
print(df.shape)

df = pd.read_csv("sale.csv",nrows=900)
##创建训练集和测试集
train = df[0:622]
test = df[622:]
##
df['Timestamp'] = pd.to_datetime(df['Datetime'], format='%Y/%m/%d ')  # 4位年用Y，2位年用y
df.index = df['Timestamp']
df = df.resample('D').mean() #按天采样，计算均值
##
train['Timestamp'] = pd.to_datetime(train['Datetime'], format='%Y/%m/%d')
train.index = train['Timestamp']
train = train.resample('D').mean()  #
##
test['Timestamp'] = pd.to_datetime(test['Datetime'], format='%Y/%m/%d')
test.index = test['Timestamp']
test = test.resample('D').mean()
# Plotting data
# train.Count.plot(figsize=(15, 8), title='Daily Ridership', fontsize=14)
# test.Count.plot(figsize=(15, 8), title='Daily Ridership', fontsize=14)
# plt.show()
#朴素法  329
# dd = np.asarray(train['Count'])
# y_hat = test.copy()
# y_hat['naive'] = dd[len(dd) - 1]
# plt.figure(figsize=(12, 8))
# plt.plot(train.index, train['Count'], label='Train')
# plt.plot(test.index, test['Count'], label='Test')
# plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast')
# plt.legend(loc='best')
# plt.title("Naive Forecast")
# plt.show()
# rms = sqrt(mean_squared_error(test['Count'], y_hat['naive']))
# print(rms)
##简单平均法  rmes值为330
# y_hat_avg = test.copy()
# y_hat_avg['avg_forecast'] = train['Count'].mean()
# plt.figure(figsize=(12,8))
# plt.plot(train['Count'], label='Train')
# plt.plot(test['Count'], label='Test')
# plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
# plt.legend(loc='best')
# plt.show()
# rms = sqrt(mean_squared_error(test['Count'], y_hat_avg['avg_forecast']))
# print(rms)
# SARIMA 737
# y_hat_avg = test.copy()
# fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
# y_hat_avg['SARIMA'] = fit1.predict(start="2020-8-16", end="2020-9-16", dynamic=True)
# plt.figure(figsize=(16, 8))
# plt.plot(train['Count'], label='Train')
# plt.plot(test['Count'], label='Test')
# plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
# plt.legend(loc='best')
# plt.show()
# rms = sqrt(mean_squared_error(test['Count'], y_hat_avg['SARIMA']))
# print(rms)
##指数法 298  一次指数
y_hat_avg = test.copy()
fit = SimpleExpSmoothing(np.asarray(train['Count'])).fit(smoothing_level=0.22, optimized=False)
y_hat_avg['SES'] = fit.forecast(len(test))
plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()
rms = sqrt(mean_squared_error(test['Count'], y_hat_avg['SES']))
r2=r2_score(test['Count'],y_hat_avg['SES'])
print(rms)
##指数法 298  二次指数
##待补充

##霍尔特(Holt)线性趋势法 700
# y_hat_avg = test.copy()
# fit = Holt(np.asarray(train['Count'])).fit(smoothing_level=1, smoothing_slope=2)
# y_hat_avg['Holt_linear'] = fit.forecast(len(test))
# plt.figure(figsize=(16, 8))
# plt.plot(train['Count'], label='Train')
# plt.plot(test['Count'], label='Test')
# plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
# plt.legend(loc='best')
# plt.show()
# rms = sqrt(mean_squared_error(test['Count'], y_hat_avg['Holt_linear']))
# print(rms)
##Holt-Winters季节性预测模型
# y_hat_avg = test.copy()
# fit1 = ExponentialSmoothing(np.asarray(train['Count']), seasonal_periods=6, trend='add', seasonal='add', ).fit()
# y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
# plt.figure(figsize=(16, 8))
# plt.plot(train['Count'], label='Train')
# plt.plot(test['Count'], label='Test')
# plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
# plt.legend(loc='best')
# plt.show()
# rms = sqrt(mean_squared_error(test['Count'], y_hat_avg['Holt_Winter']))
# print(rms)