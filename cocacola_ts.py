# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:04:20 2020

@author: Bhanu Teja
"""

import pandas as pd
cocacola = pd.read_excel("C:\\Users\\Bhanu Teja\\Downloads\\Datasets (6)\\CocaCola_Sales_Rawdata.xlsx")
quarter = 'Q1','Q2','Q3','Q4'

# Pre processing
import numpy as np

cocacola["t"] = np.arange(1,43)
cocacola.head()
cocacola["t_squared"] = cocacola["t"]*cocacola["t"]
cocacola["log_Sales"] = np.log(cocacola["Sales"])
cocacola.columns

p = cocacola["Quarter"][0]
p[0:2]
cocacola['quarter']= 0
cocacola.head()
for i in range(42):
    p = cocacola["Quarter"][i]
    cocacola['quarter'][i]= p[0:2]
    
quarter_dummies = pd.DataFrame(pd.get_dummies(cocacola['quarter']))
cocacola1 = pd.concat([cocacola, quarter_dummies], axis = 1)

#airlines["Month"] = airlines.Month.dt.strftime("%y")  # for year extraction

#airlines["Month"] = pd.to_datetime(airlines.Month,format="%b-%y")
cocacola1 = cocacola1.drop('Quarter',axis=1)
cocacola1.head()
# Visualization - Time plot
cocacola1.Sales.plot()

# Data Partition
Train = cocacola1.head(38)
Test = cocacola1.tail(4)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales ~ t', data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_Sales ~ t', data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Sales ~ t+t_squared', data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Sales ~ Q1+Q2+Q3', data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Sales ~ Q1+Q2+Q3',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

################## Additive Seasonality Exponential Trend ############################

add_sea_Quad = smf.ols('log_Sales ~ t+Q1+Q2+Q3',data=Train).fit()
pred_add_sea_exp = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','t','t_squared']]))
rmse_add_sea_exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_exp))**2))
rmse_add_sea_exp 

################## Multiplicative Seasonality Exponential Trend  ###########

Mul_Add_sea = smf.ols('log_Sales ~ t+Q1+Q2+Q3',data = Train).fit()
pred_Mult_sea_exp = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_sea_exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea_exp)))**2))
rmse_Mult_sea_exp 

#Model based methods are not giving good results so here i am going to try with data-driven method

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
# from datetime import datetime

cocacola = pd.read_excel("C:\\Users\\Bhanu Teja\\Downloads\\Datasets (6)\\CocaCola_Sales_Rawdata.xlsx")

cocacola.Sales.plot() # time series plot 

# Centering moving average for the time series
cocacola.Sales.plot(label = "org")
for i in range(2, 9, 2):
    cocacola["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)
    
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(cocacola.Sales, model = "additive", period = 4)
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(cocacola.Sales, model = "multiplicative", period = 4)
decompose_ts_mul.plot()

# ACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cocacola.Sales, lags = 4)
# tsa_plots.plot_pacf(cocacola.Sales, lags=4)

# splitting the data into Train and Test data
# Recent 4 time period values are Test data

Train = cocacola.head(38)
Test = cocacola.tail(4)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,4),inplace=True)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_ses, Test.Sales) 

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.Sales) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.Sales) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.Sales) 


# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(cocacola["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()

#######
##################################################


