# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:51:56 2021

@author: S. Gribben
"""
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# %% error functions
"""
MAPE calculation
input: the actual value (LCL data) and the predicted (guassian model)
output: the mean absolute percentage error
"""
def mape(actual, pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100

"""
Pearsons correlation coefficent
input: the actual value (LCL data) and the predicted (guassian model)
output: the Pearsons correlation coefficent
"""
def pearsons(actual, pred):
    return np.corrcoef(actual, pred)


#%% import data
# get current working directory
os.getcwd()  


# Read the data
lcl = pd.read_csv("lcl_feederPower.csv", header=None, index_col=0, parse_dates=True)
lcl['HH']=lcl.index.hour*2+(lcl.index.minute/30)

guassian = pd.read_csv("guassian_feederPower.csv", header=None, index_col=0, parse_dates=True)
guassian['HH']=guassian.index.hour*2+(guassian.index.minute/30)

guassian50 = pd.read_csv("guassian50_feederPower.csv", header=None, index_col=0, parse_dates=True)
guassian50['HH']=guassian.index.hour*2+(guassian.index.minute/30)

elexon = pd.read_csv("elexon_feederPower.csv", header=None, index_col=0, parse_dates=True)
elexon['HH']=elexon.index.hour*2+(elexon.index.minute/30)

crest = pd.read_csv("CREST_feederPower.csv", header=None, index_col=0, parse_dates=True)
crest['HH']=crest.index.hour*2+(crest.index.minute/30)

# %% calculations

#find the overall MAPE
gua_overall = mape(lcl[1].to_numpy(), guassian[1].to_numpy())
elexon_overall = mape(lcl[1].to_numpy(), elexon[1].to_numpy())
crest_overall = mape(lcl[1].to_numpy(), crest[1].to_numpy())

#find the mape for each HH value 
gua_mape = np.zeros(48)
elexon_mape = np.zeros(48)
crest_mape = np.zeros(48)

for hh in range(0, 48, 1):
    lcl_hh = lcl.loc[(lcl['HH'] == hh)].to_numpy()
    gua_hh = guassian.loc[(guassian['HH'] == hh)].to_numpy()
    elexon_hh = elexon.loc[(elexon['HH'] == hh)].to_numpy()
    crest_hh = crest.loc[(crest['HH'] == hh)].to_numpy()
    
    gua_mape[hh] = mape(lcl_hh[:, 0], gua_hh[:, 0])
    elexon_mape[hh] = mape(lcl_hh[:, 0], elexon_hh[:, 0])
    crest_mape[hh] = mape(lcl_hh[:, 0], crest_hh[:, 0])
                            

#pearsons correlation coefficent 
guassian_rho = pearsons(lcl[1].to_numpy(), guassian[1].to_numpy())
elexon_rho = pearsons(lcl[1].to_numpy(), elexon[1].to_numpy())
crest_rho = pearsons(lcl[1].to_numpy(), crest[1].to_numpy())

#mean squared error
gua_mse = mean_squared_error(lcl[1].to_numpy(), guassian[1].to_numpy())
gua50_mse = mean_squared_error(lcl[1].to_numpy(), guassian50[1].to_numpy())
elexon_mse = mean_squared_error(lcl[1].to_numpy(), elexon[1].to_numpy())
crest_mse = mean_squared_error(lcl[1].to_numpy(), crest[1].to_numpy())

# %% plots
#plot the error as a bar chart

x = np.arange(48)
fig, ax = plt.subplots()
ax.bar(x, crest_mape)
ax.set_ylabel("MAPE (%)")
ax.set_title("MAPE of each half hour")