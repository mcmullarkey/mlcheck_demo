import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import json 

# Getting input from the forum
input_str = input()
input_dict = json.loads(input_str)

fileName = 'crop.csv'

if input_dict["File"]=="Yes":
    fileName = input_dict["FileName"]



# Reading the dataset file 
dataset = pd.read_csv("uploads/"+fileName)
X = dataset.iloc[:,:-1].values 
Y = dataset.iloc[:,-1].values

# Splitting the model to train and test set 
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training the model 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

np.set_printoptions(precision=2)

print(regressor.predict([[input_dict["R"], input_dict["T"], input_dict["W"], input_dict["S"], input_dict["So"], input_dict["Af"]]]))


