import numpy as np
import pandas as pd
import csv 
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

data=pd.read_csv("8heartdiseasedata.csv")

heartDisease=pd.DataFrame(data)

from sklearn.preprocessing import LabelEncoder

heartDisease.columns
 
lb = LabelEncoder()
for col in heartDisease.columns:
    heartDisease[col] = lb.fit_transform(heartDisease[col])



print('Sample instances from the dataset are given below')
print(heartDisease.head())

print('\n Attributes and datatypes')
print(heartDisease.dtypes)

model= BayesianNetwork([('age','heartdisease'),('Gender','heartdisease'),('Family','heartdisease'),('diet','cholestrol'),('Lifestyle','diet'),('heartdisease','cholestrol')])
print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease,estimator=MaximumLikelihoodEstimator)

print('\n Inferencing with Bayesian Network:')
HeartDiseasetest_infer = VariableElimination(model)


print('For age Enter { SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4 }')

print('For cholesterol Enter { High:0, BorderLine:1, Normal:2 }')

print('\n 1. Probability of HeartDisease given evidence= age')
q1=HeartDiseasetest_infer.query(variables=['heartdisease'],evidence={'age':int(input("Enter age"))})
print(q1)

print('\n 2. Probability of HeartDisease given evidence= cholestrol ')
q2=HeartDiseasetest_infer.query(variables=['heartdisease'],evidence={'cholestrol':int(input("Enter Cholestrol"))})
print(q2)

