import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1.데이터
datasets = load_diabetes()
import pandas as pd
x = datasets.data
y = datasets.target
print(datasets.feature_names)
x = pd.DataFrame(x,columns=['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])
from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,random_state=234)


#2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor()

model.fit(x_train, y_train)

#4평가예측
from sklearn.metrics import r2_score
y_predicy = model.predict(x_test)
r2 = r2_score(y_test, y_predicy)
result = model.score(x_test,y_test)
print('model.score',result)
print('r2',r2)
print(model.feature_importances_)

import matplotlib.pyplot as plt

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel('feature importances')
    plt.ylabel('features')
    plt.ylim(-1,n_features)

plot_feature_importances_dataset(model)
plt.show()

from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()