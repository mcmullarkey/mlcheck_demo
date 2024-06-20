from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import time
from sklearn.utils import all_estimators
import warnings
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV
warnings.filterwarnings('ignore')

# 1 데이터

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (581012, 54) (581012,)
print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

# ★ 코드 완성 했을때 오류가 생길텐데 케라스 판다스 사이킷런 전처리과정에서 생길예정
# y의 시작값이 1부터 7까지의 6 개인 이유로 오류가 발생할것
# 오류가 나는 코드는 수정해서 완료시킬것

print(y)
print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))

# ============================================
# ========== 원 핫 인코딩 전처리 ==============
# 1) 케라스
# from keras.utils import to_categorical
# y_ohe = to_categorical(y)   # [0. 0. 0. ... 1. 0. 0.]
# print(y_ohe)
# print(y_ohe.shape)  # (581012, 8)

# # 2) 판다스
# y_ohe2 = pd.get_dummies(y)  # False  False  False  False   True  False  False
# print(y_ohe2)
# print(y_ohe2.shape) # (581012, 7)

# # 3) 사이킷런
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()   # (sparse=False)
# y = y.reshape(-1, 1)    # (행, 열) 형태로 재정의 // -1 은 열의 정수값에 따라 알아서 행을 맞추어 재정의하라 
# y_ohe3 = ohe.fit_transform(y).toarray() # // 투어레이 사용하면 위에 스파라스 안씀. 스파라스 사용하면 투어레이 안씀
# print(y_ohe3)
# print(y_ohe3.shape) # (581012, 7)


x_train, x_test, y_train, y_test = train_test_split(x, y, 
            shuffle=True, train_size= 0.7,
            random_state= 687415,
            stratify=y,)    # 스트레티파이 와이(예스)는 분류에서만 쓴다, 트레인 사이즈에 따라 줄여주는것


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler


scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))  # 
print(np.min(x_test))   # 
print(np.max(x_train))  # 
print(np.max(x_test))   # 

print(np.unique(y_test, return_counts=True))
# (array([0., 1.], dtype=float32), array([1220128,  174304], dtype=int64))


from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,10,12], 'min_samples_leaf':[3,10]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
    {'min_samples_split':[2,3,5,10]},
    {'n_jobs':[-1,2,4],'min_samples_split':[2,3,5,10]}
]


# 2 모델
# model = SVC(C=1, kernel='linear', degree=3)
print('==============하빙그리드서치 시작==========================')
model = HalvingGridSearchCV(RandomForestClassifier(), 
                     parameters,
                     cv = kfold,
                     verbose=1,
                    #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
                     n_jobs=3   # 24개의 코어중 3개 사용 / 전부사용 -1
                     , random_state=123,
                    #  n_iter=10
                     factor=3,
                    #  min_resources=
                     )


start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print('최적의 매개변수 : ', model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
print('최적의 파라미터 : ', model.best_params_) # 내가 선택한것
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'} 우리가 지정한거중에 가장 좋은거
print('best_score : ', model.best_score_)   # 핏한거의 최고의 스코어
# best_score :  0.975
print('model_score : ', model.score(x_test, y_test))    # 
# model_score :  0.9666666666666667


y_predict = model.predict(x_test)
print('accuracy_score', accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
            # SVC(C-1, kernel='linear').predicict(x_test)
print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))

print('걸린신간 : ', round(end_time - start_time, 2), '초')

# import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)