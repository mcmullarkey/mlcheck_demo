import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sbn
import statistics
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

test_DF=pd.DataFrame(pd.read_csv("test.csv"))
train_DF=pd.DataFrame(pd.read_csv("train.csv"))

test_DF.drop(labels=["Name","Ticket","Fare","Cabin"],axis=1,inplace=True)
train_DF.drop(labels=["PassengerId","Name","Ticket","Fare","Cabin"],axis=1,inplace=True)

train_DF=train_DF.dropna()
test_DF=test_DF.dropna()

le=LabelEncoder()
train_DF["Sex"]=le.fit_transform(train_DF["Sex"]) 
test_DF["Sex"]=le.fit_transform(test_DF["Sex"])#male 1 female 0 
train_DF["Embarked"]=le.fit_transform(train_DF["Embarked"])
test_DF["Embarked"]=le.fit_transform(test_DF["Embarked"])

ageScaler=StandardScaler()
train_DF["Age"]=ageScaler.fit_transform(train_DF[["Age"]])
test_DF["Age"]=ageScaler.fit_transform(test_DF[["Age"]])


def random_forest():
    Y_train = train_DF['Survived']
    X_train = train_DF.drop('Survived', axis=1)
    X_test=test_DF.drop(labels="PassengerId",axis=1)
    
    random_forest = RandomForestClassifier(n_estimators=100) # karar ağaçları     
    random_forest.fit(X_train, Y_train)
    Y_prediction = random_forest.predict(X_test)
    
    sonuc=pd.DataFrame(test_DF["PassengerId"])
    sonuc["Survived"]=Y_prediction
    sonuc.to_csv("mySubmissions.csv", encoding='utf-8', index=False)

    score=random_forest.score(X_train, Y_train)
    acc_random_forest = round(score * 100, 2)
    print("%",acc_random_forest)
    
    
def random_forest2():
    x=train_DF.drop('Survived', axis=1)
    y=train_DF['Survived']
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.26,random_state=20)    
    # print(x_train.shape)#(526, 6)
    # print(y_train.shape)#(526,)
    # print(x_test.shape)#(186, 6)
    # print(y_test.shape)#(186,)
    random_forest = RandomForestClassifier(n_estimators=100) # karar ağaçları     
    rfc = random_forest.fit(x_train, y_train)
    Y_prediction = rfc.predict(x_test)  
    # sonuc=pd.DataFrame(test_DF["PassengerId"])
    test_data=x_test
    test_data["Survived Gerçek"]=y_test
    test_data["Survived Tahmin"]=Y_prediction
    test_data.to_csv("mySubmissions2.csv", encoding='utf-8', index=False)
    
    score=random_forest.score(x_train, y_train)
    acc_random_forest = round(score * 100, 2)
    #print("%",acc_random_forest)

# random_forest()
# random_forest2()
deneme=pd.DataFrame(pd.read_csv("mySubmissions2.csv"))
deneme.drop(labels=["Pclass","Sex","Age","SibSp","Parch","Embarked"],axis=1,inplace=True)

b=[]
for i in range(len(deneme)):
    a=deneme.iloc[i]
    a=a.values
    a=a[0]+a[1]
    if a==1:
        b.append(deneme.iloc[i])


random_forest2()
print("Doğruluk Oranı: ",round(100-((len(b)/189)*100),2))
print("Uyumlu olmayanlar")
print(b)









#Grid Search Cross Validation