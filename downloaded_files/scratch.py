#!/usr/bin/env python

# (c) 2017, Chris Hodapp
# Code for: https://www.kaggle.com/c/titanic

# "It is your job to predict if a passenger survived the sinking of
# the Titanic or not. For each PassengerId in the test set, you must
# predict a 0 or 1 value for the Survived variable."  "Your score is
# the percentage of passengers you correctly predict."

import pandas
import numpy
import xgboost
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.ensemble

pandas.set_option("display.width", None)

# Load data:
train_raw = pandas.read_csv("train.csv")
test = pandas.read_csv("test.csv")

# Transform some columns:
def xform_input(df):
    # One-hot encode categorical Embarked column, and turn column
    # "Sex" to a number - 0 for male, 1 for female (it has only these
    # two categories, all non-null)
    onehot = pandas.get_dummies(df.Embarked, "Embarked")
    return df.assign(Sex = numpy.where(df.Sex == "male", 0, 1)) \
             .drop("Embarked", axis=1) \
             .join(onehot)

train_raw = xform_input(train_raw)
test = xform_input(test)

# Train/validation split, and throw out some columns:
train, valid = sklearn.model_selection.train_test_split(
    train_raw, test_size=0.1, random_state=1234)
cols = ("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
        "Embarked_C", "Embarked_Q", "Embarked_S")
train_X = train.loc[:, cols]
valid_X = valid.loc[:, cols]
train_Y = train.loc[:, "Survived"]
valid_Y = valid.loc[:, "Survived"]

# Fill in NAs:
imp = sklearn.preprocessing.Imputer(strategy = "mean", axis = 0)
imp = imp.fit(train_X)
train_X_arr = imp.transform(train_X)
valid_X_arr = imp.transform(valid_X)
test_X_arr = imp.transform(test.loc[:, cols])

# Train logistic regression model:
logistic = sklearn.linear_model.LogisticRegression(C = 1e5)
logistic = logistic.fit(train_X_arr, train_Y)

rf = sklearn.ensemble.RandomForestClassifier(
    n_estimators=200, criterion="entropy", max_depth=7, n_jobs=-1,
    random_state=12348)
rf = rf.fit(train_X_arr, train_Y)

ab = sklearn.ensemble.AdaBoostClassifier(
    n_estimators=500,
    learning_rate=0.95,
    random_state=12345)
ab = ab.fit(train_X_arr, train_Y)

# Training & validation accuracy:
train_acc = logistic.score(train_X_arr, train_Y)
valid_acc = logistic.score(valid_X_arr, valid_Y)
print("Logistic regression, training & validation accuracy: {0:.2f}%, {1:.2f}%".format(
    100 * train_acc, 100 * valid_acc))

train_acc = rf.score(train_X_arr, train_Y)
valid_acc = rf.score(valid_X_arr, valid_Y)
print("Random forests, training & validation accuracy: {0:.2f}%, {1:.2f}%".format(
    100 * train_acc, 100 * valid_acc))

train_acc = ab.score(train_X_arr, train_Y)
valid_acc = ab.score(valid_X_arr, valid_Y)
print("AdaBoost, training & validation accuracy: {0:.2f}%, {1:.2f}%".format(
    100 * train_acc, 100 * valid_acc))

# Generate submission:
submission = test[["PassengerId"]]
submission = submission.assign(Survived = rf.predict(test_X_arr))
submission.to_csv("submission.csv", index=False)
