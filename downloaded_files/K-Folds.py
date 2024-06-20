# This file is used to predict the result of the test data

import sklearn as sk
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

# Models to be tested (10)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


classifiers = {
    # 'LinearRegression': LinearRegression(),
    'LogisticRegression': LogisticRegression(),
    'GaussianNB': GaussianNB(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'SVC_linear': SVC(kernel='linear', probability=True),
    'SVC_poly': SVC(kernel='poly', probability=True),
    'SVC_rbf': SVC(kernel='rbf', probability=True),
    'SVC_sigmoid': SVC(kernel='sigmoid', probability=True),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}

params = {
    # 'LinearRegression': {},
    'LogisticRegression': {},
    'GaussianNB': {},
    'KNeighborsClassifier': {'clf__n_neighbors': [3, 5, 7]},
    'DecisionTreeClassifier': {'clf__criterion': ['gini', 'entropy'], 'clf__max_depth': [3, 5, 7]},
    'SVC_linear': {'clf__C': [0.1, 1, 10], 'clf__gamma': [0.1, 1, 10]},
    'SVC_poly': {'clf__C': [0.1, 1, 10], 'clf__gamma': [0.1, 1, 10]},
    'SVC_rbf': {'clf__C': [0.1, 1, 10], 'clf__gamma': [0.1, 1, 10]},
    'SVC_sigmoid': {'clf__C': [0.1, 1, 10], 'clf__gamma': [0.1, 1, 10]},
    'RandomForestClassifier': {'clf__n_estimators': [100, 200, 300, 400], 'clf__criterion': ['gini', 'entropy'],
                                    'clf__max_depth': [3, 5, 7]},
    'AdaBoostClassifier': {'clf__n_estimators': [100, 200, 300, 400]},
    'GradientBoostingClassifier': {'clf__n_estimators': [100, 200, 300, 400], 'clf__learning_rate': [0.1, 0.01, 0.001]}
}


def plot_matrix_confusion(y_test, y_pred,name):
    display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(name)

    plt.show()


# Read the csv file
df = pd.read_csv('hu_moments.csv')  # zernike_moments.csv para zernike

# Separate the features from the labels
X = df.drop('shape', axis=1)
y = df['shape']

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)


# Create a dictionary to store the results
results = {}

cv = KFold(n_splits=10, shuffle=True, random_state=0)  # CAMBIAR A 5 o 10, según el caso

# Iterate over the classifiers
for name, classifier in classifiers.items():
    pipeline = Pipeline([
        ('Scaler', StandardScaler()),
        ('clf', classifier)
    ])

    # Create a grid search object
    grid_search = GridSearchCV(pipeline, params[name], scoring='accuracy', n_jobs=-1, verbose=1, cv=cv)

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Get prediction
    y_pred = grid_search.predict(X_test)

    # print report and accuracy

    print('Classifier: {}'.format(name))
    print('Best params: {}'.format(grid_search.best_params_))
    print('Best score: {:.2f}'.format(grid_search.best_score_))
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))

    # if predict_proba is available
    y_pred_proba = grid_search.predict_proba(X_test)

    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
    print('AUC: {:.2f}'.format(auc))
    #
    # # Store the results

    plot_matrix_confusion(y_test, y_pred, name)

# Print the results
print(results)





