from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error

from impyute.imputation.cs import mice
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


print('=========================  Part A and D  =========================\n')
# Import the dataset
train_dataset = pd.read_csv('2class_trianing.csv')
test_dataset = pd.read_csv('2class_test.csv')
train_dataset4 = pd.read_csv('4class_trianing.csv')
test_dataset4 = pd.read_csv('4class_test.csv')
# Count the number of NaN
# print(train_dataset)
# print(train_dataset.isna().sum())
# print(test_dataset.isna().sum())
X_train = train_dataset.iloc[:, 1:-1]
y_train = train_dataset.iloc[:, 119]
X_test = test_dataset.iloc[:, 1:-1]
y_test = test_dataset.iloc[:, 119]
X_train4 = train_dataset4.iloc[:, 1:-1]
y_train4 = train_dataset4.iloc[:, 119]
X_test4 = test_dataset4.iloc[:, 1:-1]
y_test4 = test_dataset4.iloc[:, 119]
# 用MICE填補缺失值
X_train = mice(X_train.values)
X_test = mice(X_test.values)
X_train4 = mice(X_train4.values)
X_test4 = mice(X_test4.values)


def feature_normalize(X):
    # mean of indivdual column, hence axis = 0
    mu = np.mean(X, axis=0)
    # Notice the parameter ddof (Delta Degrees of Freedom)  value is 1
    # Standard deviation (can also use range)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma


# Normalization
X_train, mu, sigma = feature_normalize(X_train)
X_test, mu, sigma = feature_normalize(X_test)
X_train4, mu, sigma = feature_normalize(X_train4)
X_test4, mu, sigma = feature_normalize(X_test4)

# K-Fold cross-validation
kf = KFold(n_splits=10)

# Import the model
# 'lbfgs' is good for small dataset(default = 'adam')
# aplha: L2 penalty (regularization term) parameter.(default = 0.0001)
clf = MLPClassifier(solver='adam',
                    alpha=1e-5,
                    hidden_layer_sizes=(200, 3),
                    # verbose=True,
                    learning_rate_init=0.001,
                    activation='relu',
                    max_iter=400,
                    tol=0.0001,
                    n_iter_no_change=10,
                    random_state=1)

clf4 = MLPClassifier(solver='adam',
                     alpha=1e-5,
                     hidden_layer_sizes=(200, 100, 50, 32, 8),
                     #  verbose=True,
                     learning_rate_init=0.001,
                     activation='relu',
                     max_iter=800,
                     tol=0.0001,
                     n_iter_no_change=10,
                     random_state=1)

# Train and validate with K-Fold

for train_indices, val_indices in kf.split(X_train4):
    clf4.fit(X_train4[train_indices], y_train4[train_indices])
    print('val:', clf4.score(X_train4[val_indices], y_train4[val_indices]))

clf.fit(X_train, y_train)
clf4.fit(X_train4, y_train4)
print('================================================')
print('2 classes Score:', clf.score(X_test, y_test))
print('4 classes Score:', clf4.score(X_test4, y_test4))
print('2 classes last loss:', clf.loss_)
print('4 classes last loss:', clf4.loss_)
print('================================================')
print('4 classes prediction:\n', clf4.predict(X_test4))
# print('With Parameters:\n', clf.get_params())


# Visualization
plt.rcParams["figure.figsize"] = (8, 6)
plt.figure()

plt.subplot(211)
plt.grid(True, which='both')
plt.xlabel("Number of iterations")
plt.ylabel("cost (J)")
plt.title("2 Classes Loss Curve")
plt.plot(clf.loss_curve_, 'b')

plt.subplot(212)
plt.grid(True, which='both')
plt.xlabel("Number of iterations")
plt.ylabel("cost (J)")
plt.title("4 Classes Loss Curve")
plt.plot(clf4.loss_curve_, 'r')

plt.tight_layout()
plt.show()
