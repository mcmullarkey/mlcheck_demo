from __future__ import division
import numpy as np
import gzip
import cPickle
import sys
import datetime
import random
from scipy import optimize
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer

class EONet(object):

    """
    Unchanged from the feedforward architecture
    Must go down before the new EO content starts
    """

    def __init__(self, epsilon_init=0.12, hidden_layer_size=25):
        self.hidden_layer_size = hidden_layer_size
        self.epsilon_init = epsilon_init
        self.activation_func = self.sigmoid
        self.activation_func_prime = self.sigmoid_prime

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def sumsqr(self, a):
        return np.sum(a ** 2)

    def rand_init(self, l_in, l_out):
        return np.random.rand(l_out, l_in + 1) * 2 * self.epsilon_init - self.epsilon_init

    def pack_thetas(self, t1, t2):
        return np.concatenate((t1.reshape(-1), t2.reshape(-1)))

    def unpack_thetas(self, thetas, input_layer_size, hidden_layer_size, num_labels):
        t1_start = 0
        t1_end = hidden_layer_size * (input_layer_size + 1)
        t1 = thetas[t1_start:t1_end].reshape((hidden_layer_size, input_layer_size + 1))
        t2 = thetas[t1_end:].reshape((num_labels, hidden_layer_size + 1))
        return t1, t2

    def forward(self, X, t1, t2):
        m = X.shape[0]
        ones = None
        if len(X.shape) == 1:
            ones = np.array(1).reshape(1,)
        else:
            ones = np.ones(m).reshape(m,1)

        # Input layer
        a1 = np.hstack((ones, X))

        # Hidden Layer
        z2 = np.dot(t1, a1.T)
        a2 = self.activation_func(z2)
        a2 = np.hstack((ones, a2.T))

        # Output layer
        z3 = np.dot(t2, a2.T)
        a3 = self.activation_func(z3)
        return a1, z2, a2, z3, a3

    def calc_local_energy(self, thetas, input_layer_size, hidden_layer_size, num_labels, X, y):
        t1, t2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)
        m = X.shape[0]
        t1f = t1[:, 1:]
        t2f = t2[:, 1:]
        Y = np.eye(num_labels)[y]

        Delta1, Delta2 = 0, 0
        for i, row in enumerate(X):
            a1, z2, a2, z3, a3 = self.forward(row, t1, t2)

            # Backprop
            d3 = a3 - Y[i, :].T
            d2 = np.dot(t2f.T, d3) * self.activation_func_prime(z2)

            Delta2 += np.dot(d3[np.newaxis].T, a2[np.newaxis])
            Delta1 += np.dot(d2[np.newaxis].T, a1[np.newaxis])

        Theta1_grad = (1 / m) * Delta1
        Theta2_grad = (1 / m) * Delta2
        return self.pack_thetas(Theta1_grad, Theta2_grad)

    def calc_total_energy(self, thetas, input_layer_size, hidden_layer_size, num_labels, X, y):
        t1, t2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)

        m = X.shape[0]
        Y = np.eye(num_labels)[y]

        _, _, _, _, h = self.forward(X, t1, t2)
        costPositive = -Y * np.log(h).T
        costNegative = (1 - Y) * np.log(1 - h).T
        cost = costPositive - costNegative
        return np.sum(cost) / m #J

    def weight_extinction(self, energies, thetas, tau=1.15):
        """
        Entirely with theta and solution packed.
        """
        k = thetas.shape[0]
        thetas_len = thetas.shape[0]
        while k > thetas_len-1:
            k = int(np.random.pareto(tau))
        worst_city = energies.argsort()[-k:][::-1][-1]
        rand_idx = random.randrange(0, thetas_len)
        thetas[worst_city] += (np.random.rand() * 0.05 - 0.025)
        #thetas[worst_city] -= energies[worst_city] * 0.1
        #thetas[rand_idx], thetas[worst_city] = thetas[worst_city], thetas[rand_idx]
        return thetas

    def gradient_descent(self, energies, thetas, alpha=0.02):
        return thetas - (energies * alpha)

    def eo(self, X, y, steps=10000, disp=True):
        num_features = X.shape[0]
        input_layer_size = X.shape[1]
        try:
            num_labels = len(set(y))
        except:
            num_labels = y.size

        theta1_0 = self.rand_init(input_layer_size, self.hidden_layer_size)
        theta2_0 = self.rand_init(self.hidden_layer_size, num_labels)
        best_s = self.pack_thetas(theta1_0, theta2_0)
        best_energy = float("inf")
        total_energy = float("inf")
        prev_energies = np.array(0) #null
        curr_s = best_s.copy()
        num_true = 0
        num_false = 0
        for time in xrange(steps):
            if disp and time % (steps // 100) == 0:
                print "time: ", time, datetime.datetime.now().strftime("%Y %m %d %H:%M:%S")
                print num_true / (time + 1)
            energies = self.calc_local_energy(curr_s, input_layer_size, self.hidden_layer_size, num_labels, X, y)
            if prev_energies.any():
                t1, t2 = self.unpack_thetas(energies - prev_energies, input_layer_size, self.hidden_layer_size, num_labels)
                if t2[1].any() and t2[0].any() and t2[2].any():
                    num_true += 1
            prev_energies = energies
            total_energy = self.calc_total_energy(curr_s, input_layer_size, self.hidden_layer_size, num_labels, X, y)
            if total_energy < best_energy:
                best_energy = total_energy
                best_s = curr_s.copy()
            curr_s = self.weight_extinction(energies, curr_s)
        self.t1, self.t2 = self.unpack_thetas(best_s, input_layer_size, self.hidden_layer_size, num_labels)

    def predict(self, X):
        return self.predict_proba(X).argmax(0)

    def predict_proba(self, X):
        _, _, _, _, h = self.forward(X, self.t1, self.t2)
        return h

def mnist_digits():
    from scipy.io import loadmat
    data = loadmat('ex3data1.mat')
    X, y = data['X'], data['y']
    y = y.reshape(X.shape[0], )
    y = y - 1
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)
    nn = EONet()
    nn.eo(X_train, y_train)
    print "====="
    predictions = nn.predict(X_test)
    print accuracy_score(y_test, predictions)
    print confusion_matrix(y_test, predictions)
    print classification_report(y_test, predictions)

def iris_class():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)
    nn = EONet(hidden_layer_size=15)
    nn.eo(X_train, y_train)
    print accuracy_score(y_test, nn.predict(X_test))

if __name__ == "__main__":
    #mnist_digits()
    iris_class()
