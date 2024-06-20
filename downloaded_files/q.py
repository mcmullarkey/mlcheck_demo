#print(__doc__)

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
###############################################################################
# Generate sample data
parent = []
target = []
with open('sample_train.csv') as f:
    for line in f:
        line = line.split(',')
	count = 0
	count1 = 0
	fill = 0.000001
	length = -1
	line1 = line[26:209] 
	for i in line1:
	    length = length + 1
            if i == '':
		line1[length] = fill
		#print fill
	        #count = count + 1
	    else:
		fill = line1[length] 


		#count1 = count1 + 1 
        #print count1
	#for space in line1:
	#    if space == '':
	#	count= count+1
	#print line1

	#a = []

	train_X = line1[0:120]
	parent.append(train_X)	
	#print parent
	y = line1[121:209]
	target.append(y)
	
#############################################################################
# before 2 days and 180 values every minute value and after 2 days
#parent = np.sort(train_X, axis=0)

New_train_X = []
with open('test.csv') as f:
    for line in f:
        line = line.split(',')
	count = 0
	count1 = 0
	fill = 0.000001
	length = -1
	line1 = line[26:145] 
	for i in line1:
	    length = length + 1
            if i == '':
		line1[length] = fill
		#print fill
	        #count = count + 1
	    else:
		fill = line1[length] 
	New_train_X.append(line1)
#############################################################################
#print X
#y = np.sin(X).ravel()
#print y
###############################################################################
# Add noise to targets
#y[::5] += 3 * (0.5 - np.random.rand(8))
#print y
###############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(parent, target).predict(New_train_X)
#y_lin = svr_lin.fit(X, y).predict(X)
#y_poly = svr_poly.fit(X, y).predict(X)

###############################################################################
# look at the results
plt.scatter(parent, target, c='k', label='data')
plt.hold('on')
plt.plot(New_train_X, y_rbf, c='g', label='RBF model')
#plt.plot(X, y_lin, c='r', label='Linear model')
#plt.plot(X, y_poly, c='b', label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
