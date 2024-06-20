#!/usr/bin/env python

# Usage:
# ./et.py TARGET_ATTRIBUTE TRAIN_FILE TEST_FILE OUT_FILE
# Trains on TRAIN_FILE and outputs TEST_FILE with desired TARGET_ATTRIBUTE
# replaced by et predictions into OUT_FILE
# TARGET_ATTRIBUTE must be one of: sex, age, race, marital-status, education,
# education, native-country, workclass, occupation, salary-class

import csv, sys
from sklearn.ensemble import ExtraTreesClassifier

target = sys.argv[1]

attributes = []
train = {}
with open(sys.argv[2], 'rb') as csvreader:
    c = csv.DictReader(csvreader)
    attributes = c.fieldnames
    train = dict(zip(attributes, [[] for i in xrange(len(attributes))]))
    for row in c:
        for a in attributes:
            train[a].append(row[a])
n_train = len(train[attributes[0]])

test = dict(zip(attributes, [[] for i in xrange(len(attributes))]))
with open(sys.argv[3], 'rb') as csvreader:
    c = csv.DictReader(csvreader)
    for row in c:
        for a in attributes:
            test[a].append(row[a])
n_test = len(test[attributes[0]])

x = [[0 for i in xrange(len(attributes))] for j in xrange(n_train)]
y = [0 for i in xrange(n_train)]
for i in xrange(n_train):
    for j in xrange(len(attributes)):
        if attributes[j] == target:
            x[i][j] = 0
            y[i] = train[attributes[j]][i]
        else:
            x[i][j] = train[attributes[j]][i]

clf = ExtraTreesClassifier()
clf = clf.fit(x, y)

z = [[test[a][j] for a in attributes] for j in xrange(n_test)]
for j in xrange(n_test):
    for i in xrange(len(attributes)):
        if z[j][i] == '*':
            z[j][i] = 0

fout = open(sys.argv[4], 'w')
results = map(lambda x: clf.predict(x), z)
fout.write(target + '\n')
for r in results:
    fout.write(r[0] + '\n')

fail = 0
for i in xrange(len(results)):
    if results[i][0] != test[target][i]:
        fail += 1

print 'Number wrong:', fail
print "% wrong: ", float(fail)/float(len(results))*100
