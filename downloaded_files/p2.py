import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Load the Iris dataset
iris = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Define features (X) and target variable (y)
X = iris.drop("species", axis=1)
y = iris["species"]

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the model
clf = clf.fit(X, y)

# Classify a new sample
new_sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = clf.predict(new_sample)
print("Predicted class for new sample:", prediction)

# Visualize the tree
tree.plot_tree(clf)
