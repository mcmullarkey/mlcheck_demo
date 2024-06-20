# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import dtreeviz

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

species_names = ['benign', 'malignant']
y_labels = [species_names[label] for label in y]

print("___________________________________________")
# Create a DataFrame from the dataset
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = y_labels  # Add the species labels as a new column

# Display specific fields for 'malignant' (class 0)
malignant_data = df[df['target'] == 'malignant']
print("Data for malignant cancer (Class 0):")
print(malignant_data.head(3))
print("___________________________________________")
# Display specific fields for 'benign' (class 1)
benign_data = df[df['target'] == 'benign']
print("\nData for 'benign cancer' (Class 1):")
print(benign_data.head(3))

# Prepare Data for Binary Classification: Classify between 'benign' (class 1) and 'malignant' (class 0)
y_binary = (y == 1).astype(int)  # 1 for 'benign' and 0 for 'malignant'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()
# Train (fit) the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("___________________________________________")
# Visualize the decision tree using 'dtreeviz' MALIGNANT and BENIGN
m = dtreeviz.model(clf, X_train, y_train, target_name='Cancer type', feature_names=data.feature_names, class_names=['malignant', 'benign'])
m.view()


from sklearn.tree import DecisionTreeClassifier, plot_tree

import matplotlib.pyplot as plt

# Visualize the decision tree using Matplotlib
plt.figure(figsize=(30, 10))
plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=['malignant', 'benign'], fontsize=10)
plt.show()

# Export the decision tree as text to analyze entropy and information gain
from sklearn.tree import export_text
# tree_rules = export_text(clf, feature_names=data.feature_names)
tree_rules = export_text(clf, feature_names=list(data.feature_names))
print("Decision Tree Rules:")
print(tree_rules)
