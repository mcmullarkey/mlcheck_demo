import util
import json
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
import re
import nltk
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

y = pd.read_pickle('xys/y_test.pkl')
y_pred = pd.read_pickle('xys/y_pred.pkl')

cm = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)

print('Average Accuracy:', accuracy)

# Plot confusion matrix
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Lose', 'Predicted Win'], yticklabels=['True Lose', 'True Win'])
plt.title(f'Confusion Matrix for NN on Custom Word2Vec')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(f'xys/word2veccm.png')