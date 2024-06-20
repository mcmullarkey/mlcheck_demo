import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('csv/fruit_data_with_colors.csv')
label_count = len(df['fruit_label'].unique())

scores = []
for k in range(1, label_count + 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train[['mass', 'color_score', 'width', 'height']], train['fruit_label'])
    score = knn.score(test[['mass', 'color_score', 'width', 'height']], test['fruit_label'])
