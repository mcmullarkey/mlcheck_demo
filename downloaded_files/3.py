import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
iris_data = pd.read_csv("Data/iris.data", header=None, skiprows=1)
iris_data.columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"]

# Extract species labels
species_labels_actual = iris_data.pop("Class")

# Transforming the iris data using PCA to 2 components
transformed_data = PCA(n_components=2).fit_transform(iris_data)

# Performing KMeans clustering with 3 clusters
kmeans_model = KMeans(n_clusters=3, n_init=10)
kmeans_model.fit(transformed_data)
cluster_labels = kmeans_model.labels_
centroids = kmeans_model.cluster_centers_

# Visualizing the KMeans clusters with explicit species colors in the legend
plt.figure(figsize=(12, 7))

# Scatter plot for KMeans clusters
scatter_kmeans = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.6)

# Scatter plot for true labels
color_map = {"setosa": 0, "versicolor": 1, "virginica": 2}
colors = [color_map[label] for label in species_labels_actual]
scatter_true = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=colors, cmap='jet', s=15, marker='x')

# Centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

# Adding the legend
legend1 = plt.legend(*scatter_kmeans.legend_elements(), loc="upper left", title="KMeans Clusters")
legend2 = plt.legend(*scatter_true.legend_elements(), loc="upper right", title="True Labels")
plt.gca().add_artist(legend1)

# Manual legend for species
from matplotlib.lines import Line2D
legend_species = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Setosa (KMeans)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Versicolor (KMeans)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, label='Virginica (KMeans)'),
    Line2D([0], [0], marker='x', color='blue', markersize=10, label='Setosa (True)'),
    Line2D([0], [0], marker='x', color='green', markersize=10, label='Versicolor (True)'),
    Line2D([0], [0], marker='x', color='red', markersize=10, label='Virginica (True)')
]
plt.legend(handles=legend_species, loc='lower right')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('KMeans Clustering vs True Labels')

# Save the plot
plt.tight_layout()
plt.savefig("Data/iris_kmeans_3clust_vs_true_explicit.png")

# Show the plot
plt.show()

# Tableau de contingence
print("Tableau de contingence :")
contingency_table = pd.crosstab(cluster_labels, species_labels_actual)
print(contingency_table)

# Indice de silhouette
print("Indice de silhouette :")
from sklearn.metrics import silhouette_score
silhouette_score = silhouette_score(transformed_data, cluster_labels)
print(silhouette_score)


