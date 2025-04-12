# main.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load Data
df = pd.read_csv('Mall_Customers.csv')   
print(df.head())

# Basic EDA
print("\nDataset Info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDescriptive Stats:")
print(df.describe())


 
# Spending Score vs Income
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
plt.title('Spending Score vs Income')
plt.show()

# Data Preprocessing for Clustering
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal K
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply KMeans with optimal K (e.g., 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 2D PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:,0]
df['PCA2'] = X_pca[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title('Customer Segments (PCA-reduced)')
plt.show()

# Cluster profiles
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='Cluster', y='Annual Income (k$)')
plt.title('Income by Cluster')
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='Cluster', y='Spending Score (1-100)')
plt.title('Spending Score by Cluster')
plt.show()

