#data from kaggle https://www.kaggle.com/competitions/titanic/data 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df_train = pd.read_csv('titanic/train.csv')
df_test = pd.read_csv('titanic/test.csv')
df_gender_submission = pd.read_csv('titanic/gender_submission.csv')

# Display first few rows
print("Train Data:")
print(df_train.head())
print("\nTest Data:")
print(df_test.head())
print("\nGender Submission Data:")
print(df_gender_submission.head())

# Data Overview
print("\nTrain Data Info:")
df_train.info()
print("\nMissing Values:")
print(df_train.isnull().sum())

# Visualization of missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df_train.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Values in Train Dataset")
plt.show()

# Survival Rate Analysis
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df_train, palette='Set1')
plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Class Distribution and Survival
plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', hue='Survived', data=df_train, palette='coolwarm')
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_train['Age'].dropna(), bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_train.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()
