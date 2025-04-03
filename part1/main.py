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

# 🔹 2. Age Distribution of Survivors vs. Non-Survivors
plt.figure(figsize=(10, 6))
sns.histplot(df_train[df_train["Survived"] == 1]["Age"], bins=30, kde=True, color="green", label="Survived")
sns.histplot(df_train[df_train["Survived"] == 0]["Age"], bins=30, kde=True, color="red", label="Did Not Survive")
plt.title("Age Distribution: Survivors vs. Non-Survivors")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.show()

# 🔹 3. Fare Distribution by Class
plt.figure(figsize=(10, 5))
sns.boxplot(x="Pclass", y="Fare", data=df_train, palette="viridis")
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.ylim(0, 300)  # Avoid extreme outliers
plt.show()

# 🔹 4. Gender & Survival Relationship
plt.figure(figsize=(6, 4))
sns.barplot(x="Sex", y="Survived", data=df_train, palette="pastel")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.show()
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
