#Data via Kaggle; https://www.kaggle.com/datasets/gregorut/videogamesales?resource=download 

# Data from Kaggle: https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df_games = pd.read_csv('vgsales.csv')  # Adjust path to your actual file

# Display first few rows
print("Video Game Sales Data:")
print(df_games.head())

# 🔹 1. Top Genres by Game Count
plt.figure(figsize=(10, 5))
sns.countplot(data=df_games, x='Genre', order=df_games['Genre'].value_counts().index, palette="Set3")
plt.title("Top Genres by Game Count")
plt.xticks(rotation=45)
plt.ylabel("Number of Games")
plt.tight_layout()
plt.show()

# 🔹 2. Global Sales by Platform
plt.figure(figsize=(10, 6))
platform_sales = df_games.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False)
sns.barplot(x=platform_sales.index, y=platform_sales.values, palette="coolwarm")
plt.title("Total Global Sales by Platform")
plt.xlabel("Platform")
plt.ylabel("Global Sales (millions)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 🔹 3. Sales Correlation Between Regions
plt.figure(figsize=(8, 6))
sns.heatmap(df_games[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].corr(), annot=True, cmap='YlGnBu', fmt='.2f')
plt.title("Sales Correlation by Region")
plt.tight_layout()
plt.show()

# 🔹 4. Global Sales Over Time
plt.figure(figsize=(10, 5))
df_year_sales = df_games.groupby("Year")["Global_Sales"].sum().sort_index()
sns.lineplot(x=df_year_sales.index, y=df_year_sales.values, marker='o', color='purple')
plt.title("Total Global Sales Over Time")
plt.xlabel("Release Year")
plt.ylabel("Global Sales (millions)")
plt.tight_layout()
plt.show()
