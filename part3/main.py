# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 2. Load Dataset
data = pd.read_csv('/Users/koushiksarkar/data-science-projects/part3/AmesHousing.csv')
print("Data Shape:", data.shape)
print(data.head())

# 3. Basic Visualization
plt.figure(figsize=(10,6))
sns.histplot(data['SalePrice'], bins=30, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()  # Ensure this is added to display the plot

# 4. Correlation Heatmap
plt.figure(figsize=(12,10))
top_corr = data.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)[1:11]
sns.heatmap(data[top_corr.index.tolist() + ['SalePrice']].corr(), annot=True, cmap='coolwarm')
plt.title('Top Correlated Features with SalePrice')
plt.tight_layout()
plt.show()  # Ensure this is added to display the plot

# 5. Data Preprocessing
# Drop rows with missing target
data = data.dropna(subset=['SalePrice'])

# Fill missing values
data = data.fillna(data.median(numeric_only=True))

# Encode categorical columns
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# 6. Define Features and Target
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# 7. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train Model (Random Forest)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 9. Predictions and Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# 10. Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]
features = X.columns[indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features)
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()  # Ensure this is added to display the plot
