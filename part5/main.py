#required libraries 
# pip install pandas matplotlib seaborn prophet scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

# Load dataset
df = pd.read_csv("day.csv")

# Preprocess
df['dteday'] = pd.to_datetime(df['dteday'])
df = df[['dteday', 'cnt']].rename(columns={'dteday': 'ds', 'cnt': 'y'})

# Visualize original time series
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='ds', y='y')
plt.title("Daily Bike Rentals Over Time")
plt.xlabel("Date")
plt.ylabel("Total Rentals")
plt.tight_layout()
plt.savefig("data1.png")
plt.close()

# Train/test split
train = df.iloc[:-60]
test = df.iloc[-60:]

# Fit Prophet model
model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.fit(train)

# Forecast future
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# Plot forecast
fig1 = model.plot(forecast)
plt.title("Forecast with Prophet")
plt.xlabel("Date")
plt.ylabel("Predicted Rentals")
plt.tight_layout()
fig1.savefig("data2.png")

# Plot components
fig2 = model.plot_components(forecast)
fig2.savefig("data3.png")

# Evaluation
forecast_filtered = forecast[['ds', 'yhat']].merge(test, on='ds')
mae = mean_absolute_error(forecast_filtered['y'], forecast_filtered['yhat'])
rmse = mean_squared_error(forecast_filtered['y'], forecast_filtered['yhat'])

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Actual vs Predicted plot
plt.figure(figsize=(10, 5))
plt.plot(test['ds'], test['y'], label='Actual', marker='o')
plt.plot(forecast_filtered['ds'], forecast_filtered['yhat'], label='Predicted', marker='x')
plt.title("Actual vs Predicted Rentals")
plt.xlabel("Date")
plt.ylabel("Rentals")
plt.legend()
plt.tight_layout()
plt.savefig("data4.png")
plt.close()
