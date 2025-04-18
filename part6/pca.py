import yfinance as yf
import pandas as pd


tickers = [
    "AAPL",  # Tech
    "MSFT",  
    "GOOG",  
    "AMZN",  # E‑comm
    "JPM",   # Finance
    "XOM",   # Energy
    "JNJ",   # Healthcare
    "TSLA",  # Auto
    "WMT",   # Retail
    "PG",    # Consumer Staples
    "DIS",   # Entertainment
    "NVDA"   # Semiconductors
]

# 2) Download 1 year of daily closes
df = yf.download(
    tickers,
    period="1y",
    interval="1d",
    auto_adjust=True,    # gives you adjusted closes directly
    progress=False
)["Close"]

# 3) Drop any tickers with missing data
df = df.dropna(axis=1)

print("Downloaded data shape:", df.shape)
print(df.head())


# Compute daily returns
rets = df.pct_change().dropna()

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=5).fit(rets)

# Scree plot
import matplotlib.pyplot as plt
plt.plot(range(1,6), pca.explained_variance_ratio_, "o-")
plt.xlabel("PC"); plt.ylabel("Variance Ratio"); plt.show()

# 2D projection
pcs2 = pca.transform(rets)[:,:2]
import seaborn as sns
sns.scatterplot(x=pcs2[:,0], y=pcs2[:,1])
plt.title("Tickers in PC-space"); plt.show()
