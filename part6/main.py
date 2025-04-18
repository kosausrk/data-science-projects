import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

# 1) Grab 1yr of adjusted closes for a dozen tickers
tickers = ["AAPL","MSFT","GOOG","AMZN","JPM","XOM","JNJ","TSLA","WMT","PG","DIS","NVDA"]
df = yf.download(tickers, period="1y", auto_adjust=True, progress=False)["Close"].dropna(axis=1)
rets = df.pct_change().dropna()

# 2) One‑shot PCA & loadings heatmap
pca = PCA(n_components=5).fit(rets)
loadings = pd.DataFrame(pca.components_.T,
                        index=rets.columns,
                        columns=[f"PC{i+1}" for i in range(5)])
plt.figure(figsize=(8,6))
sns.heatmap(loadings.iloc[:,:2], annot=True, cmap="coolwarm")
plt.title("PC1 & PC2 Loadings by Ticker")
plt.show()

# 3) Scree of explained variance
plt.figure()
plt.plot(np.arange(1,6), pca.explained_variance_ratio_, "o-")
plt.xlabel("Principal Component"); plt.ylabel("Variance Ratio")
plt.title("Scree Plot")
plt.show()

# 4) Rolling PCA (60‑day window) to track PC1/PC2 variance over time
window = 60
explained, dates = [], []
for i in range(window, len(rets)):
    sub = rets.iloc[i-window:i]
    p = PCA(n_components=2).fit(sub)
    explained.append(p.explained_variance_ratio_)
    dates.append(rets.index[i])
expl_df = pd.DataFrame(explained, index=dates, columns=["PC1","PC2"])
plt.figure(figsize=(10,4))
expl_df.plot(ax=plt.gca())
plt.title("Rolling Explained Variance (60‑day window)")
plt.ylabel("Variance Ratio")
plt.show()



# --- APPEND: Hierarchical Risk‑Parity Portfolio ---

# 5) Compute covariance & distance matrix
cov = rets.cov()
corr = rets.corr()
dist = np.sqrt(0.5 * (1 - corr))

# 6) Hierarchical clustering & dendrogram
Z = linkage(dist, method="single")
plt.figure(figsize=(8,4))
dendrogram(Z, labels=rets.columns, orientation="top")
plt.title("Stock Clustering Dendrogram")
plt.tight_layout()
plt.show()

# 7) Determine leaf order
ordered = rets.columns[leaves_list(Z)].tolist()

# 8) Recursive bisection for HRP weights
def hrp_weights(cov, labels):
    if len(labels) == 1:
        return pd.Series(1.0, index=labels)
    # split in two halves
    split = len(labels) // 2
    left, right = labels[:split], labels[split:]
    varL = cov.loc[left, left].values.diagonal().sum()
    varR = cov.loc[right, right].values.diagonal().sum()
    wL = varR / (varL + varR)
    wR = varL / (varL + varR)
    w_left  = hrp_weights(cov, left)  * wL
    w_right = hrp_weights(cov, right) * wR
    return w_left.append(w_right)

weights = hrp_weights(cov, ordered)
weights /= weights.sum()

# 9) Plot HRP weights
plt.figure(figsize=(6,4))
weights.sort_values().plot.barh()
plt.title("HRP Portfolio Weights")
plt.show()

# 10) Backtest performance
port_rets   = (rets * weights).sum(axis=1)
cum_returns = (1 + port_rets).cumprod()
sharpe      = port_rets.mean() / port_rets.std() * np.sqrt(252)

fig, axes = plt.subplots(1,2, figsize=(12,4))
cum_returns.plot(ax=axes[0]); axes[0].set_title("Cumulative Returns")
axes[1].bar(["HRP"], [sharpe]); axes[1].set_title("Annualized Sharpe")
plt.tight_layout()
plt.show()
