# part7_names_streamgraph.py

import os
import zipfile
import requests
import pandas as pd
import matplotlib.pyplot as plt

# 0) Make sure part7/ exists
os.makedirs("part7", exist_ok=True)

# 1) Download & extract SSA baby‑names data
url = "https://www.ssa.gov/oact/babynames/names.zip"
r = requests.get(url)
with open("part7/names.zip", "wb") as f:
    f.write(r.content)
with zipfile.ZipFile("part7/names.zip", "r") as z:
    z.extractall("part7/names")

# 2) Read all years into one DataFrame
years = range(1880, 2021)
records = []
for y in years:
    path = f"part7/names/yob{y}.txt"
    dfy = pd.read_csv(path, names=["Name","Sex","Count"])
    dfy["Year"] = y
    records.append(dfy)
df = pd.concat(records, ignore_index=True)

# 3) Compute each name’s share per year & sex
df["Total"] = df.groupby(["Year","Sex"])["Count"].transform("sum")
df["Share"] = df["Count"] / df["Total"]

# 4) Select top 10 names by share for each Year×Sex
top10 = df.groupby(["Year","Sex"], group_keys=False).apply(
    lambda g: g.nlargest(10, "Share")
)

# 5) Pivot to wide format: index=Year, columns=(Sex,Name), values=Share
wide = top10.pivot_table(
    index="Year",
    columns=["Sex","Name"],
    values="Share",
    fill_value=0
)



# --- after you’ve built `wide` as before ---

import matplotlib.pyplot as plt
import seaborn as sns

# 1) Pick the 10 names per sex with highest *average* share over all years
top_boys  = wide['M'].mean().nlargest(10).index
top_girls = wide['F'].mean().nlargest(10).index

# 2) Subset to just those names
wide_b = wide['M'][top_boys]
wide_g = wide['F'][top_girls]

# 3) Prepare a shared color palette
palette = sns.color_palette("tab10", n_colors=10)

# 4) Plot
fig, axes = plt.subplots(2,1, figsize=(14,8), sharex=True)

for ax, data, title, names in zip(
    axes,
    [wide_b, wide_g],
    ["Top 10 Boys' Names", "Top 10 Girls' Names"],
    [top_boys, top_girls]
):
    ax.stackplot(
        data.index, 
        data.T, 
        baseline="wiggle", 
        labels=names, 
        colors=palette,
        alpha=0.9
    )
    ax.set_title(f"{title} Streamgraph", fontsize=14)
    ax.set_ylim(-data.values.max()*1.1, data.values.max()*1.1)
    ax.set_ylabel("Share")
    # move legend off to the right
    ax.legend(
        loc="upper left", 
        bbox_to_anchor=(1.02, 1), 
        fontsize="small", 
        ncol=1, 
        frameon=False
    )

axes[-1].set_xlabel("Year")
plt.subplots_adjust(right=0.75, hspace=0.3)  # make room for legends
plt.tight_layout()
plt.savefig("part7/data1.png", dpi=300)
plt.show()



# 6) Plot streamgraphs for Boys (M) and Girls (F)
fig, axes = plt.subplots(2,1, figsize=(12,8), sharex=True)
for ax, sex, title in zip(axes, ["M","F"], ["Boys","Girls"]):
    data = wide[sex]
    ax.stackplot(
        data.index,
        data.T,
        baseline="wiggle",
        labels=data.columns,
        alpha=0.8
    )
    ax.set_title(f"Top 10 {title} Names Streamgraph")
    ax.legend(loc="upper right", ncol=2, fontsize="small")
axes[-1].set_xlabel("Year")
plt.tight_layout()
plt.savefig("part7/data1.png", dpi=300)
plt.show()
