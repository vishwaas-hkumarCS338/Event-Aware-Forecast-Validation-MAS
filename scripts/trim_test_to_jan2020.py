import pandas as pd

df = pd.read_parquet("evaluations/test.parquet")
df = df[(df["year"] == 2020) & (df["month"] == 1)]
df.to_parquet("evaluations/test.parquet", index=False)

print("Trimmed test periods:", sorted(df["year"].astype(str) + "-" + df["month"].astype(str)))
