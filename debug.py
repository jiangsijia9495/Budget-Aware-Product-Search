import pandas as pd

path = "data/processed/biencoder_pairs_v1.parquet"
df = pd.read_parquet(path)

positive = df["positive"].fillna("").astype(str).str.strip()
negative = df["negative"].fillna("").astype(str).str.strip()
bad = positive.eq(negative)

cols = [
    "query_id",
    "anchor",
    "positive_product_id",
    "negative_product_id",
    "positive_label",
    "negative_label",
]

print(f"Identical-text rows: {int(bad.sum())}")
print(df.loc[bad, cols].to_string(index=False))

for _, row in df.loc[bad].iterrows():
    print("\n--- QUERY ---")
    print(row["anchor"])
    print("\n--- SHARED PRODUCT TEXT, FIRST 500 CHARS ---")
    print(str(row["positive"])[:500])