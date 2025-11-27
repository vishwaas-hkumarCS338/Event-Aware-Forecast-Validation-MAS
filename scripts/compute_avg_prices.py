# scripts/compute_avg_prices.py
import duckdb
import json
from pathlib import Path

con = duckdb.connect("data/sales.duckdb")  # or :memory: if using parquet path
# compute product-level average price (ignore negative/zero)
q = """
SELECT LOWER(item_type) as product_type,
       AVG(retail_sales) as avg_price,
       COUNT(*) as rows
FROM 'data/sales.parquet'
WHERE retail_sales > 0
GROUP BY LOWER(item_type)
"""
df = con.execute(q).fetchdf()
map_out = {}
for idx, row in df.iterrows():
    prod = row['product_type']
    avgp = float(row['avg_price']) if row['avg_price'] is not None else None
    map_out[prod] = {"avg_price": avgp, "rows": int(row['rows'])}

out_path = Path("data/avg_price_map.json")
out_path.write_text(json.dumps(map_out, indent=2))
print(f"Wrote avg_price_map to {out_path} with {len(map_out)} products.")
