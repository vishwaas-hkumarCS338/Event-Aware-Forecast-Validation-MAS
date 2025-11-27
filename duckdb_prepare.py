# duckdb_prepare.py
import duckdb, os

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "sales_full.csv")
PARQUET_PATH = os.path.join(DATA_DIR, "sales.parquet")
DUCKDB_PATH = os.path.join(DATA_DIR, "sales.duckdb")

print("Converting CSV -> Parquet (reads CSV via DuckDB). This may take a few minutes...")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

# Create DuckDB and read CSV efficiently
con = duckdb.connect(DUCKDB_PATH)

# Create a table directly from the CSV
con.execute(f"""
CREATE OR REPLACE TABLE sales AS
SELECT
    TRY_CAST(year AS INTEGER) AS year,
    month,
    supplier,
    "item code" AS item_code,
    "item description" AS item_description,
    "item type" AS item_type,
    TRY_CAST("retail sales" AS DOUBLE) AS retail_sales,
    TRY_CAST("retail transfers" AS DOUBLE) AS retail_transfers,
    TRY_CAST("warehouse sales" AS DOUBLE) AS warehouse_sales
FROM read_csv_auto('{CSV_PATH}', header=True);
""")

# Save Parquet for fast access
print("Exporting to Parquet...")
con.execute(f"COPY sales TO '{PARQUET_PATH}' (FORMAT PARQUET);")

print("✅ Done! DuckDB + Parquet created successfully.")
con.close()
