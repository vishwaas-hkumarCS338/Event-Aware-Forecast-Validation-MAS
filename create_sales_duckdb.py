# create_sales_duckdb.py
import duckdb, os
PARQUET = "data/sales.parquet"
DB = "data/sales.duckdb"

if not os.path.exists(PARQUET):
    raise SystemExit("Place sales.parquet at data/sales.parquet first.")

print("Creating DuckDB and importing parquet into table 'sales' ...")
con = duckdb.connect(DB)
# create a persistent table pointing to parquet (this copies into the .duckdb file)
con.execute(f"CREATE TABLE IF NOT EXISTS sales AS SELECT * FROM '{PARQUET}';")
print("Created/updated table 'sales' in", DB)
con.close()
