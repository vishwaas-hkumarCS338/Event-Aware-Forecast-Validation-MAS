# Event-Aware Forecast Validation Lighweight Multi-Agent System (MAS)

This repository implements a complete **Event-Aware Lightweight Multi-Agent System (MAS)** for **validating retail demand forecasts under real-world disruptions**.  
The system combines **machine learning**, **quantile calibration**, **retrieval-augmented reasoning**, and **linear programming–based supplier reallocation** inside a unified **FastAPI microservice**.

The MAS evaluates forecasts using four cooperating agents:

1. **ML Forecasting Agent** – Predicts demand quantiles (p10, p50, p90).  
2. **Event-Aware RAG + LLM Agent** – Interprets disruptions using retrieved documents.  
3. **Supply Chain Simulation & LP Agent** – Computes supplier loss, reallocation, and feasibility.  
4. **Coordinator Agent** – Orchestrates the entire workflow and returns the final verdict.

The repository includes training pipelines, calibration tools, benchmarking scripts, RAG evaluation, LP feasibility testing, and unified comparison of calibrated vs uncalibrated forecasts.

---

## 🔧 Project Features

- LightGBM quantile forecasting  
- Post-training quantile calibration (p10/p50/p90)  
- RAG retrieval using ChromaDB  
- LLM-based event reasoning  
- Supplier reallocation using Linear Programming  
- Full FastAPI microservice with Swagger UI  
- Unified comparison of calibrated vs raw ML forecasts  
- Automated evaluation metrics + visualization  

---

# 📦 Installation Guide  
Follow the steps **in this exact order**.

---

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/vishwaas-hkumarCS338/Event-Aware-Forecast-Validation-MAS.git
cd Event-Aware-Forecast-Validation-MAS
```

---

## 2️⃣ Create & Activate Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 3️⃣ Install Dependencies

```bash
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

**If `requirements.txt` is missing, install core packages:**

```bash
pip install duckdb numpy pandas scipy scikit-learn lightgbm python-dotenv openai fastapi uvicorn pydantic chromadb
```

---

## 4️⃣ Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=sk_your_key_here
VALIDATOR_DISABLE_LLM=0
VALIDATOR_LLM_TIMEOUT_S=6.0
```

**Notes:**
- If you don't have an OpenAI API key, set `VALIDATOR_DISABLE_LLM=1` to skip LLM calls.
- The retriever (ChromaDB) requires embeddings at `data/events_embeddings.npy` and metadata at `data/events_metadata.json`.

---

## 5️⃣ Prepare DuckDB Retail Dataset

Run both scripts to generate the database and feature stores:

```bash
python create_sales_duckdb.py
python duckdb_prepare.py
```

**Creates:**
- `data/sales.duckdb`
- `data/sales.parquet`
- `data/features_product.parquet`

**Expected output:** Confirmation messages in terminal; verify files exist in `data/` folder.

---

## 6️⃣ Train the ML Forecast Model

Run the full ML workflow to generate splits, train, evaluate, calibrate, and plot:

```bash
python scripts/01_make_splits.py
python scripts/02_train_lgbm_split.py
python scripts/03_eval_lgbm_split.py
python scripts/04_calibrate_quantiles.py
python scripts/05_plot_eval.py
```

**Artifacts created:**
- `models/lgbm_split_*.pkl` – Trained models
- `evaluations/` – Metrics, plots, and calibration data

**Plots appear in:** `evaluations/` (scatter, residuals, distribution plots)

---

## 7️⃣ Run Unified Comparison (Calibration vs Raw)

```bash
python scripts/08_compare_unified.py
```

**Output saved to:** `evaluations/compare_unified.json`

This file contains side-by-side quantile predictions (raw vs calibrated) for benchmark scenarios.

---

# 🚀 Running the API (Main MAS System)

### Step 1: Start the FastAPI Server (Terminal 1)

```bash
# Ensure venv is active
python -m uvicorn app:app --reload --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### Step 2: Open Swagger UI

👉 Navigate to: **http://127.0.0.1:8000/docs**

You'll see interactive API documentation with a "Try it out" button for each endpoint.

### Step 3: Submit a Validation Request

**Example Request Payload (in Swagger):**

```json
{
  "product": "wine",
  "period": "2020-01",
  "forecast_units": 150000,
  "disrupted_supplier": "RELIABLE CHURCHILL LLLP"
}
```
---

# 📊 Running Evaluation Pipelines (RAG + LP)

After submitting a validation request through the API (which populates logs), run the evaluators.

**Open Terminal 2:**

```bash
cd scripts
python 06_eval_rag_llm.py
python 07_eval_lp.py
```

**Outputs saved in:**
- `evaluations/rag_eval.jsonl` – RAG + LLM reasoning details
- `evaluations/lp_eval.json` – Linear Programming feasibility results

---

# ⚙️ Project Structure

```plaintext
.
├── app.py                          # FastAPI microservice
├── validation_agent.py             # Main agentic orchestration logic
├── ml_forecast.py                  # ML model (uncalibrated)
├── ml_forecast_calibrated.py       # Calibrated inference logic
├── reallocator.py                  # Linear Programming reallocation engine
├── retriever.py                    # RAG retriever (ChromaDB)
├── retriever_simple.py             # Simple cosine similarity retriever
├── indexer.py                      # Embedding indexer
├── 
├── scripts/
│   ├── 01_make_splits.py           # Create train/val/test splits
│   ├── 02_train_lgbm_split.py      # Train quantile models
│   ├── 03_eval_lgbm_split.py       # Evaluate on test set
│   ├── 04_calibrate_quantiles.py   # Post-training calibration
│   ├── 05_plot_eval.py             # Generate plots
│   ├── 06_eval_rag_llm.py          # Evaluate RAG + LLM reasoning
│   ├── 07_eval_lp.py               # Test LP solver feasibility
│   └── 08_compare_unified.py       # Compare calibrated vs raw
│
├── data/
│   ├── sales_full.csv              # Raw retail sales dataset
│   ├── sales.duckdb                # DuckDB database (after prepare)
│   ├── sales.parquet               # Parquet format (after prepare)
│   ├── features_product.parquet    # Computed features
│   ├── avg_price_map.json          # Product average prices
│   ├── events_embeddings.npy       # RAG embeddings (optional)
│   ├── events_metadata.json        # RAG metadata (optional)
│   └── chroma_persist/             # ChromaDB persistence (optional)
│
├── evaluations/
│   ├── metrics.json                # Training metrics
│   ├── train_meta.json             # Split metadata
│   ├── ml_backtest_summary.json    # Backtest results
│   ├── calibration.json            # Calibration parameters
│   ├── compare_unified.json        # Raw vs calibrated comparison
│   ├── rag_eval.jsonl              # RAG evaluation logs
│   ├── lp_eval.json                # LP feasibility results
│   └── plots/                      # Generated PNG plots
│
├── models/
│   ├── lgbm_split_*.pkl            # Trained LightGBM models
│   └── scaler_*.pkl                # Feature scalers
│
├── src/validation/
│   └── supplierValidation.ts       # TypeScript supplier validator
│
├── utils/
│   └── utils.py                    # Helper functions
│
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (create this)
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

---

# 📈 Key Outputs Generated

### 1. Machine Learning Metrics

**File:** `evaluations/metrics.json`

Includes:
- **Pinball loss** (p10, p50, p90)
- **MAE, RMSE, MAPE** (accuracy metrics)
- **Coverage %** (confidence interval coverage)
- **Winkler interval width** (interval score)

### 2. Calibration Comparison

**File:** `evaluations/compare_unified.json`

Contains raw vs calibrated quantiles for the same scenario:
```json
{
  "scenarios": [
    {
      "product": "wine",
      "period": "2020-01",
      "forecast_units": 150000,
      "raw_quantiles": { "p10": 120000, "p50": 145000, "p90": 170000 },
      "calibrated_quantiles": { "p10": 125000, "p50": 148000, "p90": 168000 }
    }
  ]
}
```

### 3. RAG + LLM Verdict

**File:** `evaluations/rag_eval.jsonl`

Each line is a JSON object:
```json
{
  "product": "wine",
  "period": "2020-01",
  "verdict": "OVER-ESTIMATING",
  "confidence": 72,
  "reasons": ["seasonal uplift", "supplier disruption"],
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### 4. LP Supply Feasibility

**File:** `evaluations/lp_eval.json`

```json
{
  "scenario": "wine_2020-01_disrupted",
  "solver_status": "optimal",
  "total_cost": 5234.50,
  "allocation": [
    { "supplier": "E & J GALLO WINERY", "alloc_units": 50000 },
    { "supplier": "CONSTELLATION BRANDS", "alloc_units": 40000 }
  ],
  "feasible": true
}
```

---

# 🧪 End-to-End Demonstration (Complete Workflow)

To run the entire project from data creation to API demonstration in one session:

### Terminal 1: Training & Data Prep

```bash
# Activate venv (if not already active)
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # Mac/Linux

# Data & Training Pipeline
python create_sales_duckdb.py
python duckdb_prepare.py
python scripts/01_make_splits.py
python scripts/02_train_lgbm_split.py
python scripts/03_eval_lgbm_split.py
python scripts/04_calibrate_quantiles.py
python scripts/05_plot_eval.py
python scripts/08_compare_unified.py

# Expected: All evaluation artifacts in evaluations/
ls evaluations/
```

### Terminal 2: Start API Server

```bash
# Activate venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # Mac/Linux

python -m uvicorn app:app --reload --port 8000
# Navigate to: http://127.0.0.1:8000/docs
```



### Terminal 1: Run Evaluation Pipelines

```bash
cd scripts
python 06_eval_rag_llm.py
python 07_eval_lp.py

# Check outputs
type ..\evaluations\rag_eval.jsonl
type ..\evaluations\lp_eval.json
```

---

# 🧹 Troubleshooting & Common Issues

### Issue 1: `ModuleNotFoundError: No module named 'duckdb'`

**Solution:**
```bash
pip install duckdb
```

### Issue 2: `FileNotFoundError: data/sales.duckdb`

**Solution:** Run the data preparation script first:
```bash
python create_sales_duckdb.py
python duckdb_prepare.py
```

### Issue 3: `OPENAI_API_KEY not set`

**Solution:** Either:
- Add valid key to `.env`: `OPENAI_API_KEY=sk_...`
- Or disable LLM: `VALIDATOR_DISABLE_LLM=1` in `.env`

### Issue 4: LLM calls timeout or fail

**Solution:** Increase timeout in `.env`:
```bash
VALIDATOR_LLM_TIMEOUT_S=10.0
```

### Issue 5: `supplier_found: false` even for known suppliers

**Solution:** Check that supplier name is in extended_suppliers list in `validation_agent.py`:
```python
extended_suppliers = ["RELIABLE CHURCHILL LLLP"]  # Add your supplier here
```

### Issue 6: No plots generated in `evaluations/`

**Solution:** Ensure matplotlib is installed:
```bash
pip install matplotlib seaborn
```

---

# ✅ Project Status

- Fully functional
- End-to-end ML + RAG + LP pipeline
- FastAPI microservice with Swagger UI
- Calibration & comparison tools
- Ready for demonstration, benchmarking, and research integration

---

# 📚 Key References

- **DuckDB:** https://duckdb.org/docs/
- **FastAPI:** https://fastapi.tiangolo.com/
- **LightGBM:** https://lightgbm.readthedocs.io/
- **ChromaDB:** https://docs.trychroma.com/
- **SciPy Linear Programming:** https://docs.scipy.org/doc/scipy/reference/optimize.linprog.html

---

**Last Updated:** November 2025  
**Status:** Production-Ready
