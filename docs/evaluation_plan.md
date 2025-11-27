\# Evaluation Plan — Event-Aware Forecast Validation (Wine)



\## 1. Objectives

\- Quantify \*\*forecast realism\*\* using Quantile LightGBM (p10/p50/p90).

\- Measure \*\*predictive accuracy\*\* and \*\*interval quality\*\* (coverage/Winkler).

\- Test \*\*supply feasibility\*\* under disruptions via LP reallocation.

\- Validate \*\*explanations\*\* with RAG+LLM parse success and evidence relevance.



\## 2. Data \& Splits

\- Source: features from `data/features\_product.parquet` (category-level).

\- Target: `units` (retail\_sales + retail\_transfers + warehouse\_sales).

\- Time-based split (no leakage):

&nbsp; - \*\*Train\*\*: all months ≤ `SPLIT\_END`

&nbsp; - \*\*Test\*\*: months > `SPLIT\_END`

&nbsp; - Optional \*\*Calibration\*\*: last K months of Train for interval calibration.



\## 3. Models

\- Quantile LightGBM: α ∈ {0.1, 0.5, 0.9}; features = lags/seasonality/deltas.

\- Calibration (optional):

&nbsp; - \*\*Isotonic\*\* mapping per-quantile on calibration set.

&nbsp; - \*\*Conformal\*\* adjustment using residual quantiles for valid coverage.



\## 4. Metrics

\- Point error (p50): MAE, RMSE, MAPE.

\- Quantile quality: Pinball loss for p10/p50/p90.

\- Interval metrics (80% band = \[p10,p90]):

&nbsp; - \*\*Coverage\_80\*\* = mean( y ∈ \[p10,p90] ).

&nbsp; - \*\*Winkler\_80\*\* = mean( (p90−p10) + penalty if y outside ).

\- RAG/LLM:

&nbsp; - LLM JSON \*\*parse rate\*\*, \*\*mean top-1 similarity\*\* of retrieved chunks.



\## 5. Procedures

1\) \*\*Build features\*\* → `python features.py`  

2\) \*\*Split\*\* → `python scripts/01\_make\_splits.py --product wine --split\_end 2019-10`  

3\) \*\*Train\*\* (quantiles) → `python scripts/02\_train\_lgbm\_split.py --product wine`  

4\) \*\*Evaluate\*\* (test) → `python scripts/03\_eval\_lgbm\_split.py --product wine`  

5\) \*\*Calibrate\*\* (optional):

&nbsp;  - Isotonic → `python scripts/04\_calibrate\_quantiles.py --method isotonic`

&nbsp;  - Conformal → `python scripts/04\_calibrate\_quantiles.py --method conformal`

6\) \*\*Plot\*\* → `python scripts/05\_plot\_eval.py`

7\) \*\*Scenario API\*\* (optional) → `python evaluate\_all.py` (already available)



\## 6. Reporting

\- Artifacts in `evaluations/`:

&nbsp; - `split\_summary.json`, `preds\_test.csv`, `metrics.json`, `calibration.json`

&nbsp; - Plots: `errors.png`, `coverage.png`, `interval\_scatter.png`, `residuals.png`



