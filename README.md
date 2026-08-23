# 🏦 KBA Credit Risk & Alternative Data AutoML Engine

> An enterprise-grade, explainable AutoML platform designed for commercial banks, microfinance institutions (MFIs), and SACCOs in Kenya and East Africa. Enables automated layering of multi-source alternative data, TabFM foundation modeling, GBDT multi-model benchmarking, TreeSHAP adverse action reason codes, and dual-persona MLOps artifact delivery.

---

## 📖 Master Technical Methodology Guide
For a deep dive into data security, privacy compliance (Kenya DPA 2019), cloud hosting, DuckDB lakehouse storage, metadata catalogs, and machine learning mathematics, read the **[METHODOLOGY.md](METHODOLOGY.md)** guide.

---

## 🌟 Key Capabilities

### 1. 🔒 Security & Ephemeral In-Memory Execution
- **Zero-PII Disk Persistence**: Processes loan files and mobile telemetry strictly in ephemeral RAM (`duckdb.connect(':memory:')`).
- **Cryptographic Pseudonymization**: Salted non-reversible HMAC-SHA256 identity masking.
- **Dead-Letter Queue (DLQ)**: Quarantines malformed records and enforces strict data quality gates.

### 2. 📡 Multi-Source Alternative Data Layering
- **Macroeconomic & County GCP Indicators**: KNBS County Gross Product and inflation data (`macro_layer.db`).
- **Mobile Money Velocity**: M-Pesa 30d/90d inflow velocity ratios, Lipa Na M-Pesa turnover, and Fuliza overdraft utilization.
- **Geospatial & Satellite Telemetry**: Sentinel-2 NDVI vegetative health and CHIRPS rainfall precipitation anomalies.
- **Dynamic Vectorized DuckDB Joins**: Vectorized joins via `country_code` and temporal dimensions.

### 3. 🤖 TabFM & PyCaret AutoML Engine
- **Multi-Model Benchmarking**:
  - **TabFM (Tabular Foundation Model)**: Deep tabular embeddings with residual layers.
  - **LightGBM (`LGBMClassifier`)**: Fast histogram gradient boosting.
  - **XGBoost (`XGBClassifier`)**: Regularized depth-wise gradient boosting.
  - **CatBoost (`CatBoostClassifier`)**: Oblivious symmetric decision trees.
  - **Soft-Voting Ensemble (`VotingClassifier`)**: Automated probability-blended champion ensemble.
- **Actuarial Metric Optimization**: Stratified 5-Fold Cross-Validation targeting **PR-AUC (Precision-Recall AUC)** and **Gini Coefficient** for imbalanced credit portfolios.

### 4. 🔍 Explainable AI (XAI) & TreeSHAP Compliance
- **Recursive Ensemble TreeSHAP**: Recursively extracts attributions across soft-voting ensemble sub-models.
- **Portfolio-Wide Feature Importance**: Interactive horizontal bar charts (`mean(|SHAP|)`), beeswarm plots, and ranked importance tables.
- **Adverse Action Reason Codes**: Plain-English percentage breakdowns of top risk drivers per borrower for regulatory compliance.

### 5. 💼 Dual-Persona Architecture & MLOps Hub
- **Business & Risk Analysts**: 1-Click scored portfolio exports in **CSV** and **Excel** with predicted PD% and credit limit recommendations.
- **Data Scientists & ML Engineers**: Serialized champion pipeline (`champion_pipeline.pkl`), automated batch inference script (`infer_credit_model.py`), analytical feature store (`.parquet`), and benchmark leaderboards.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+
- `libgomp1` (Debian/Ubuntu for OpenMP support)

### Installation
```bash
git clone https://github.com/Bluuclues/CreditRisk_AutoML.git
cd CreditRisk_AutoML
pip install -r requirements.txt
```

### Running Locally
```bash
streamlit run app.py
```
