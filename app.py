import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import sqlite3
import os
import io
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import uuid
import streamlit.components.v1 as components
from typing import Optional, Dict, Any, List, Tuple

# Import custom backend modules
from modules.data_validator import CreditRiskDataValidator
from modules.feature_store import apply_macro_layers, export_parquet_snapshot, export_parquet_bytes, export_csv_bytes
from modules.models.dispatcher import run_automl_pipeline
from modules.models.shap_explainer import CreditRiskExplainer
from modules.eda_visualizer import CreditRiskEDA

# Streamlit Page Config - Wide Layout
st.set_page_config(
    page_title="KBA Credit Risk AutoML Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS INJECTION ---
def load_local_css(file_name: str) -> None:
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_local_css("style.css")

# Base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALTERNATIVE_DATA_DIR = os.path.join(BASE_DIR, "Data", "Alternative_Data")

# --- COUNTRY CODE MAPPING ---
@st.cache_data
def load_country_mapping() -> dict:
    db_path = os.path.join(ALTERNATIVE_DATA_DIR, 'countries.db')
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        try:
            df = pd.read_sql("SELECT * FROM country_mapping ORDER BY country_name", conn)
            conn.close()
            return dict(zip(df['country_name'], df['country_code']))
        except Exception:
            conn.close()
    return {"Kenya": "KEN", "Tanzania": "TZA", "Uganda": "UGA", "Rwanda": "RWA"}

COUNTRY_MAPPING = load_country_mapping()

# --- DATA SOURCES REGISTRY CATALOG ---
DATA_SOURCES_CATALOG = [
    {
        "variable": "Gross County Product (GCP) & Per Capita GDP",
        "category": "Macroeconomic & Output",
        "collection_method": "API via OpenData Platform; https://kenya.opendataforafrica.org/ivpwyob/gdp-expenditure",
        "reference": "Kenya OpenData / KNBS",
        "url": "https://kenya.opendataforafrica.org/ivpwyob/gdp-expenditure",
        "last_updated": "Annual (KNBS 2024)",
        "status": "Pending",
        "actuarial_rationale": "County-level economic output, real GDP growth, and per capita economic productivity used to index regional purchasing power and shock resistance."
    },
    {
        "variable": "Google Trends Financial Distress Search Index",
        "category": "Behavioral & Search Trends",
        "collection_method": "Google Trends API (pytrends Python library)",
        "reference": "Google Trends Behavioral Telemetry",
        "url": "https://trends.google.com/trends/",
        "last_updated": "Rolling Monthly",
        "status": "Pending",
        "actuarial_rationale": "Search volume intensity for financial distress keywords (e.g., loan penalties, debt collectors, auctioneers) per geographic region."
    },
    {
        "variable": "KAMIS Wholesale Staple Food Price Volatility",
        "category": "Commodities & Agriculture",
        "collection_method": "Kenya Agricultural Market Information System (KAMIS) API / Scraping (https://kamis.kilimo.go.ke/)",
        "reference": "Ministry of Agriculture & Livestock Development (KAMIS)",
        "url": "https://kamis.kilimo.go.ke/",
        "last_updated": "Weekly Market Releases",
        "status": "Pending",
        "actuarial_rationale": "Wholesale commodity price variance for maize, beans, potatoes, and staple cereals across county markets affecting household disposable income."
    },
    {
        "variable": "Dynamic Tax Movements",
        "category": "Fiscal & Statutory",
        "collection_method": "KRA & National Treasury Gazette Notices & Finance Act Amendments",
        "reference": "Kenya Revenue Authority (KRA) & National Treasury",
        "url": "https://www.kra.go.ke/",
        "last_updated": "Fiscal / Gazette Cycle",
        "status": "Pending",
        "actuarial_rationale": "Statutory tax rate shifts, excise changes, withholding VAT adjustments, and turnover tax modifications impacting MSME operating margins."
    },
    {
        "variable": "EPRA Monthly Fuel & Transport Tariff Changes",
        "category": "Energy & Transport",
        "collection_method": "EPRA Monthly Public Price Press Releases / Scraping (https://www.epra.go.ke/)",
        "reference": "Energy and Petroleum Regulatory Authority (EPRA)",
        "url": "https://www.epra.go.ke/",
        "last_updated": "14th of Every Month",
        "status": "Pending",
        "actuarial_rationale": "Super Petrol, Diesel, and Kerosene pump price caps across major county pricing zones tracking supply chain input costs."
    },
    {
        "variable": "OpenStreetMap (OSM) Commercial POI Density",
        "category": "Geospatial & Footfall",
        "collection_method": "Overpass API / OSM Geofabrik Extracts (https://overpass-turbo.eu/)",
        "reference": "OpenStreetMap Contributors & Overpass Turbo",
        "url": "https://overpass-turbo.eu/",
        "last_updated": "Quarterly Geospatial Aggregation",
        "status": "Pending",
        "actuarial_rationale": "Density of commercial retail points of interest (POIs), financial agents, markets, and transit hubs within borrower vicinity."
    },
    {
        "variable": "KNBS County Multidimensional Poverty Index (MPI)",
        "category": "Socio-Economic Deprivation",
        "collection_method": "KNBS MPI data and other data can be scraped (https://www.knbs.or.ke/)",
        "reference": "Kenya National Bureau of Statistics (KNBS)",
        "url": "https://www.knbs.or.ke/",
        "last_updated": "Periodic Survey Releases",
        "status": "Pending",
        "actuarial_rationale": "County deprivation indices measuring health, education, living standards, and economic vulnerability."
    },
    {
        "variable": "Central Bank Rate (CBR) & Interbank Rate Spread",
        "category": "Monetary & Liquidity",
        "collection_method": "Central Bank of Kenya (CBK) Weekly / Monthly Statistical Bulletin & API",
        "reference": "Central Bank of Kenya (CBK)",
        "url": "https://www.centralbank.go.ke/",
        "last_updated": "Monetary Policy Committee (MPC) Cycle",
        "status": "Pending",
        "actuarial_rationale": "Benchmark CBR, interbank lending rate spread, and Kenya Shilling sovereign yield movements reflecting systemic credit liquidity."
    },
    {
        "variable": "M-Pesa Transaction Volume & Velocity",
        "category": "Mobile Money Telemetry",
        "collection_method": "Safaricom Daraja API / Consented M-Pesa Statement Parsing",
        "reference": "Safaricom Daraja Developer Platform & Mobile Statements",
        "url": "https://developer.safaricom.co.ke/",
        "last_updated": "Real-time / 30d-90d Rolling Window",
        "status": "Pending",
        "actuarial_rationale": "30d/90d turnover velocity, Fuliza overdraft utilization rate, and peer-to-peer liquidity cycles for thin-file underwriting."
    }
]

# --- SINGLE-PAGE SESSION STATE INITIALIZATION ---
def init_single_page_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'sqlite_conn' not in st.session_state:
        st.session_state.sqlite_conn = sqlite3.connect(':memory:', check_same_thread=False)
    if 'duck_conn' not in st.session_state:
        st.session_state.duck_conn = duckdb.connect(':memory:')
    
    # Progressive Disclosure Section Flags
    if 'data_ingested' not in st.session_state:
        st.session_state.data_ingested = False
    if 'layers_applied' not in st.session_state:
        st.session_state.layers_applied = False
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False

    # Data Payloads
    if 'primary_df' not in st.session_state:
        st.session_state.primary_df = None
    if 'final_layered_df' not in st.session_state:
        st.session_state.final_layered_df = None
    if 'validation_messages' not in st.session_state:
        st.session_state.validation_messages = []

    # AutoML & XAI Payloads
    if 'automl_results' not in st.session_state:
        st.session_state.automl_results = None
    if 'selected_borrower_idx' not in st.session_state:
        st.session_state.selected_borrower_idx = 0

init_single_page_state()


# ==============================================================================
# TOP CANVAS HEADER (PERSISTENT ACROSS TABS)
# ==============================================================================
top_header_col1, top_header_col2 = st.columns([1, 1], gap="large")

with top_header_col1:
    st.markdown('<div class="massive-title">CREDIT RISK<br>AUTOML</div>', unsafe_allow_html=True)
    st.markdown("""
    **Kenya Bankers Association (KBA) Credit Scoring & Alternative Data Initiative**  
    *Developed by Francis Nyangoma, Linda Kimilu, and Shirleen Chege.*
    
    This end-to-end Automated Machine Learning (AutoML) platform evaluates thin-file and informal sector borrowers by **fusion of panel repayment records with multi-source alternative data**.
    """)

with top_header_col2:
    st.info("🔒 **Privacy Guarantee:** Customer data is processed in ephemeral RAM. Non-reversible SHA-256 pseudonymization protects individual borrower identities (Kenya DPA 2019).")
    
    with st.popover("ℹ️ Ephemeral Architecture & Privacy Standards"):
        st.markdown("""
        ### 🔒 Security, Ephemeral RAM & Regulatory Compliance
        
        * **Zero-Persistence Guarantee:**  
          Uploaded loan portfolios and borrower telemetry are held strictly in temporary RAM (`duckdb.connect(':memory:')`). No persistent database or borrower record is saved to disk.
        * **Salted SHA-256 Pseudonymization:**  
          Primary customer identifiers (National ID, MSISDN) are converted into non-reversible hashes:
          $$\\text{Pseudonymized ID} = \\text{HMAC-SHA256}(\\text{Identifier}, \\text{Salt})$$
        * **Session Isolation:**  
          Each session receives an ephemeral UUID (`uuid.uuid4()`). Closing the browser or resetting the session instantly wipes the in-memory DuckDB warehouse, fully complying with **Kenya Data Protection Act (DPA 2019)** and **CBK Digital Credit Provider (DCP) Regulations 2022**.
        """)


# ==============================================================================
# TOP-LEVEL TWO-TAB NAVIGATION
# ==============================================================================
tab_engine, tab_sources = st.tabs([
    "⚡ Credit Risk AutoML Engine", 
    "📚 Data Sources & Methodology"
])


# ##############################################################################
# TAB 1: CREDIT RISK AUTOML ENGINE (OPERATIONAL WORKFLOW)
# ##############################################################################
with tab_engine:

    # ==============================================================================
    # SECTION 1: HEADER & PORTFOLIO SETUP (TOP CANVAS)
    # ==============================================================================
    with st.expander("📄 View Data Upload Guidelines & Template", expanded=False):
        st.markdown("""
        Uploaded `.csv` files must contain standard credit panel features:
        * `borrower_id`: Unique client identifier.
        * `amount`: Requested principal amount (numeric KES).
        * `tenure_days`: Duration in days.
        * `default_flag`: Target Variable (`1` = Defaulted, `0` = Performing).
        * `loan_date`: Origination date (used for temporal macro joins).
        """)
        
        sample_csv = (
            "borrower_id,borrower_type,loan_no,loan_date,due_date,payoff_date,tenure_days,amount,default_flag,country_code\n"
            "CUST-10492,Retail,LN-88102,2025-03-14,2025-04-14,2025-04-14,30,50000,1,KEN\n"
            "CUST-20831,SME,LN-88103,2025-03-15,2025-04-15,2025-04-10,30,120000,0,KEN\n"
            "CUST-30119,Microfinance,LN-88104,2025-03-16,2025-04-16,2025-04-16,30,25000,0,KEN\n"
        )
        st.download_button(
            label="⬇️ Download Sample Credit Panel CSV",
            data=sample_csv,
            file_name="kba_sample_credit_panel.csv",
            mime="text/csv"
        )

    if not st.session_state.data_ingested:
        col_s1_title, col_s1_info = st.columns([4, 1])
        with col_s1_title:
            st.subheader("1. Portfolio Setup & Ingestion")
        with col_s1_info:
            with st.popover("ℹ️ Ingestion Quality Gate & DLQ"):
                st.markdown("""
                ### 🛡️ Data Quality Gate & Dead-Letter Queue (DLQ)
                
                The `CreditRiskDataValidator` executes rigorous sanity checks on incoming loan data:
                * **Mandatory Column Check:** Verifies `borrower_id`, `amount`, and `default_flag`.
                * **Binary Target Cleanliness:** Checks `default_flag` $\\in \\{0, 1\\}$. Malformed or missing targets are moved to the **Dead-Letter Queue (DLQ)** to prevent corrupting model loss gradients.
                * **Imputation & Bounding:** Missing tenors default to 30 days; non-positive principal amounts are imputed with the portfolio median.
                """)
        
        country_list = list(COUNTRY_MAPPING.keys())
        default_idx = country_list.index("Kenya") if "Kenya" in country_list else 0
        selected_country_name = st.selectbox("Country Jurisdiction:", country_list, index=default_idx, help="Select national jurisdiction for macroeconomic and regulatory matching.")
        selected_country_code = COUNTRY_MAPPING[selected_country_name]

        uploaded_file = st.file_uploader("Upload Panel Loan Data (.CSV)", type=["csv"], help="Upload CSV containing borrower panel repayments and default flags.")

        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            
            # Run Data Quality Gate
            is_valid, msgs, clean_df, dlq = CreditRiskDataValidator.validate_ingestion_payload(raw_df)
            st.session_state.validation_messages = msgs

            for msg in msgs:
                if "✅" in msg:
                    st.success(msg)
                elif "⚠️" in msg:
                    st.warning(msg)
                else:
                    st.error(msg)

            if is_valid:
                st.dataframe(clean_df.head(3), use_container_width=True)

                if st.button("⚡ Ingest into DuckDB Memory Store", type="primary", key="ingest_btn", help="Registers clean records into the ephemeral DuckDB in-memory analytical warehouse."):
                    clean_df['session_id'] = st.session_state.session_id
                    clean_df['country_code'] = selected_country_code

                    st.session_state.primary_df = clean_df.copy()
                    st.session_state.duck_conn.register('temp_df', clean_df)
                    st.session_state.duck_conn.execute("CREATE OR REPLACE TABLE ml_features AS SELECT * FROM temp_df")

                    st.session_state.data_ingested = True
                    st.session_state.final_layered_df = clean_df.copy()
                    st.rerun()

    else:
        st.success(f"✅ Ingested {len(st.session_state.primary_df):,} records into active DuckDB memory store!")
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("🔄 Reset Portfolio & Upload New CSV", use_container_width=True, help="Wipes DuckDB in-memory tables and resets session state."):
                st.session_state.data_ingested = False
                st.session_state.layers_applied = False
                st.session_state.training_completed = False
                st.session_state.primary_df = None
                st.session_state.final_layered_df = None
                st.session_state.automl_results = None
                st.rerun()
        with col_btn2:
            baseline_csv = st.session_state.primary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Clean Ingested Baseline (.CSV)",
                data=baseline_csv,
                file_name="kba_ingested_baseline.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.write("---")


    # ==============================================================================
    # SECTION 2: ALTERNATIVE DATA LAYERING ENGINE
    # ==============================================================================
    if st.session_state.data_ingested:
        col_s2_title, col_s2_info = st.columns([4, 1])
        with col_s2_title:
            st.subheader("💡 2. Layer Alternative Data Streams")
        with col_s2_info:
            with st.popover("ℹ️ Vectorized Joins & Feature Stores"):
                st.markdown(r"""
                ### ⚡ DuckDB Vectorized In-Memory Joins
                
                * **High-Speed Columnar Execution:**  
                  DuckDB leverages SIMD vectorized columnar query execution, joining multi-gigabyte alternative datasets in sub-millisecond RAM latency.
                * **Dynamic Temporal & Regional Alignment:**  
                  Features are mapped via composite left joins matching `country_code` and temporal `year`:
                  ```sql
                  SELECT loan.*, macro.* EXCLUDE (country_code, year)
                  FROM ml_features loan
                  LEFT JOIN macro_warehouse macro
                    ON loan.country_code = macro.country_code
                    AND loan.year = macro.year;
                  ```
                * **Information Value (IV) Pre-Screening:**  
                  Signals with $IV \ge 0.10$ are retained for modeling; noisy features ($IV < 0.02$) are pruned to prevent overfitting.
                """)

        available_files = [f for f in os.listdir(ALTERNATIVE_DATA_DIR) if f.endswith(('.db', '.csv'))] if os.path.exists(ALTERNATIVE_DATA_DIR) else []
        
        col_layer_left, col_layer_right = st.columns([2, 1])

        with col_layer_left:
            selected_layers = []
            if available_files:
                st.write("Select local alternative data feeds to join in RAM:")
                for f in available_files:
                    if st.checkbox(f"🌿 Join `{f}` (Macro GCP & County Indicators)", value=(f == 'macro_layer.db'), key=f"chk_{f}", help=f"Executes in-memory vectorized left join with {f}"):
                        selected_layers.append(f)
            else:
                st.warning(f"No alternative data fixtures found in `{ALTERNATIVE_DATA_DIR}`.")

        with col_layer_right:
            st.caption("Layering Action:")
            if st.button("⚡ Execute Vectorized Join in DuckDB", type="primary", help="Merges selected alternative feeds into the analytical feature store in ephemeral RAM."):
                layered_df = apply_macro_layers(
                    st.session_state.duck_conn,
                    selected_layers,
                    ALTERNATIVE_DATA_DIR
                )
                st.session_state.final_layered_df = layered_df
                st.session_state.layers_applied = True
                st.success(f"Successfully merged features! Matrix shape: {layered_df.shape[0]:,} rows × {layered_df.shape[1]} columns.")
                st.rerun()

            if st.button("Continue Without Layering", help="Proceed directly with ingested baseline features."):
                st.session_state.final_layered_df = st.session_state.primary_df.copy()
                st.session_state.layers_applied = True
                st.rerun()

        if st.session_state.layers_applied:
            st.info(f"📊 Analytical Feature Store: **{st.session_state.final_layered_df.shape[1]} Total Features** Ready for AutoML Dispatcher")
            
            with st.expander("🔍 Preview Merged Feature Store & Data Science Exports"):
                st.dataframe(st.session_state.final_layered_df.head(3), use_container_width=True)
                col_exp_fs1, col_exp_fs2 = st.columns(2)
                with col_exp_fs1:
                    fs_csv = export_csv_bytes(st.session_state.final_layered_df)
                    st.download_button(
                        label="📥 Export Feature Store (.CSV)",
                        data=fs_csv,
                        file_name="kba_feature_store_snapshot.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col_exp_fs2:
                    try:
                        fs_parquet = export_parquet_bytes(st.session_state.final_layered_df)
                        st.download_button(
                            label="📦 Export Feature Store (.Parquet)",
                            data=fs_parquet,
                            file_name="kba_feature_store_snapshot.parquet",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                    except Exception:
                        st.caption("Parquet export engine (pyarrow) optional")

        st.write("---")

        # ==============================================================================
        # SECTION 2.5: EXPLORATORY DATA ANALYSIS (EDA) & DESCRIPTIVE STATISTICS
        # ==============================================================================
        with st.expander("📊 2.5 Exploratory Data Analysis (EDA) & Descriptive Statistics Hub", expanded=False):
            st.markdown("Automated portfolio profiling, collinearity heatmaps, and distribution histograms for risk analysts and data scientists.")
            
            active_eda_df = st.session_state.final_layered_df
            
            tab_stat, tab_dist, tab_corr, tab_box = st.tabs([
                "📋 Descriptive Statistics Table", 
                "📈 Distribution Histograms", 
                "🔥 Collinearity Heatmap", 
                "📦 Outliers & Quantile Boxplots"
            ])
            
            with tab_stat:
                col_eda_s1, col_eda_s2 = st.columns([4, 1])
                with col_eda_s2:
                    with st.popover("ℹ️ Statistical Metrics Guide"):
                        st.markdown("""
                        ### 📋 Portfolio Dispersion & Skew Metrics
                        
                        * **Mean vs. Median:** Large divergence signals high skewness in loan sizing or income distributions.
                        * **Standard Deviation (Std):** Measures dispersion around the mean.
                        * **Interquartile Range (IQR):** $Q3 - Q1$ (middle 50% of portfolio values), immune to extreme outliers.
                        * **Missing Rate %:** Flags data collection gaps in alternative channels.
                        """)
                
                stats_df = CreditRiskEDA.generate_descriptive_stats_df(active_eda_df)
                st.dataframe(stats_df, use_container_width=True)
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.download_button(
                        label="📥 Download Descriptive Statistics (.CSV)",
                        data=stats_df.to_csv(index=False).encode('utf-8'),
                        file_name="kba_descriptive_statistics.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col_d2:
                    buf_stat = io.BytesIO()
                    with pd.ExcelWriter(buf_stat, engine='openpyxl') as writer:
                        stats_df.to_excel(writer, index=False, sheet_name='Descriptive_Stats')
                    st.download_button(
                        label="📥 Download Descriptive Statistics (.Excel)",
                        data=buf_stat.getvalue(),
                        file_name="kba_descriptive_statistics.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
            with tab_dist:
                with st.popover("ℹ️ Class Imbalance in Distribution Plots"):
                    st.markdown("""
                    ### 📈 Distribution Histograms by Loan Outcome
                    
                    * **🟢 Performing vs. 🔴 Defaulted:**  
                      Histograms compare feature distributions between paying borrowers and defaulters.
                    * **Discriminatory Power:**  
                      Features with clear separation between green and red distributions (e.g. low M-Pesa velocity or high Fuliza utilization among defaulters) have strong predictive power.
                    """)
                dist_fig = CreditRiskEDA.generate_feature_distributions_fig(active_eda_df)
                if dist_fig:
                    st.plotly_chart(dist_fig, use_container_width=True)
                dist_png = CreditRiskEDA.generate_feature_distributions_bytes(active_eda_df)
                if dist_png:
                    st.download_button(
                        label="⬇️ Download Distribution Histograms (.PNG)",
                        data=dist_png,
                        file_name="kba_feature_distributions.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    
            with tab_corr:
                with st.popover("ℹ️ Understanding Pearson Collinearity"):
                    st.markdown("""
                    ### 🔥 Pearson Cross-Correlation & Multicollinearity
                    
                    * **Correlation Coefficient ($r$):**  
                      Ranges from $-1.0$ (perfect inverse correlation) to $+1.0$ (perfect direct correlation).
                    * **Multicollinearity Risk ($|r| > 0.80$):**  
                      Highly correlated features (e.g. GDP and County Output) provide redundant information and can inflate variance in linear and tree models. TreeSHAP handles non-linear cross-interactions effectively.
                    """)
                corr_fig = CreditRiskEDA.generate_correlation_heatmap_fig(active_eda_df)
                if corr_fig:
                    st.plotly_chart(corr_fig, use_container_width=True)
                
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    corr_png = CreditRiskEDA.generate_correlation_heatmap_bytes(active_eda_df)
                    if corr_png:
                        st.download_button(
                            label="⬇️ Download Correlation Heatmap (.PNG)",
                            data=corr_png,
                            file_name="kba_correlation_heatmap.png",
                            mime="image/png",
                            use_container_width=True
                        )
                with col_c2:
                    corr_matrix = CreditRiskEDA.generate_correlation_matrix(active_eda_df)
                    if not corr_matrix.empty:
                        st.download_button(
                            label="📥 Download Correlation Matrix (.CSV)",
                            data=corr_matrix.to_csv().encode('utf-8'),
                            file_name="kba_correlation_matrix.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
            with tab_box:
                with st.popover("ℹ️ Tukey Boxplots & Outlier Detection"):
                    st.markdown("""
                    ### 📦 Outlier Bounds & Quantile Spread
                    
                    * **Box Dimensions:** Represents the Interquartile Range ($IQR = Q3 - Q1$, middle 50%).
                    * **Whiskers:** Extend to $1.5 \\times IQR$ from the upper/lower quartiles.
                    * **Outliers (Dots):** Loan amounts or tenors exceeding the whiskers indicate extreme borrowing behavior requiring strict credit limit caps.
                    """)
                box_fig = CreditRiskEDA.generate_boxplots_by_target_fig(active_eda_df)
                if box_fig:
                    st.plotly_chart(box_fig, use_container_width=True)

        st.write("---")


    # ==============================================================================
    # SECTION 3: TABFM & PYCARET AUTOML ENGINE & DISPATCHER
    # ==============================================================================
    if st.session_state.layers_applied:
        st.subheader("🤖 3. TabFM & PyCaret AutoML Engine")

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

        with col_cfg1:
            st.markdown("**Task & Optimization Metric**")
            st.caption("Target Detected: `default_flag` (Binary Default Classification)")
            optimize_metric = st.selectbox(
                "Optimization Metric:", 
                ["PR-AUC", "ROC-AUC", "F1", "Accuracy"], 
                index=0,
                help="PR-AUC is prioritized for imbalanced credit default data to minimize costly false negative lending approvals."
            )
            with st.popover("ℹ️ Why PR-AUC Optimization?"):
                st.markdown("""
                ### 🎯 PR-AUC vs. ROC-AUC in Credit Risk
                
                **The Class Imbalance Challenge:**  
                In standard credit portfolios, defaults are relatively rare (**3% to 15%** default rate).
                
                * **Why Not Standard Accuracy?**  
                  A naive model that simply predicts *every loan will perform (0)* achieves 90%+ accuracy, but misses **100% of defaults**, causing massive portfolio losses.
                * **Why ROC-AUC Can Be Misleading:**  
                  ROC-AUC measures True Positive Rate against False Positive Rate. Because performing loans (the negative class) vastly outnumber defaulters, the False Positive Rate stays small, making ROC-AUC look overly optimistic.
                * **Why PR-AUC (Precision-Recall Area Under Curve)?**  
                  PR-AUC focuses directly on the **minority class ($Default = 1$)**. It strictly penalizes approving risky borrowers (false positives) and failing to detect defaults (false negatives):
                  $$\\text{Precision} = \\frac{TP}{TP + FP}, \\quad \\text{Recall} = \\frac{TP}{TP + FN}$$
                
                **Takeaway:** Optimizing for PR-AUC directly preserves lending capital and lowers Non-Performing Loans (NPLs).
                """)

        with col_cfg2:
            st.markdown("**Hyperparameter Tuning & Ensembling**")
            tune_toggle = st.checkbox(
                "Enable Automated Optuna Hyperparameter Tuning", 
                value=True,
                help="Uses Tree-structured Parzen Estimator (TPE) Bayesian search across tree depths, learning rates, and L1/L2 penalties."
            )
            ensemble_toggle = st.checkbox(
                "Construct Soft-Voting GBDT & TabFM Ensemble", 
                value=True,
                help="Averages predicted probability distributions across heterogeneous model families to reduce variance."
            )
            with st.popover("ℹ️ How Optuna & Ensembles Work"):
                st.markdown("""
                ### 🔬 Optuna Tuning & Soft-Voting Ensembles
                
                * **Optuna (Bayesian TPE Optimization):**  
                  Unlike brute-force Grid Search, Optuna constructs a probabilistic surrogate model of the objective function. It intelligently focuses computational trials on the most promising hyperparameter regions (learning rate, tree depth, subsample ratio, and L2 regularization penalties).
                * **Soft-Voting Ensemble ($VotingClassifier$):**  
                  Combines the calibrated predicted default probabilities of the top champion algorithms:
                  $$P(\\text{Default} = 1 \\mid x) = \\sum_{m=1}^{M} w_m \\cdot P_m(\\text{Default} = 1 \\mid x)$$
                * **Why Blend Models?**  
                  GBDTs (LightGBM/XGBoost/CatBoost) capture orthogonal split boundaries, while TabFM captures continuous non-linear representations. Combining them reduces model variance and dampens idiosyncratic errors.
                """)

        with col_cfg3:
            st.markdown("**Engine Execution**")
            if st.button("🚀 Run TabFM & AutoML Pipeline", type="primary", use_container_width=True, help="Trains LightGBM, XGBoost, CatBoost, TabFM, and candidate models via 5-Fold Stratified Cross-Validation."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(pct: int, msg: str):
                    progress_bar.progress(pct)
                    status_text.markdown(f"**Status:** {msg}")

                with st.spinner("Benchmarking TabFM, LightGBM, XGBoost, CatBoost & candidate models..."):
                    results = run_automl_pipeline(
                        st.session_state.final_layered_df,
                        optimize_metric=optimize_metric,
                        tune_hyperparams=tune_toggle,
                        create_ensemble=ensemble_toggle,
                        progress_callback=update_progress
                    )

                st.session_state.automl_results = results
                st.session_state.training_completed = True
                status_text.success("🎉 AutoML Pipeline Completed Successfully!")
                st.rerun()

            with st.popover("ℹ️ What is TabFM (Foundation Model)?"):
                st.markdown("""
                ### 🤖 TabFM (Tabular Foundation Model)
                
                * **Deep Tabular Embeddings:**  
                  Maps discrete categorical and continuous alternative signals into continuous dense vector representations.
                * **Residual Feature Interactors:**  
                  Uses multi-layer residual blocks `(128 -> 64 -> 32)` with LayerNorm and Dropout to learn non-linear cross-feature relationships between macro indicators (e.g. county GCP) and mobile money velocity.
                * **Sigmoid Probability Calibration:**  
                  Outputs well-calibrated default probabilities ready for regulatory scoring.
                """)

        st.write("---")


    # ==============================================================================
    # SECTION 4: LIVE DASHBOARD & ONSET DEFAULT SCREENING
    # ==============================================================================
    if st.session_state.training_completed and st.session_state.automl_results is not None:
        results = st.session_state.automl_results
        df = st.session_state.final_layered_df
        probs = results.get("predicted_probs", np.zeros(len(df)))
        explainer: Optional[CreditRiskExplainer] = results.get("explainer", None)
        leaderboard_df = results.get("leaderboard", pd.DataFrame())

        champion_name = results.get("champion_name", "AutoML Champion Model")
        engine_name = results.get("engine_name", "AutoML Engine")

        col_s4_title, col_s4_info = st.columns([4, 1])
        with col_s4_title:
            st.markdown("## 📈 4. Onset Default Risk Screening & Live Analytics")
        with col_s4_info:
            with st.popover("ℹ️ Understanding PD & Risk Tiers"):
                st.markdown("""
                ### 📊 Credit Risk Key Performance Indicators (KPIs)
                
                * **Predicted Mean PD (Probability of Default):**  
                  The portfolio-wide expected default rate over the loan tenor under current alternative data conditions.
                * **🔴 High Risk (PD $\\ge$ 60%):**  
                  Immediate adverse action / credit rejection or mandatory guarantor requirement.
                * **🟡 Medium Risk (30% $\\le$ PD < 60%):**  
                  Eligible for risk-adjusted dynamic credit limit scaling.
                * **🟢 Low Risk (PD < 30%):**  
                  Prime informal sector borrower eligible for accelerated disbursement.
                """)

        # --- CHAMPION MODEL ANNOUNCEMENT BANNER ---
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px; padding: 16px 22px; margin-bottom: 20px; border-left: 6px solid #3b82f6; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1.2px; color: #94a3b8; font-weight: 700;">🏆 Winning Champion Model Selected by AutoML</div>
            <div style="font-size: 22px; font-weight: 800; color: #60a5fa; margin-top: 4px;">{champion_name}</div>
            <div style="font-size: 13px; color: #cbd5e1; margin-top: 4px;">
                <b>Active Architecture:</b> {champion_name} &nbsp;|&nbsp; <b>Execution Engine:</b> {engine_name} &nbsp;|&nbsp; <b>Explainability:</b> TreeSHAP Feature Attributions
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- 4.1 EXECUTIVE KPI ROW ---
        total_records = len(df)
        hist_default_rate = (df['default_flag'].sum() / total_records * 100.0) if 'default_flag' in df and total_records > 0 else 0.0
        mean_pred_pd = np.mean(probs) * 100.0
        high_risk_count = int(np.sum(probs >= 0.60))

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

        with kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Evaluated Portfolio</div>
                <div class="kpi-value">{total_records:,} Records</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Historical Default Rate</div>
                <div class="kpi-value" style="color: #dc2626;">{hist_default_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Predicted Mean PD</div>
                <div class="kpi-value" style="color: #d97706;">{mean_pred_pd:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">High Risk Borrowers</div>
                <div class="kpi-value" style="color: #dc2626;">{high_risk_count:,}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Champion Algorithm</div>
                <div class="kpi-value" style="font-size: 15px; color: #2563eb; line-height: 1.3; font-weight: 700; word-break: break-word;">{champion_name}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # --- 4.2 THE ONSET DEFAULT DECISION TABLE ---
        col_dt_title, col_dt_info = st.columns([4, 1])
        with col_dt_title:
            st.subheader("📋 Onset Default Risk Decision Table")
        with col_dt_info:
            with st.popover("ℹ️ Adverse Action Reason Codes"):
                st.markdown("""
                ### 📋 Adverse Action Decomposition & CBK Compliance
                
                Under the **Central Bank of Kenya (CBK) Fair Lending Regulations**, automated credit models cannot act as black boxes:
                * **Reason Code Attribution:** Positive SHAP values (factors increasing default odds) are normalized into exact percentage shares:
                  $$\\text{Impact Share}_j = \\left( \\frac{\\phi_j(x)}{\\sum_{i \\in \\text{Risk Factors}} \\phi_i(x)} \\right) \\times 100\\%$$
                * **Plain-English Explanations:** Credit officers receive exact monetary and behavioral drivers (e.g. *"Fuliza Overdraft Utilization drives 42.1% of default risk"*).
                """)

        # Filters
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            risk_tier_filter = st.multiselect(
                "Filter Risk Tier:",
                ["🔴 High Risk (PD ≥ 60%)", "🟡 Medium Risk (30% ≤ PD < 60%)", "🟢 Low Risk (PD < 30%)"],
                default=["🔴 High Risk (PD ≥ 60%)", "🟡 Medium Risk (30% ≤ PD < 60%)", "🟢 Low Risk (PD < 30%)"]
            )

        # Build Decision Table Data
        table_rows = []
        for idx in range(len(df)):
            pd_val = probs[idx] if idx < len(probs) else 0.0
            pd_pct = pd_val * 100.0
            
            if pd_pct >= 60.0:
                tier = "🔴 High Risk (PD ≥ 60%)"
            elif pd_pct >= 30.0:
                tier = "🟡 Medium Risk (30% ≤ PD < 60%)"
            else:
                tier = "🟢 Low Risk (PD < 30%)"

            if tier not in risk_tier_filter:
                continue

            borrower_id = df['borrower_id'].iloc[idx] if 'borrower_id' in df else f"CUST-{idx+1000}"
            amount = df['amount'].iloc[idx] if 'amount' in df else 0.0
            tenure = df['tenure_days'].iloc[idx] if 'tenure_days' in df else 30

            # Get Adverse Action Reasons
            if explainer is not None:
                reasons = explainer.get_borrower_adverse_reasons(idx, top_k=3)
                reason_str = "<br>".join([f"• {r['feature_name']}: {r['reason_text']} ({r['impact_percentage']}%)" for r in reasons])
            else:
                reason_str = "• Historical default history and debt ratio elevate risk."

            table_rows.append({
                "Index": idx,
                "Borrower ID": borrower_id,
                "Principal (KES)": f"{amount:,.0f}",
                "Tenor": f"{tenure} Days",
                "Predicted PD": f"{pd_pct:.1f}%",
                "Risk Tier": tier,
                "Primary Adverse Action Reasons (Decomposed Weight)": reason_str
            })

        decision_df = pd.DataFrame(table_rows)

        if not decision_df.empty:
            st.write(f"Displaying **{len(decision_df):,}** borrowers matching selected risk tiers:")
            
            # Select individual borrower for Data Card Drawer
            selected_borrower_id = st.selectbox(
                "🔍 Select Borrower ID to inspect Borrower Risk Data Card:",
                options=decision_df["Borrower ID"].tolist(),
                index=0
            )
            
            # Render Decision Table
            display_table = decision_df.drop(columns=["Index"])
            st.dataframe(display_table, use_container_width=True, height=280)

            # Get selected borrower record index
            sel_idx = decision_df[decision_df["Borrower ID"] == selected_borrower_id]["Index"].values[0]

            # --- 4.3 BORROWER RISK DATA CARD DRAWER (ZERO FILLER GUARANTEE) ---
            st.write("")
            st.markdown(f"### 🎴 Borrower Risk Data Card: `{selected_borrower_id}`")

            card_row = df.iloc[sel_idx]
            card_pd = probs[sel_idx] * 100.0
            card_amt = card_row.get("amount", 0.0)
            card_tenure = card_row.get("tenure_days", 30)

            # Zero-Filler field extraction helper: leaves blank / '—' if not in portfolio
            def get_card_field(row, col_name, fmt_fn=None, fallback="—"):
                if col_name in row and pd.notna(row[col_name]):
                    val_str = str(row[col_name]).strip()
                    if val_str and val_str.lower() not in ["nan", "none", "null", ""]:
                        try:
                            return fmt_fn(row[col_name]) if fmt_fn else val_str
                        except Exception:
                            return val_str
                return fallback

            # Inspect actual county in portfolio without hardcoded filler fallback
            county_val = None
            for c in ["county", "county_name", "county_code", "borrower_county", "County", "County_Code"]:
                if c in card_row and pd.notna(card_row[c]):
                    s = str(card_row[c]).strip()
                    if s and s.lower() not in ["nan", "none", "null", ""]:
                        county_val = s
                        break

            county_line = f" &nbsp;|&nbsp; <b>County:</b> {county_val}" if county_val else " &nbsp;|&nbsp; <b>County:</b> <i>—</i>"

            rec_limit = max(10000, int(card_amt * (1.0 - (card_pd / 100.0)))) if card_amt > 0 else 0

            # Extract actual values or return '—' (no synthetic filler numbers)
            lifetime_amt = get_card_field(card_row, 'feat_loan_amount_borrowed_sum_lifetime', lambda v: f"KES {float(v):,.0f}")
            lifetime_def = get_card_field(card_row, 'feat_loan_prior_default_count_lifetime', lambda v: f"{int(float(v))} times")
            crb_score = get_card_field(card_row, 'feat_bureau_crb_score', lambda v: f"{float(v):.0f}")

            mpesa_vel = get_card_field(card_row, 'feat_mpesa_velocity_30d_over_90d', lambda v: f"{float(v):.2f}")
            fuliza_util = get_card_field(card_row, 'feat_fuliza_util_pct_30d', lambda v: f"{float(v)*100:.1f}%" if float(v) <= 1.0 else f"{float(v):.1f}%")
            ndvi_anom = get_card_field(card_row, 'feat_gis_ndvi_anomaly_30d', lambda v: f"{float(v):.3f}")

            col_card1, col_card2 = st.columns([1, 1])

            with col_card1:
                st.markdown(f"""
                <div class="borrower-card">
                    <div class="card-header-title">👤 BORROWER PROFILE & DECISION</div>
                    <p><b>Borrower ID:</b> {selected_borrower_id} &nbsp;|&nbsp; <b>Jurisdiction:</b> Kenya{county_line}</p>
                    <p><b>Requested Loan Principal:</b> KES {card_amt:,.0f} &nbsp;|&nbsp; <b>Tenor:</b> {card_tenure} Days</p>
                    <div style="margin-top: 12px; margin-bottom: 12px;">
                        <b>Predicted Default Probability (PD):</b> <span style="font-size: 20px; font-weight: 800; color: #dc2626;">{card_pd:.1f}%</span>
                        <br><b>Recommended Max Credit Limit:</b> <span style="font-size: 16px; font-weight: 700; color: #16a34a;">KES {rec_limit:,.0f}</span>
                    </div>
                    <div class="card-section-label">📋 TRADITIONAL PANEL LOAN HISTORY</div>
                    <ul>
                        <li><b>Lifetime Total Borrowed:</b> {lifetime_amt}</li>
                        <li><b>Lifetime Historical Defaults:</b> {lifetime_def}</li>
                        <li><b>Bureau CRB Score:</b> {crb_score}</li>
                    </ul>
                    <div class="card-section-label">🌿 ALTERNATIVE DATA SIGNALS</div>
                    <ul>
                        <li><b>M-Pesa 30d/90d Inflow Velocity:</b> {mpesa_vel}</li>
                        <li><b>Fuliza Overdraft Utilization:</b> {fuliza_util}</li>
                        <li><b>Regional Crop NDVI Anomaly:</b> {ndvi_anom}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col_card2:
                col_w_title, col_w_pop = st.columns([3, 1])
                with col_w_title:
                    st.markdown("**🔍 TreeSHAP Waterfall Explanation Plot**")
                with col_w_pop:
                    with st.popover("ℹ️ Reading Waterfall"):
                        st.markdown("""
                        ### 🔍 TreeSHAP Waterfall Plot Guide
                        
                        * **Baseline Expected Value ($E[f(x)] = \\phi_0$):** Average log-odds default rate across the entire portfolio.
                        * **Red Arrows (+ SHAP):** Factors increasing default risk above the baseline (e.g. high Fuliza utilization or historical arrears).
                        * **Blue Arrows (- SHAP):** Factors reducing default risk (e.g. high M-Pesa inflow velocity or low debt burden).
                        * **Final Prediction ($f(x)$):** Sum of baseline + individual feature contributions converted to final predicted PD %.
                        """)

                if explainer is not None:
                    img_bytes = explainer.generate_waterfall_plot_bytes(sel_idx)
                    if img_bytes:
                        st.image(img_bytes, use_container_width=True)
                    else:
                        st.info("TreeSHAP waterfall plot rendering unavailable for this model architecture.")
                else:
                    st.info("Fit TreeSHAP explainer to view waterfall decomposition.")

        else:
            st.warning("No records match the selected Risk Tier filter.")

        st.write("---")

        # --- 4.4 MODEL LEADERBOARD & PORTFOLIO SHAP FEATURE IMPORTANCE ---
        with st.expander("📊 View Comparative Model Leaderboard & Portfolio Feature Importance Graphs", expanded=True):
            col_lead1, col_lead2 = st.columns([1, 1], gap="medium")

            with col_lead1:
                col_l1_t, col_l1_i = st.columns([3, 1])
                with col_l1_t:
                    st.markdown("#### 🏆 Comparative Model Leaderboard")
                with col_l1_i:
                    with st.popover("ℹ️ Model Metrics"):
                        st.markdown("""
                        ### 🏆 Leaderboard Evaluation Metrics
                        
                        * **PR-AUC (Precision-Recall AUC):** Primary ranking metric for imbalanced default detection.
                        * **ROC-AUC & Gini Index:** $\\text{Gini} = 2 \\times \\text{ROC-AUC} - 1$. Measures general discrimination power.
                        * **LogLoss:** Measures cross-entropy calibration accuracy of predicted probabilities.
                        * **F1 Score:** Harmonic mean of precision and recall at default decision threshold.
                        """)

                if not leaderboard_df.empty:
                    st.dataframe(leaderboard_df, use_container_width=True)
                else:
                    st.write("Leaderboard data unavailable.")

            with col_lead2:
                col_l2_t, col_l2_i = st.columns([3, 1])
                with col_l2_t:
                    st.markdown("#### 🌟 Portfolio-Wide Feature Importance (SHAP)")
                with col_l2_i:
                    with st.popover("ℹ️ Reading Beeswarm Plots"):
                        st.markdown("""
                        ### 🐝 Interpreting SHAP Beeswarm & Importance
                        
                        * **Feature Importance Bar Graph:** Ranks features by mean absolute SHAP value (overall global predictive impact).
                        * **Beeswarm Plot:**  
                          - Each dot represents a single borrower.
                          - **Color:** Red = High feature value; Blue = Low feature value.
                          - **Position (X-axis):** Positive value pushes risk up; Negative value pulls risk down.
                        """)

                if explainer is not None:
                    tab_bar, tab_bee, tab_tbl = st.tabs(["📊 Feature Bar Graph", "🐝 Beeswarm Plot", "📋 Importance Table"])
                    
                    with tab_bar:
                        plotly_fig = explainer.generate_plotly_feature_bar_fig(top_n=12)
                        if plotly_fig:
                            st.plotly_chart(plotly_fig, use_container_width=True)
                        else:
                            bar_bytes = explainer.generate_bar_plot_bytes()
                            if bar_bytes:
                                st.image(bar_bytes, use_container_width=True)
                            else:
                                st.info("Feature importance bar plot unavailable for this model architecture.")
                        
                        bar_bytes = explainer.generate_bar_plot_bytes()
                        if bar_bytes:
                            st.download_button(
                                label="⬇️ Download Portfolio SHAP Bar Graph (.PNG)",
                                data=bar_bytes,
                                file_name="portfolio_shap_feature_importance.png",
                                mime="image/png",
                                use_container_width=True
                            )

                    with tab_bee:
                        beeswarm_bytes = explainer.generate_beeswarm_plot_bytes()
                        if beeswarm_bytes:
                            st.image(beeswarm_bytes, use_container_width=True)
                            st.caption("Dots represent individual borrowers. Color denotes feature value (Red = High, Blue = Low). Position on X-axis denotes risk impact.")

                    with tab_tbl:
                        imp_df = explainer.get_global_feature_importance_df(top_n=25)
                        st.dataframe(imp_df, use_container_width=True)
                        st.download_button(
                            label="⬇️ Export Feature Importance Table (.CSV)",
                            data=imp_df.to_csv(index=False).encode('utf-8'),
                            file_name="kba_global_feature_importance.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.info("Fit TreeSHAP explainer to view portfolio-wide feature attributions.")

        # --- 4.5 EXPORT SCORED PORTFOLIO & MLOPS HUB ---
        st.write("")
        st.markdown("### 📥 Portfolio Decisions & Artifacts Export")

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            scored_df = df.copy()
            scored_df["predicted_pd_pct"] = np.round(probs * 100.0, 2)
            scored_df["risk_tier"] = np.where(
                probs >= 0.60, "High Risk",
                np.where(probs >= 0.30, "Medium Risk", "Low Risk")
            )
            csv_bytes = scored_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Scored Portfolio as CSV (Plug & Play)",
                data=csv_bytes,
                file_name="kba_scored_portfolio.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

        with col_exp2:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                scored_df.to_excel(writer, index=False, sheet_name='Scored_Portfolio')
            st.download_button(
                label="📥 Export Scored Portfolio as Excel (.xlsx)",
                data=buf.getvalue(),
                file_name="kba_scored_portfolio.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        # Data Scientist & ML Engineer Artifacts Hub
        with st.expander("🛠️ Data Scientist & MLOps Artifacts Export Hub (Pipelines, Scripts & Feature Stores)"):
            st.markdown("""
            **Modular MLOps Artifacts:**  
            Export production-ready model pipelines, scoring recipes, and feature snapshots for offline fine-tuning, Optuna searches, or CI/CD model serving.
            """)

            engine = results.get("engine", None)
            champion_model = results.get("champion_model", None)

            col_ds1, col_ds2, col_ds3 = st.columns(3)

            with col_ds1:
                st.markdown("**1. Trained Model Pipeline**")
                st.caption("Fitted preprocessing + classifier artifact (.pkl)")
                pkl_bytes = b""
                if engine is not None and hasattr(engine, "export_pipeline_bytes"):
                    try:
                        pkl_bytes = engine.export_pipeline_bytes()
                    except Exception:
                        try:
                            import pickle
                            pkl_bytes = pickle.dumps(champion_model)
                        except Exception:
                            try:
                                import cloudpickle
                                pkl_bytes = cloudpickle.dumps(champion_model)
                            except Exception:
                                pkl_bytes = b""
                elif champion_model is not None:
                    try:
                        import pickle
                        pkl_bytes = pickle.dumps(champion_model)
                    except Exception:
                        try:
                            import cloudpickle
                            pkl_bytes = cloudpickle.dumps(champion_model)
                        except Exception:
                            pkl_bytes = b""

                if pkl_bytes:
                    st.download_button(
                        label="💾 Download Champion Pipeline (.pkl)",
                        data=pkl_bytes,
                        file_name="champion_pipeline.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                else:
                    st.info("Pipeline serialization will be ready once model fitting is complete.")

            with col_ds2:
                st.markdown("**2. Python Scoring Script**")
                st.caption("Standalone offline inference recipe (.py)")
                if engine is not None and hasattr(engine, "generate_inference_script"):
                    py_script = engine.generate_inference_script()
                else:
                    py_script = "# Standalone inference script\nimport pickle, pandas as pd\n"

                st.download_button(
                    label="📄 Download Inference Code (.py)",
                    data=py_script.encode('utf-8'),
                    file_name="infer_credit_model.py",
                    mime="text/x-python",
                    use_container_width=True
                )

            with col_ds3:
                st.markdown("**3. Benchmark Leaderboard**")
                st.caption("Detailed multi-model CV metrics (.csv)")
                leaderboard_csv = leaderboard_df.to_csv(index=False).encode('utf-8') if not leaderboard_df.empty else b""
                st.download_button(
                    label="📊 Download Leaderboard (.csv)",
                    data=leaderboard_csv,
                    file_name="automl_leaderboard_benchmark.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            st.markdown("**Python Scoring Recipe (Copy & Paste):**")
            st.code("""
import pickle
import pandas as pd

# 1. Load trained Champion Pipeline
with open("champion_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# 2. Score incoming borrower records
new_loans = pd.read_csv("new_borrowers.csv")
pd_scores = pipeline.predict_proba(new_loans)[:, 1]
new_loans["predicted_pd"] = pd_scores
print(new_loans[["borrower_id", "predicted_pd"]].head())
            """, language="python")


    # ==============================================================================
    # SECTION 5: AI ANALYST (NATURAL-LANGUAGE QUERY -> ANSWER + AUTO DASHBOARD)
    # ==============================================================================
    if st.session_state.data_ingested:
        st.write("---")
        col_ai_t, col_ai_i = st.columns([4, 1])
        with col_ai_t:
            st.subheader("🤖 5. AI Analyst — Ask Your Portfolio Anything")
        with col_ai_i:
            with st.popover("ℹ️ How AI Analyst Operates"):
                st.markdown("""
                ### 🧠 Private In-Memory AI Reasoning
                
                * **Zero External Data Leakage:**  
                  The AI Analyst synthesizes queries against DuckDB in local RAM. Customer data is never transmitted to third-party proprietary LLM APIs.
                * **Deterministic Calculation:**  
                  Numbers and charts are computed strictly through in-memory SQL aggregates, guaranteeing 100% mathematical accuracy without LLM hallucinations.
                * **Local Ollama Integration:**  
                  If Ollama is running locally, it translates unstructured risk queries into precise analytics pipelines.
                """)

        try:
            from modules.ai_assistant import NLQueryEngine, OllamaClient
            _HAS_AI = True
        except Exception:
            _HAS_AI = False

        if not _HAS_AI:
            st.warning("AI Analyst module unavailable (`modules/ai_assistant.py`).")
        else:
            with st.expander("⚙️ Ollama Settings & Model Selection", expanded=False):
                cfg_col1, cfg_col2 = st.columns([3, 1])
                with cfg_col1:
                    ollama_base = st.text_input(
                        "Ollama server URL",
                        value=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                        key="ollama_base",
                    )
                with cfg_col2:
                    st.write("")
                    st.write("")
                    st.button("🔄 Test / Refresh", key="recheck_ollama", use_container_width=True)

            _oc = OllamaClient(base_url=ollama_base.strip())
            _available_models = _oc.list_models(timeout=2)

            if _available_models:
                default_idx = 0
                for idx, m in enumerate(_available_models):
                    if "llama3" in m.lower():
                        default_idx = idx
                        break
                
                selected_model = _available_models[0]
                if len(_available_models) > 1:
                    selected_model = st.selectbox(
                        "🤖 Detected Ollama Models (Select active model for reasoning):",
                        options=_available_models,
                        index=default_idx,
                        key="active_ollama_model_select"
                    )
                _oc.model = selected_model
                ai_llm = _oc
                st.caption(f"🟢 **Ollama connected** — model `{selected_model}`. Free-form questions are interpreted locally via local LLM.")
            else:
                ai_llm = None
                st.caption(f"🟡 **Ollama not detected** at `{ollama_base.strip()}` — using the built-in offline engine. Start `ollama serve` and pull a model to enable free-form LLM answers.")

            ai_df = st.session_state.final_layered_df
            ai_probs = None
            ai_leaderboard = pd.DataFrame()
            ai_importance = None
            ai_champion = None
            ai_explainer = None

            if st.session_state.training_completed and st.session_state.automl_results is not None:
                _res = st.session_state.automl_results
                ai_probs = _res.get("predicted_probs")
                ai_leaderboard = _res.get("leaderboard", pd.DataFrame())
                ai_champion = _res.get("champion_name")
                ai_explainer = _res.get("explainer")
                if ai_explainer is not None:
                    try:
                        ai_importance = ai_explainer.get_global_feature_importance_df(top_n=25)
                    except Exception:
                        ai_importance = None

            ai_context = {
                "df": ai_df,
                "probs": ai_probs,
                "leaderboard": ai_leaderboard,
                "feature_importance": ai_importance,
                "explainer": ai_explainer,
                "champion_name": ai_champion,
            }

            st.caption("Type a question in plain English — I'll answer it and build a dashboard. Charts and numbers are always computed accurately in-memory.")

            ai_query = st.text_input(
                "💬 Your question:",
                placeholder="e.g. 'default rate by county', 'top 10 riskiest borrowers', 'what drives risk?', 'distribution of loan amount'",
                key="ai_query",
            )

            if ai_query and ai_query.strip():
                ai_engine = NLQueryEngine(llm=ai_llm)
                with st.spinner("🧠 Analyzing your portfolio..."):
                    ai_result = ai_engine.run(ai_query.strip(), ai_context)

                st.markdown(ai_result.get("answer", ""))

                # KPI cards
                ai_kpis = ai_result.get("kpis", []) or []
                if ai_kpis:
                    kpi_cols = st.columns(min(len(ai_kpis), 4))
                    for i, k in enumerate(ai_kpis):
                        with kpi_cols[i % len(kpi_cols)]:
                            st.markdown(f"""
                            <div class="kpi-card">
                                <div class="kpi-title">{k.get('label', '')}</div>
                                <div class="kpi-value">{k.get('value', '')}</div>
                            </div>
                            """, unsafe_allow_html=True)

                # Charts
                for fig in ai_result.get("figures", []) or []:
                    st.plotly_chart(fig, use_container_width=True)

                # Tables
                for tbl in ai_result.get("tables", []) or []:
                    st.dataframe(tbl, use_container_width=True)


# ##############################################################################
# TAB 2: DATA SOURCES & METHODOLOGY REGISTRY
# ##############################################################################
with tab_sources:
    col_reg_t, col_reg_i = st.columns([4, 1])
    with col_reg_t:
        st.markdown("## 📚 Alternative Data Sources & Indicator Registry")
    with col_reg_i:
        with st.popover("ℹ️ Econometric Transmission Channels"):
            st.markdown("""
            ### 📡 How Alternative Feeds Transmit into Default Risk
            
            * **Macro GCP & Output:** Regional GDP contractions directly reduce commercial footfall and revenue for unbanked micro-merchants.
            * **Food & Fuel Inflation (KAMIS & EPRA):** Volatility in food and transport expenses diminishes household disposable income, triggering default cascades.
            * **High-Frequency Behavioral Distress:** Spikes in Google Searches for debt renegotiation or auctioneers serve as a 30-day leading indicator of default onset.
            * **M-Pesa Cash Flow Velocity:** Declining 30d/90d inflow ratios signal working capital contraction before formal CRB records reflect arrears.
            """)

    st.markdown("""
    This registry establishes the **authoritative collection methods, reference sources, and update cadences** 
    for all alternative data streams indexed across Kenya's 47 counties. 
    Inspired by open data initiatives such as the [Kenya Agri Atlas](https://saficagriatlas.strathmore.edu/sources), 
    this framework leverages zero-marginal-cost public APIs, statutory gazette trackers, and automated data scraping.
    """)

    # Top KPI summary cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Monitored Indicators</div>
            <div class="kpi-value">9 Streams</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Geographic Scope</div>
            <div class="kpi-value">47 Counties</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">API Marginal Cost</div>
            <div class="kpi-value" style="color: #16a34a;">0.00 KES (Free/Open)</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Ingestion Status</div>
            <div class="kpi-value" style="color: #c2410c;">Pending Live Feeds</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("📋 Alternative Data Indicator Catalog")

    # Filters
    col_search, col_cat, col_stat = st.columns([2, 1, 1])
    with col_search:
        search_term = st.text_input("🔍 Search Variable, Source, or Method:", placeholder="e.g. GDP, Food, EPRA, M-Pesa, OSM, Poverty...", help="Filter data variables by name, endpoint, authority, or risk rationale.")
    with col_cat:
        all_categories = ["All Domains"] + sorted(list(set(d["category"] for d in DATA_SOURCES_CATALOG)))
        selected_category = st.selectbox("Filter Domain:", all_categories, help="Filter indicators by categorical domain.")
    with col_stat:
        all_statuses = ["All Statuses"] + sorted(list(set(d["status"] for d in DATA_SOURCES_CATALOG)))
        selected_status = st.selectbox("Filter Status:", all_statuses, help="Filter indicators by live ingestion readiness status.")

    # Filter records
    filtered_data = []
    for item in DATA_SOURCES_CATALOG:
        if selected_category != "All Domains" and item["category"] != selected_category:
            continue
        if selected_status != "All Statuses" and item["status"] != selected_status:
            continue
        if search_term:
            q = search_term.lower()
            match = (
                q in item["variable"].lower()
                or q in item["collection_method"].lower()
                or q in item["reference"].lower()
                or q in item["category"].lower()
                or q in item["actuarial_rationale"].lower()
            )
            if not match:
                continue
        filtered_data.append(item)

    # Build Display Table
    table_rows = []
    for item in filtered_data:
        table_rows.append({
            "Variable": item["variable"],
            "Domain Category": item["category"],
            "Collection Method": item["collection_method"],
            "Reference / Authority": item["reference"],
            "Reference Link": item["url"],
            "Update Cadence": item["last_updated"],
            "Status": f"🟡 {item['status']}" if item['status'] == "Pending" else f"🟢 {item['status']}"
        })

    df_sources = pd.DataFrame(table_rows)

    if not df_sources.empty:
        st.dataframe(
            df_sources,
            column_config={
                "Reference Link": st.column_config.LinkColumn(
                    "Source Link",
                    display_text="Open Portal ↗"
                ),
                "Variable": st.column_config.TextColumn(
                    "Variable Name",
                    width="medium"
                ),
                "Collection Method": st.column_config.TextColumn(
                    "Collection Method & Endpoint",
                    width="large"
                ),
            },
            use_container_width=True,
            height=380
        )

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            sources_csv = df_sources.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Sources Catalog (.CSV)",
                data=sources_csv,
                file_name="kba_alternative_data_sources.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            buf_src = io.BytesIO()
            with pd.ExcelWriter(buf_src, engine='openpyxl') as writer:
                df_sources.to_excel(writer, index=False, sheet_name='Data_Sources')
            st.download_button(
                label="📥 Download Sources Catalog (.Excel)",
                data=buf_src.getvalue(),
                file_name="kba_alternative_data_sources.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    else:
        st.info("No data sources match the selected search query or filters.")

    st.write("---")

    # Detailed Cards View
    with st.expander("🔍 Detailed Variable Specifications & Actuarial Risk Rationales", expanded=False):
        for item in filtered_data:
            badge_cls = "badge-pending" if item["status"] == "Pending" else "badge-active"
            st.markdown(f"""
            <div class="source-card">
                <div class="source-card-title">
                    <span>{item['variable']}</span>
                    <div>
                        <span class="source-category-tag">{item['category']}</span>
                        <span class="{badge_cls}">● {item['status']}</span>
                    </div>
                </div>
                <div class="source-meta-row">
                    <b>Collection Method:</b> {item['collection_method']} &nbsp;|&nbsp; <b>Update Cadence:</b> {item['last_updated']}
                </div>
                <div class="source-meta-row">
                    <b>Data Authority & Reference:</b> <a href="{item['url']}" target="_blank">{item['reference']}</a>
                </div>
                <div class="source-desc">
                    <b>Actuarial & Credit Risk Rationale:</b> {item['actuarial_rationale']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("---")

    # Interactive Methodology Viewer
    with st.expander("📖 View Master Methodology & Architectural Guide (Full Document)", expanded=False):
        try:
            methodology_path = os.path.join(BASE_DIR, "METHODOLOGY.md")
            if os.path.exists(methodology_path):
                with open(methodology_path, "r", encoding="utf-8") as f:
                    meth_content = f.read()
                st.markdown(meth_content)
            else:
                st.info("Methodology file not found at METHODOLOGY.md")
        except Exception as e:
            st.warning(f"Unable to load METHODOLOGY.md: {str(e)}")