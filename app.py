import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import sqlite3
import os
import io
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import uuid
import streamlit.components.v1 as components
from typing import Optional, Dict, Any, List, Tuple

# Explicitly configure robust default sans-serif fonts and silence font lookup warnings
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI', 'Liberation Sans', 'sans-serif']
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Import custom backend modules
from modules.data_validator import CreditRiskDataValidator
from modules.feature_store import apply_macro_layers, export_parquet_snapshot, export_parquet_bytes, export_csv_bytes, update_iv_metadata
from modules.iv_engine import calculate_portfolio_iv, plot_iv_chart, plot_iv_quadrant_chart
from modules.models.dispatcher import run_automl_pipeline
from modules.models.shap_explainer import CreditRiskExplainer
from modules.eda_visualizer import CreditRiskEDA
from modules.login_page import render_login_signup_page, render_authenticated_user_bar

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
    
    # Authentication & Access Control
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None

    # Workflow Flags
    if 'data_ingested' not in st.session_state:
        st.session_state.data_ingested = False
    if 'layers_applied' not in st.session_state:
        st.session_state.layers_applied = False
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False

    # Data Payloads
    if 'raw_upload_df' not in st.session_state:
        st.session_state.raw_upload_df = None
    if 'primary_df' not in st.session_state:
        st.session_state.primary_df = None
    if 'final_layered_df' not in st.session_state:
        st.session_state.final_layered_df = None
    if 'validation_messages' not in st.session_state:
        st.session_state.validation_messages = []
    if 'cutoff_stats' not in st.session_state:
        st.session_state.cutoff_stats = None
    if 'iv_df' not in st.session_state:
        st.session_state.iv_df = None

    # AutoML & XAI Payloads
    if 'automl_results' not in st.session_state:
        st.session_state.automl_results = None
    if 'selected_borrower_idx' not in st.session_state:
        st.session_state.selected_borrower_idx = 0


def reset_portfolio_state():
    """Wipes in-memory session, tables, and models to restart from a clean slate."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.data_ingested = False
    st.session_state.layers_applied = False
    st.session_state.training_completed = False
    st.session_state.raw_upload_df = None
    st.session_state.primary_df = None
    st.session_state.final_layered_df = None
    st.session_state.automl_results = None
    st.session_state.cutoff_stats = None
    st.session_state.iv_df = None
    st.session_state.validation_messages = []
    st.session_state.selected_borrower_idx = 0
    try:
        st.session_state.duck_conn.execute("DROP TABLE IF EXISTS ml_features")
        st.session_state.duck_conn.execute("DROP TABLE IF EXISTS temp_df")
        st.session_state.duck_conn.execute("DROP TABLE IF EXISTS macro_warehouse_temp")
    except Exception:
        pass


from modules.portfolio_balancer import (
    generate_automated_segments, 
    balance_portfolio_by_defaulter_pct
)


init_single_page_state()


# ==============================================================================
# AUTHENTICATION & ACCESS GATE
# ==============================================================================
if not st.session_state.get('authenticated', False):
    render_login_signup_page()
    st.stop()


# ==============================================================================
# AUTHENTICATED USER SESSION BAR
# ==============================================================================
render_authenticated_user_bar()
st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)


# ==============================================================================
# TOP-LEVEL TWO-TAB NAVIGATION
# ==============================================================================
tab_engine, tab_sources = st.tabs([
    "📊 Dashboard", 
    "📜 Data Governance"
])


# ##############################################################################
# TAB 1: CREDIT RISK AUTOML ENGINE (OPERATIONAL WORKFLOW)
# ##############################################################################
with tab_engine:
    
    # We define sample_csv here since we removed the expander that used to define it
    sample_csv = (
        "borrower_id,borrower_type,loan_no,loan_date,due_date,payoff_date,tenure_days,amount,default_flag,country_code\n"
        "CUST-10492,Retail,LN-88101,2025-01-10,2025-02-10,2025-02-10,30,45000,0,KEN\n"
        "CUST-10493,Retail,LN-88102,2025-01-14,2025-02-14,2025-02-28,30,50000,1,KEN\n"
        "CUST-20831,SME,LN-88103,2025-01-15,2025-02-15,2025-02-10,30,120000,0,KEN\n"
        "CUST-30119,Microfinance,LN-88104,2025-01-16,2025-02-16,2025-02-16,30,25000,0,KEN\n"
    )

    if not st.session_state.training_completed:
        # ==============================================================================
        # DATA INGESTION MOCKUP UI
        # ==============================================================================
        st.markdown('''
        st.markdown('''
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
        
        /* Target the left column (Info - Blue) */
        div[data-testid="stTabs"] div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:nth-child(1) {
            background-color: #2b5b7c;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.15);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        /* Target the right column (Upload - Orange) */
        div[data-testid="stTabs"] div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:nth-child(2) {
            background-color: #d8982a;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Make the file dropzone container white */
        div[data-testid="stTabs"] div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:nth-child(2) div[data-testid="stFileUploader"] {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 20px;
            margin-top: 10px;
            border: 1px dashed #cbd5e1;
        }

        /* Streamlit File Uploader Override */
        div[data-testid='stFileUploader'] section {
            background-color: transparent !important;
            border: none !important;
            padding: 10px !important;
        }
        div[data-testid='stFileUploader'] section > button { display: none; }
        
        /* Run tool button styling - Massive Orange Block */
        button[kind="primary"] {
            background-color: #d8982a !important;
            color: #000 !important;
            font-weight: 900 !important;
            border: none !important;
            border-radius: 8px !important;
            height: 60px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            display: flex;
            justify-content: flex-start;
            padding-left: 30px !important;
            margin-top: 40px !important;
        }
        button[kind="primary"] p {
            font-family: 'Press Start 2P', monospace !important;
            font-size: 16px !important;
            margin: 0;
            text-transform: uppercase;
        }
        button[kind="primary"]:hover {
            background-color: #c4821f !important;
        }
        </style>
        ''', unsafe_allow_html=True)
        
        col_info, col_up = st.columns([1, 1.3], gap="large")
        
        with col_info:
            st.markdown('''
            <div style="font-size: 32px; font-weight: 900; margin-bottom: 15px; font-family: 'DM Sans', sans-serif; color: #ffffff; line-height: 1.2;">What is happening<br>with your data?</div>
            <div style="font-size: 15px; line-height: 1.6; font-family: 'Century Gothic', sans-serif; color: #e2e8f0;">
                When you upload your financial data or portfolios into Credit Analyze, we process it entirely in temporary memory. 
                <span style="color: #fcd34d; font-style: italic;">Think of it like reading a document on a whiteboard, once you close your browser or log out, the whiteboard is wiped completely clean.</span> 
                Your financial files are never permanently saved to our servers, nor are they downloaded to your computer's hard drive.
            </div>
            <div style="margin-top: 25px; font-size: 13px; color: #e2e8f0; text-decoration: underline; cursor: pointer;">Read Data Governance</div>
            ''', unsafe_allow_html=True)
            
        with col_up:
            st.markdown('''
            <div style="width: 32px; height: 32px; border-radius: 50%; border: 1px solid #e2e8f0; display: flex; align-items: center; justify-content: center; color: #4f46e5; font-size: 16px; margin-bottom: 10px; margin-left: auto; margin-right: auto;">↑</div>
            <div style="font-weight: 800; color: #0f172a; font-size: 14px; margin-bottom: 5px; text-align: center;">Click to select or drag & drop panel CSV file</div>
            ''', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Upload", type=["csv"], label_visibility="collapsed")
            
            if uploaded_file is not None:
                st.session_state.raw_upload_df = pd.read_csv(uploaded_file)
                st.markdown(f'<div style="text-align: center;"><span style="font-size: 12px; color: #64748b; background: #f1f5f9; padding: 4px 8px; border-radius: 4px;">File loaded: {uploaded_file.name}</span></div>', unsafe_allow_html=True)
            elif st.session_state.raw_upload_df is not None:
                st.markdown('<div style="text-align: center;"><span style="font-size: 12px; color: #64748b; background: #f1f5f9; padding: 4px 8px; border-radius: 4px;">File loaded from cache</span></div>', unsafe_allow_html=True)
            
            st.write("")
            # Bottom footer for sample template
            col_c1, col_c2 = st.columns([1, 1])
            with col_c1:
                st.markdown('<div style="margin-top:10px; font-size: 11px; color: #475569;">ⓘ Need baseline template format?</div>', unsafe_allow_html=True)
            with col_c2:
                st.download_button(
                    label="↓ Download Sample Template",
                    data=sample_csv,
                    file_name="kba_sample_credit_panel.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
        # Anonymize Checkbox
        col_cb1, col_cb2 = st.columns([1.5, 4])
        with col_cb1:
            st.markdown('<div style="font-weight: 800; font-size: 18px; color: #000; font-family: \'DM Sans\', sans-serif; margin-top: 10px; padding-left: 20px;">Anonymize your data?</div>', unsafe_allow_html=True)
        with col_cb2:
            st.markdown('<div style="margin-top: 15px;">', unsafe_allow_html=True)
            anonymize = st.checkbox("Anonymize", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.write("")
        st.write("")
        
        # RUN TOOL Button
        run_btn = st.button("RUN TOOL", type="primary", use_container_width=True)
        
        status_text = st.empty()
        progress_bar = st.empty()
        
        if run_btn:
            if st.session_state.raw_upload_df is None:
                status_text.error("Please upload a file first!")
            else:
                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.markdown(f"<div class='tool-loading-text'>{msg}</div>", unsafe_allow_html=True)

                try:
                    update_progress(10, "Validating data...")
                    is_valid, msgs, clean_df, dlq = CreditRiskDataValidator.validate_ingestion_payload(st.session_state.raw_upload_df)
                    st.session_state.validation_messages = msgs

                    if not is_valid:
                        for m in msgs:
                            st.error(m)
                    else:
                        update_progress(25, "Balancing portfolio...")
                        balanced_df, cutoff_stats = balance_portfolio_by_defaulter_pct(clean_df, None, stratify_col=None)
                        cutoff_stats["segment_dimension"] = "None"
                        cutoff_stats["analysis_scope"] = "Full Portfolio"

                        balanced_df['session_id'] = st.session_state.session_id
                        balanced_df['country_code'] = "KEN" # Default
                        st.session_state.primary_df = balanced_df.copy()
                        st.session_state.cutoff_stats = cutoff_stats

                        update_progress(40, "Ingesting into DuckDB...")
                        st.session_state.duck_conn.register('temp_df', balanced_df)
                        st.session_state.duck_conn.execute("CREATE OR REPLACE TABLE ml_features AS SELECT * FROM temp_df")

                        update_progress(55, "Joining macro data...")
                        layered_df = apply_macro_layers(st.session_state.duck_conn, ['macro_layer.db'], ALTERNATIVE_DATA_DIR)
                        st.session_state.final_layered_df = layered_df.copy()

                        update_progress(70, "Calculating IV...")
                        iv_df = calculate_portfolio_iv(layered_df, target="default_flag")
                        update_iv_metadata(st.session_state.duck_conn, iv_df)
                        st.session_state.iv_df = iv_df

                        update_progress(85, "Training models...")
                        results = run_automl_pipeline(
                            st.session_state.final_layered_df,
                            optimize_metric="PR-AUC",
                            tune_hyperparams=True,
                            create_ensemble=True,
                            progress_callback=lambda p, m: update_progress(int(85 + (p * 0.15)), m)
                        )

                        st.session_state.automl_results = results
                        st.session_state.data_ingested = True
                        st.session_state.layers_applied = True
                        st.session_state.training_completed = True
                        update_progress(100, "Done!")
                        st.rerun()

                except Exception as e:
                    status_text.error(f"❌ Error: {str(e)}")
                    st.exception(e)

    else:
        # ==============================================================================
        # LIVE DASHBOARD & ONSET DEFAULT SCREENING
        # ==============================================================================
        # --- CUTOFF & INGESTION SUMMARY TOP BANNER ---
        cutoff = st.session_state.cutoff_stats
        col_sum_m, col_sum_r1, col_sum_r2 = st.columns([3, 1, 1])
        with col_sum_m:
            if cutoff:
                scope_badge = f" &nbsp;|&nbsp; <b>Scope:</b> <span style='color: #60a5fa;'>{cutoff.get('analysis_scope', 'All Segments')}</span>" if 'analysis_scope' in cutoff else ""
                st.markdown(f"""
                <div style="background: rgba(37, 99, 235, 0.09); border: 1px solid rgba(37, 99, 235, 0.3); border-radius: 10px; padding: 12px 18px; margin-bottom: 12px;">
                    <span style="font-size: 14px; font-weight: 800; color: #3b82f6;">🎯 Portfolio Balanced & Analyzed:</span> 
                    Retained <b>{cutoff['kept_total']:,} records</b> ({cutoff['kept_def']:,} Defaulters &nbsp;|&nbsp; {cutoff['kept_non_def']:,} Performing loans &nbsp;=&nbsp; <b>{cutoff['actual_pct']:.1f}% Defaulters</b>){scope_badge}. 
                    Cut off <b>{cutoff['cut_off']:,}</b> excess records from original {cutoff['original_total']:,} rows.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success(f"✅ Ingested and evaluated {len(st.session_state.primary_df):,} records!")

        with col_sum_r1:
            if st.button("⚙️ Adjust Defaulter %", width='stretch', help="Return to configuration panel to adjust target % without losing uploaded file"):
                st.session_state.training_completed = False
                st.session_state.automl_results = None
                st.rerun()

        with col_sum_r2:
            if st.button("🔄 Reset Portfolio", width='stretch', help="Wipes all memory stores and uploads a new portfolio"):
                reset_portfolio_state()
                st.rerun()

        st.write("---")

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
            ticket_seg = df['loan_ticket_segment'].iloc[idx] if 'loan_ticket_segment' in df else "—"

            # Get Adverse Action Reasons
            if explainer is not None:
                reasons = explainer.get_borrower_adverse_reasons(idx, top_k=3)
                reason_str = "<br>".join([f"• {r['feature_name']}: {r['reason_text']} ({r['impact_percentage']}%)" for r in reasons])
            else:
                reason_str = "• Historical default history and debt ratio elevate risk."

            table_rows.append({
                "Index": idx,
                "Borrower ID": borrower_id,
                "Ticket Segment": ticket_seg,
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
            st.dataframe(display_table, width='stretch', height=280)

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
            ticket_seg_val = get_card_field(card_row, 'loan_ticket_segment', fallback="—")

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
                    <p><b>Borrower ID:</b> {selected_borrower_id} &nbsp;|&nbsp; <b>Jurisdiction:</b> Kenya{county_line} &nbsp;|&nbsp; <b>Segment:</b> {ticket_seg_val}</p>
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
                        st.image(img_bytes, width='stretch')
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
                    st.dataframe(leaderboard_df, width='stretch')
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
                            st.plotly_chart(plotly_fig, width='stretch')
                        else:
                            bar_bytes = explainer.generate_bar_plot_bytes()
                            if bar_bytes:
                                st.image(bar_bytes, width='stretch')
                            else:
                                st.info("Feature importance bar plot unavailable for this model architecture.")
                        
                        bar_bytes = explainer.generate_bar_plot_bytes()
                        if bar_bytes:
                            st.download_button(
                                label="⬇️ Download Portfolio SHAP Bar Graph (.PNG)",
                                data=bar_bytes,
                                file_name="portfolio_shap_feature_importance.png",
                                mime="image/png",
                                width='stretch'
                            )

                    with tab_bee:
                        beeswarm_bytes = explainer.generate_beeswarm_plot_bytes()
                        if beeswarm_bytes:
                            st.image(beeswarm_bytes, width='stretch')
                            st.caption("Dots represent individual borrowers. Color denotes feature value (Red = High, Blue = Low). Position on X-axis denotes risk impact.")

                    with tab_tbl:
                        imp_df = explainer.get_global_feature_importance_df(top_n=25)
                        st.dataframe(imp_df, width='stretch')
                        st.download_button(
                            label="⬇️ Export Feature Importance Table (.CSV)",
                            data=imp_df.to_csv(index=False).encode('utf-8'),
                            file_name="kba_global_feature_importance.csv",
                            mime="text/csv",
                            width='stretch'
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
                width='stretch'
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
                width='stretch'
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
                        width='stretch'
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
                    width='stretch'
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
                    width='stretch'
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
        # SECTION 2.6: INFORMATION VALUE (IV) SCREENING EXPANDER
        # ==============================================================================
        with st.expander("🏷️ Information Value (IV) Screening & Feature Catalog", expanded=False):
            iv_df = st.session_state.iv_df if st.session_state.iv_df is not None else calculate_portfolio_iv(st.session_state.final_layered_df, target="default_flag")
            
            col_iv_table, col_iv_chart = st.columns([1, 1])
            with col_iv_table:
                st.dataframe(
                    iv_df.style.background_gradient(subset=["Information Value (IV)"], cmap="YlGn"),
                    width='stretch'
                )
                
                # Download IV Table
                st.download_button(
                    label="📥 Download IV Table (.CSV)",
                    data=iv_df.to_csv(index=False).encode('utf-8'),
                    file_name="kba_iv_screening.csv",
                    mime="text/csv",
                    width='stretch'
                )
                
            with col_iv_chart:
                iv_fig = plot_iv_chart(iv_df)
                st.plotly_chart(iv_fig, width='stretch')

            st.write("---")
            st.markdown("#### 🧭 Variable Discoverability Matrix")
            st.caption("Plots Collection Hardness vs. Evidence x Information Value (IV) to prioritize feature acquisition.")
            quadrant_fig = plot_iv_quadrant_chart(iv_df)
            if quadrant_fig:
                st.plotly_chart(quadrant_fig, width='stretch')

        # ==============================================================================
        # SECTION 2.5: EXPLORATORY DATA ANALYSIS (EDA) & DESCRIPTIVE STATISTICS EXPANDER
        # ==============================================================================
        with st.expander("📊 Exploratory Data Analysis (EDA) & Descriptive Statistics Hub", expanded=False):
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
                st.dataframe(stats_df, width='stretch')
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.download_button(
                        label="📥 Download Descriptive Statistics (.CSV)",
                        data=stats_df.to_csv(index=False).encode('utf-8'),
                        file_name="kba_descriptive_statistics.csv",
                        mime="text/csv",
                        width='stretch'
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
                        width='stretch'
                    )
                    
            with tab_dist:
                with st.popover("ℹ️ Class Imbalance in Distribution Plots"):
                    st.markdown("""
                    ### 📈 Distribution Histograms by Loan Outcome
                    
                    * **🟢 Performing vs. 🔴 Defaulted:**  
                      Histograms compare feature distributions between paying borrowers and defaulters.
                    * **Discriminatory Power:**  
                      Features with clear separation between green and red distributions have strong predictive power.
                    """)
                dist_fig = CreditRiskEDA.generate_feature_distributions_fig(active_eda_df)
                if dist_fig:
                    st.plotly_chart(dist_fig, width='stretch')
                dist_png = CreditRiskEDA.generate_feature_distributions_bytes(active_eda_df)
                if dist_png:
                    st.download_button(
                        label="⬇️ Download Distribution Histograms (.PNG)",
                        data=dist_png,
                        file_name="kba_feature_distributions.png",
                        mime="image/png",
                        width='stretch'
                    )
                    
            with tab_corr:
                with st.popover("ℹ️ Understanding Pearson Collinearity"):
                    st.markdown("""
                    ### 🔥 Pearson Cross-Correlation & Multicollinearity
                    
                    * **Correlation Coefficient ($r$):**  
                      Ranges from $-1.0$ (perfect inverse correlation) to $+1.0$ (perfect direct correlation).
                    * **Multicollinearity Risk ($|r| > 0.80$):**  
                      Highly correlated features provide redundant information and can inflate variance in linear and tree models.
                    """)
                corr_fig = CreditRiskEDA.generate_correlation_heatmap_fig(active_eda_df)
                if corr_fig:
                    st.plotly_chart(corr_fig, width='stretch')
                
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    corr_png = CreditRiskEDA.generate_correlation_heatmap_bytes(active_eda_df)
                    if corr_png:
                        st.download_button(
                            label="⬇️ Download Correlation Heatmap (.PNG)",
                            data=corr_png,
                            file_name="kba_correlation_heatmap.png",
                            mime="image/png",
                            width='stretch'
                        )
                with col_c2:
                    corr_matrix = CreditRiskEDA.generate_correlation_matrix(active_eda_df)
                    if not corr_matrix.empty:
                        st.download_button(
                            label="📥 Download Correlation Matrix (.CSV)",
                            data=corr_matrix.to_csv().encode('utf-8'),
                            file_name="kba_correlation_matrix.csv",
                            mime="text/csv",
                            width='stretch'
                        )
                        
            with tab_box:
                with st.popover("ℹ️ Tukey Boxplots & Outlier Detection"):
                    st.markdown("""
                    ### 📦 Outlier Bounds & Quantile Spread
                    
                    * **Box Dimensions:** Represents the Interquartile Range ($IQR = Q3 - Q1$, middle 50%).
                    * **Whiskers:** Extend to $1.5 \\times IQR$ from the upper/lower quartiles.
                    * **Outliers (Dots):** Loan amounts or tenors exceeding the whiskers indicate extreme borrowing behavior.
                    """)
                box_fig = CreditRiskEDA.generate_boxplots_by_target_fig(active_eda_df)
                if box_fig:
                    st.plotly_chart(box_fig, width='stretch')

        # ==============================================================================
        # SECTION 2: MERGED FEATURE STORE SNAPSHOT EXPANDER
        # ==============================================================================
        with st.expander("🔍 Merged Feature Store Snapshot & Data Science Exports", expanded=False):
            st.dataframe(st.session_state.final_layered_df.head(5), width='stretch')
            col_exp_fs1, col_exp_fs2 = st.columns(2)
            with col_exp_fs1:
                fs_csv = export_csv_bytes(st.session_state.final_layered_df)
                st.download_button(
                    label="📥 Export Feature Store (.CSV)",
                    data=fs_csv,
                    file_name="kba_feature_store_snapshot.csv",
                    mime="text/csv",
                    width='stretch'
                )
            with col_exp_fs2:
                try:
                    fs_parquet = export_parquet_bytes(st.session_state.final_layered_df)
                    st.download_button(
                        label="📦 Export Feature Store (.Parquet)",
                        data=fs_parquet,
                        file_name="kba_feature_store_snapshot.parquet",
                        mime="application/octet-stream",
                        width='stretch'
                    )
                except Exception:
                    st.caption("Parquet export engine (pyarrow) optional")


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
            import importlib
            import modules.ai_assistant
            importlib.reload(modules.ai_assistant)
            from modules.ai_assistant import NLQueryEngine, OllamaClient
            _HAS_AI = True
        except Exception:
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
                        value=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                        help="Default is http://127.0.0.1:11434 for local sessions. If deployed on Streamlit Cloud, enter your public tunnel URL (e.g., via ngrok or cloudflared).",
                        key="ollama_base",
                    )
                with cfg_col2:
                    st.write("")
                    st.write("")
                    st.button("🔄 Test / Refresh", key="recheck_ollama", width='stretch')

            _oc = OllamaClient(base_url=ollama_base.strip())
            _available_models = _oc.list_models(timeout=15)

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
                last_err = getattr(_oc, "last_error", None)
                err_hint = f"\n\n**Error details:** `{last_err}`" if last_err else ""
                st.caption(
                    f"🟡 **Ollama not detected** at `{ollama_base.strip()}`{err_hint} — using the deterministic offline analytical engine.\n\n"
                    f"* **Tunneling from your PC?** Ensure Ollama was started with `OLLAMA_ORIGINS=*` so it doesn't block tunnel traffic with 403 Forbidden.\n"
                    f"* **Running locally?** Try `http://127.0.0.1:11434`."
                )

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
                    st.plotly_chart(fig, width='stretch')

                # Tables
                for tbl in ai_result.get("tables", []) or []:
                    st.dataframe(tbl, width='stretch')


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
            width='stretch',
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
                width='stretch'
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
                width='stretch'
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


if __name__ == "__main__":
    import sys
    import os

    # Auto-relaunch into Python 3.10 virtual environment (.venv) if invoked under Python 3.12+ (for PyCaret compatibility)
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
    if sys.version_info >= (3, 12) and os.path.exists(venv_python) and sys.executable.lower() != venv_python.lower():
        import subprocess
        cmd = [venv_python, os.path.abspath(__file__)] + sys.argv[1:]
        sys.exit(subprocess.call(cmd))

    try:
        from streamlit.web import cli as stcli
    except ImportError:
        try:
            import streamlit.cli as stcli
        except ImportError:
            stcli = None

    if stcli is not None and not st.runtime.exists():
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:]
        sys.exit(stcli.main())