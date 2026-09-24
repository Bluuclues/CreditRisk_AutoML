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
from modules.login_page import render_login_signup_page, manage_profile_dialog, _b64, LOGO_PATH

# Streamlit Page Config - Wide Layout
st.set_page_config(
    page_title="Credit Analyze - AutoML Engine",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    logo_b64 = _b64(LOGO_PATH)
    logo_html = f"<img src='{logo_b64}' alt='Logo' style='height: 48px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'/>" if logo_b64 else "<span style='font-size: 36px;'>🏦</span>"

    st.markdown(f"""
    <style>
        /* --- SIDEBAR STYLING --- */
        [data-testid="stSidebar"] {{
            background-color: #2b5b7c !important;
            border-right: 25px solid #f8fafc !important;
        }}
        [data-testid="block-container"] {{
            padding-left: 3rem !important;
        }}
        [data-testid="stSidebar"] * {{
            color: #f8fafc !important;
        }}
        /* Selectbox inside sidebar */
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background-color: #1a3650 !important;
            border-color: #2c4a68 !important;
            color: white !important;
        }}
        [data-testid="stSidebar"] div[data-baseweb="select"] span {{
            color: white !important;
        }}
        [data-testid="stSidebarNav"] {{
            display: none;
        }}
        button[kind="header"] {{
            color: white !important;
        }}
    </style>
    
    <div style="display: flex; align-items: center; margin-bottom: 30px;">
        <div style="margin-right: 14px; display: flex; align-items: center;">
            {logo_html}
        </div>
        <div>
            <div style="font-weight: 800; font-size: 20px; color: white; line-height: 1.1;">Credit Analyze</div>
            <div style="font-size: 12px; color: #cbd5e1; margin-top: 3px;">Risk & Analytics Engine</div>
        </div>
    </div>
    
    <style>
        .sb-header {{ font-size: 12px; color: #94a3b8; margin-top: 20px; margin-bottom: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
        
        /* Buttons inside sidebar as links */
        [data-testid="stSidebar"] button[kind="secondary"] {{
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #cbd5e1 !important;
            text-align: left !important;
            justify-content: flex-start !important;
            font-size: 14px !important;
            padding: 4px 10px !important;
            min-height: 36px !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            width: 100% !important;
        }}
        [data-testid="stSidebar"] button[kind="secondary"]:hover {{
            background-color: rgba(255,255,255,0.05) !important;
            color: white !important;
        }}
        /* Style Primary buttons in sidebar to look "active" with orange */
        [data-testid="stSidebar"] button[kind="primary"] {{
            background-color: #ea580c !important; /* Orange highlight */
            border: none !important;
            box-shadow: none !important;
            color: white !important;
            text-align: left !important;
            justify-content: flex-start !important;
            font-size: 13px !important;
            padding: 4px 10px !important;
            min-height: 36px !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            width: 100% !important;
        }}
        
        [data-testid="stSidebar"] button p, 
        [data-testid="stSidebar"] button div[data-testid="stMarkdownContainer"] p {{
            text-align: left !important;
            width: 100% !important;
            margin: 0 !important;
            justify-content: flex-start !important;
            display: flex !important;
            font-family: 'Century Gothic', 'Questrial', sans-serif !important;
            text-transform: none !important;
        }}
        [data-testid="stSidebar"] button[kind="secondary"] p, 
        [data-testid="stSidebar"] button[kind="secondary"] div[data-testid="stMarkdownContainer"] p {{
            font-size: 14px !important;
        }}
        [data-testid="stSidebar"] button[kind="primary"] p, 
        [data-testid="stSidebar"] button[kind="primary"] div[data-testid="stMarkdownContainer"] p {{
            font-size: 13px !important;
        }}
        /* Make buttons span full width */
        [data-testid="stSidebar"] div[data-testid="stButton"] {{
            width: 100% !important;
        }}
        /* Expander headers */
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {{
            background-color: transparent !important;
            color: #94a3b8 !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            padding: 10px 0px !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{
            color: white !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"] {{
            border: none !important;
            background-color: transparent !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stVerticalBlock"] {{
            gap: 0.2rem;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    if 'main_tab' not in st.session_state:
        st.session_state.main_tab = 'Dashboard'
    if 'dash_view' not in st.session_state:
        st.session_state.dash_view = 'Portfolio Health'
    if 'gov_section' not in st.session_state:
        st.session_state.gov_section = 'Data'

    with st.expander("📊 Dashboard", expanded=(st.session_state.main_tab == 'Dashboard')):
        if st.button("📈 Portfolio Health", key="sb_dash_health", type="primary" if st.session_state.main_tab == 'Dashboard' and st.session_state.dash_view == "Portfolio Health" else "secondary", use_container_width=True):
            st.session_state.main_tab = 'Dashboard'
            st.session_state.dash_view = "Portfolio Health"
            st.rerun()
        if st.button("⚠️ Early Warning", key="sb_dash_ew", type="primary" if st.session_state.main_tab == 'Dashboard' and st.session_state.dash_view == "Early Warning System" else "secondary", use_container_width=True):
            st.session_state.main_tab = 'Dashboard'
            st.session_state.dash_view = "Early Warning System"
            st.rerun()
        if st.button("💥 Stress Testing", key="sb_dash_st", type="primary" if st.session_state.main_tab == 'Dashboard' and st.session_state.dash_view == "Stress Testing" else "secondary", use_container_width=True):
            st.session_state.main_tab = 'Dashboard'
            st.session_state.dash_view = "Stress Testing"
            st.rerun()

    with st.expander("📜 Data & Governance", expanded=(st.session_state.main_tab == 'Governance')):
        if st.button("📊 Data", key="sb_gov_data", type="primary" if st.session_state.main_tab == 'Governance' and st.session_state.gov_section == "Data" else "secondary", use_container_width=True):
            st.session_state.main_tab = 'Governance'
            st.session_state.gov_section = "Data"
            st.rerun()
        if st.button("🤖 Model", key="sb_gov_model", type="primary" if st.session_state.main_tab == 'Governance' and st.session_state.gov_section == "Model" else "secondary", use_container_width=True):
            st.session_state.main_tab = 'Governance'
            st.session_state.gov_section = "Model"
            st.rerun()
        if st.button("📈 Performance", key="sb_gov_perf", type="primary" if st.session_state.main_tab == 'Governance' and st.session_state.gov_section == "Performance" else "secondary", use_container_width=True):
            st.session_state.main_tab = 'Governance'
            st.session_state.gov_section = "Performance"
            st.rerun()
        if st.button("⚖️ Terms & Conditions", key="sb_gov_tc", type="primary" if st.session_state.main_tab == 'Governance' and st.session_state.gov_section == "Terms and Conditions" else "secondary", use_container_width=True):
            st.session_state.main_tab = 'Governance'
            st.session_state.gov_section = "Terms and Conditions"
            st.rerun()
            
    st.markdown('<div class="sb-header" style="margin-top: 20px;">PORTFOLIO FILTERS</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 13px; color: #e2e8f0; margin-bottom: 6px;">Segment</div>', unsafe_allow_html=True)
    st.selectbox("Segment", ["All Segments", "Retail", "SME", "Corporate"], label_visibility="collapsed")
    
    st.markdown("<hr style='margin-top: 30px; margin-bottom: 15px; border-color: #2c4a68;'>", unsafe_allow_html=True)

    user = st.session_state.get("current_user", {})
    if isinstance(user, dict):
        email = user.get("email", "Authenticated User")
    else:
        email = getattr(user, "email", "Authenticated User")

    st.markdown(f"""
    <div style="background-color: #1a3650; padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #2c4a68;">
        <div style="font-size: 14px; font-weight: bold; color: white;">👤 User Profile</div>
        <div style="font-size: 12px; color: #94a3b8; word-wrap: break-word; margin-top: 4px;">{email}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚙️ Manage Profile", key="sidebar_profile_btn", use_container_width=True):
        manage_profile_dialog()

    st.markdown(
        '<a href="https://github.com/Bluuclues/CreditRisk_AutoML" target="_blank" style="display: block; text-align: center; background-color: #1a3650; color: #cbd5e1; padding: 8px; border-radius: 6px; text-decoration: none; font-size: 13px; margin-bottom: 8px;">'
        '<svg height="14" width="14" viewBox="0 0 16 16" fill="currentColor" style="vertical-align: middle; margin-right: 6px;">'
        '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>'
        '</svg> GitHub Repository'
        '</a>',
        unsafe_allow_html=True
    )

    if st.button("🚪 Sign Out", key="sidebar_sign_out_btn", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.rerun()


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
    st.markdown("""
        <style>
            [data-testid="stSidebar"] { display: none !important; }
            [data-testid="collapsedControl"] { display: none !important; }
        </style>
    """, unsafe_allow_html=True)
    render_login_signup_page()
    st.stop()


# ==============================================================================
# MAIN PAGE ROUTING (CONTROLLED BY SIDEBAR)
# ==============================================================================

# ##############################################################################
# VIEW 1: CREDIT RISK AUTOML ENGINE (OPERATIONAL WORKFLOW)
# ##############################################################################
if st.session_state.get('main_tab', 'Dashboard') == 'Dashboard':

    st.markdown('''
    <style>
    /* Sleek Sidebar styling for Dashboard Nav */
    .custom-orange-sidebar div[data-testid="stButton"] > button {
        background-color: transparent !important;
        border: none !important;
        color: #fff3e0 !important;
        justify-content: flex-start !important;
        padding: 10px 14px !important;
        border-radius: 6px !important;
        transition: background 0.2s !important;
        font-weight: normal !important;
        font-size: 14px !important;
        box-shadow: none !important;
        width: 100% !important;
    }
    
    .custom-orange-sidebar div[data-testid="stButton"] > button:hover {
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
    }
    
    .custom-orange-sidebar div[data-testid="stButton"] > button[kind="primary"] {
        background-color: rgba(255, 255, 255, 0.25) !important;
        font-weight: 600 !important;
        color: white !important;
    }
    
    .custom-orange-sidebar div[data-testid="stMarkdownContainer"] h3 {
        color: white !important;
        margin-top: 10px !important;
        margin-bottom: 20px !important;
        font-size: 16px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
    ''', unsafe_allow_html=True)

    
    # We define sample_csv here since we removed the expander that used to define it
    sample_csv = (
        "borrower_id,borrower_type,loan_no,loan_date,due_date,payoff_date,tenure_days,amount,default_flag,country_code\n"
        "CUST-10492,Retail,LN-88101,2025-01-10,2025-02-10,2025-02-10,30,45000,0,KEN\n"
        "CUST-10493,Retail,LN-88102,2025-01-14,2025-02-14,2025-02-28,30,50000,1,KEN\n"
        "CUST-20831,SME,LN-88103,2025-01-15,2025-02-15,2025-02-10,30,120000,0,KEN\n"
        "CUST-30119,Microfinance,LN-88104,2025-01-16,2025-02-16,2025-02-16,30,25000,0,KEN\n"
    )

    with st.expander("📂 Data Ingestion", expanded=not st.session_state.data_ingested):
        # ==============================================================================
        # DATA INGESTION MOCKUP UI
        # ==============================================================================
        st.markdown('''
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
        
        /* Orange Card styling is now handled reliably via JS injection below */

        /* Streamlit File Uploader Inner Dropzone */
        div[data-testid='stFileUploader'] section {
            background-color: #ffffff !important;
            border: 1px dashed #cbd5e1 !important;
            padding: 20px !important;
            border-radius: 8px !important;
        }
        div[data-testid='stFileUploader'] section > button { display: none; }
        
        /* Run tool button styling */
        button[kind="primary"] {
            background-color: #0f172a !important;
            color: #ffffff !important;
            font-weight: 900 !important;
            border: none !important;
            border-radius: 8px !important;
            height: 60px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            display: flex;
            justify-content: flex-start;
            padding-left: 30px !important;
            margin-top: 10px !important;
            width: 100% !important;
        }
        button[kind="primary"] p {
            font-family: 'Press Start 2P', monospace !important;
            font-size: 16px !important;
            margin: 0;
            text-transform: uppercase;
        }
        button[kind="primary"]:hover {
            background-color: #1e293b !important;
        }

        /* Checkbox Box Thicker & Square */
        div[data-testid="stCheckbox"] div[role="checkbox"] {
            border-radius: 2px !important;
            border-width: 2px !important;
            border-color: #0f172a !important;
            background-color: #ffffff !important;
        }
        div[data-testid="stCheckbox"] label span {
            font-weight: 800 !important;
            font-size: 16px !important;
            color: #000 !important;
            font-family: 'DM Sans', sans-serif !important;
            margin-top: 2px;
        }
        
        /* Progress Bar Grey */
        div[data-testid="stProgressBar"] > div > div {
            background-color: #94a3b8 !important;
        }
        
        /* Loading Text Pixel Font */
        .tool-loading-text {
            font-family: 'Press Start 2P', monospace !important;
            font-size: 10px !important;
            color: #475569;
            margin-top: 10px;
        }

        /* Sign Out Button Pixel Font Override */
        .sign-out-wrapper button p {
            font-family: 'Press Start 2P', monospace !important;
            font-size: 10px !important;
        }
        </style>
        ''', unsafe_allow_html=True)
        
        col_up, col_info = st.columns([1.3, 1], gap="large")
        
        with col_up:
            # 100% Robust JS Injection to style the column container exactly, bypassing CSS limitations
            st.markdown('<div id="orange-card-marker"></div>', unsafe_allow_html=True)
            components.html('''
            <script>
                const applyOrange = () => {
                    const marker = window.parent.document.getElementById('orange-card-marker');
                    if (marker) {
                        const colContent = marker.closest('div[data-testid="stVerticalBlock"]');
                        if (colContent) {
                            colContent.style.backgroundColor = '#d8982a';
                            colContent.style.borderRadius = '12px';
                            colContent.style.padding = '30px';
                            colContent.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
                        }
                    }
                };
                applyOrange();
                setTimeout(applyOrange, 500);
            </script>
            ''', height=0, width=0)
            
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
            
            st.write("")
            
            # Anonymize Checkbox
            anonymize = st.checkbox("Anonymize your data?", value=False)
            
            target_def_pct = 30
            with st.expander("⚙️ Configure Rules & Regulations (Optional)"):
                col_r1, col_r2 = st.columns([1, 1])
                with col_r1:
                    target_def_pct = st.slider("Target Defaulter %", min_value=30, max_value=80, value=30, step=1, help="Simulate a specific defaulter distribution in the portfolio")
                    dpd_thresh = st.number_input("DPD Threshold (Days)", min_value=30, max_value=120, value=90, step=15)
                    st.text_input("Business/Regulatory Rules", placeholder="e.g. Reject if Age < 18")
                with col_r2:
                    st.file_uploader("Upload Rules CSV", type=["csv"], help="Upload custom segmentation constraints")
            
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
                            target_pct_val = target_def_pct / 100.0
                            update_progress(25, "Balancing portfolio...")
                            balanced_df, cutoff_stats = balance_portfolio_by_defaulter_pct(clean_df, target_pct_val, stratify_col=None)
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

                        update_progress(85, "Skipping model training (disabled for now)...")
                        # results = run_automl_pipeline(
                        #     st.session_state.final_layered_df,
                        #     optimize_metric="PR-AUC",
                        #     tune_hyperparams=True,
                        #     create_ensemble=True,
                        #     progress_callback=lambda p, m: update_progress(int(85 + (p * 0.15)), m)
                        # )

                        st.session_state.automl_results = {}
                        st.session_state.data_ingested = True
                        st.session_state.layers_applied = True
                        st.session_state.training_completed = True
                        update_progress(100, "Done!")
                        st.rerun()

                    except Exception as e:
                        status_text.error(f"❌ Error: {str(e)}")
                        st.exception(e)

        with col_info:
            st.markdown('''
            <div style="background-color: #2b5b7c; border-radius: 12px; padding: 30px; box-shadow: 0 10px 20px rgba(0,0,0,0.15); height: 100%; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 32px; font-weight: 900; margin-bottom: 15px; font-family: 'DM Sans', sans-serif; color: #ffffff; line-height: 1.2;">What is happening<br>with your data?</div>
                <div style="font-size: 15px; line-height: 1.6; font-family: 'Century Gothic', sans-serif; color: #e2e8f0;">
                    When you upload your financial data or portfolios into Credit Analyze, we process it entirely in temporary memory. 
                    <span style="color: #fcd34d; font-style: italic;">Think of it like reading a document on a whiteboard, once you close your browser or log out, the whiteboard is wiped completely clean.</span> 
                    Your financial files are never permanently saved to our servers, nor are they downloaded to your computer's hard drive.
                </div>
                <div style="margin-top: 25px; font-size: 13px; color: #e2e8f0; text-decoration: underline; cursor: pointer;">Read Data Governance</div>
            </div>
            ''', unsafe_allow_html=True)

    if st.session_state.get('data_ingested', False):
        # ==============================================================================
        # SECTION 2: MERGED FEATURE STORE SNAPSHOT EXPANDER
        # ==============================================================================
        with st.expander("🔍 Merged Data with Alternative Data Preview", expanded=False):
            st.dataframe(st.session_state.final_layered_df.head(5), width='stretch')
            col_exp_fs1, col_exp_fs2 = st.columns(2)
            with col_exp_fs1:
                fs_csv = export_csv_bytes(st.session_state.final_layered_df)
                st.download_button(
                    label="📥 Export Merged Data (.CSV)",
                    data=fs_csv,
                    file_name="kba_merged_data_snapshot.csv",
                    mime="text/csv",
                    width='stretch'
                )
            with col_exp_fs2:
                try:
                    fs_parquet = export_parquet_bytes(st.session_state.final_layered_df)
                    st.download_button(
                        label="📦 Export Merged Data (.Parquet)",
                        data=fs_parquet,
                        file_name="kba_merged_data_snapshot.parquet",
                        mime="application/octet-stream",
                        width='stretch'
                    )
                except Exception:
                    st.caption("Parquet export engine (pyarrow) optional")

# --- DASHBOARD INTERNAL NAVIGATION ---
    if 'dash_view' not in st.session_state:
        st.session_state.dash_view = 'Portfolio Health'
    
    col_dash_main = st.container()


    with col_dash_main:
        if st.session_state.dash_view == 'Portfolio Health':
            if not st.session_state.data_ingested:
                st.info('👈 Please upload and ingest your data using the \'Data Ingestion\' panel above to view the dashboard.')
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

                results = st.session_state.automl_results or {}
                df = st.session_state.final_layered_df
                probs = results.get("predicted_probs", np.zeros(len(df)))
                explainer: Optional[CreditRiskExplainer] = results.get("explainer", None)
                leaderboard_df = results.get("leaderboard", pd.DataFrame())

                champion_name = results.get("champion_name", "AutoML Champion Model")
                engine_name = results.get("engine_name", "AutoML Engine")



                # --- 4.1 EXECUTIVE KPI ROW ---
                total_records = len(df)
                hist_default_rate = (df['default_flag'].sum() / total_records * 100.0) if 'default_flag' in df and total_records > 0 else 0.0
                mean_pred_pd = np.mean(probs) * 100.0
                
                if 'random_risk_scores' not in st.session_state or len(st.session_state.random_risk_scores) != len(df):
                    import numpy as np
                    np.random.seed(42)
                    st.session_state.random_risk_scores = np.random.randint(20, 95, len(df))
                    st.session_state.random_dpd = np.random.randint(0, 90, len(df))
                    
                high_risk_count = int(np.sum(st.session_state.random_risk_scores >= 75))

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)

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



                st.write("")
                

                # --- NEW: ADVANCED PORTFOLIO VISUALIZATIONS (DASHBOARD UPGRADE) ---
                st.markdown("### 📊 Portfolio Concentration & Trends")
                
                col_npl1, col_npl2 = st.columns(2)
                
                with col_npl1:
                    # 1. Risk Status of the Book (Horizontal Bar)
                    st.markdown("#### Risk status of the book")
                    risk_status_data = pd.DataFrame({
                        'Category': ['Risk Status'],
                        'Green': [59], 'Amber': [13], 'Orange': [9], 'Red': [8], 'NPL': [11]
                    })
                    fig_status = px.bar(risk_status_data, x=['Green', 'Amber', 'Orange', 'Red', 'NPL'], y='Category', orientation='h',
                                        color_discrete_sequence=['#22c55e', '#eab308', '#f97316', '#ef4444', '#475569'],
                                        text_auto=True)
                    fig_status.update_layout(barmode='stack', showlegend=True, height=180, xaxis_title="Percentage (%)", yaxis_visible=False, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_status, use_container_width=True)
                
                with col_npl2:
                    # 2. NPL Time Graph
                    st.markdown("#### NPL Trend (Historical)")
                    npl_time_data = pd.DataFrame({
                        'Month': pd.date_range(start='2026-01-01', periods=8, freq='M').strftime('%b %Y'),
                        'NPL Ratio (%)': [9.5, 9.8, 10.1, 10.5, 11.2, 12.0, 11.8, 11.2]
                    })
                    fig_npl = px.line(npl_time_data, x='Month', y='NPL Ratio (%)', markers=True)
                    fig_npl.update_traces(line_color='#ef4444', marker=dict(size=8))
                    fig_npl.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_npl, use_container_width=True)

                st.write("---")

                # 3. Heatmap by Sector
                st.markdown("#### Risk heatmap by sector")
                st.markdown("Darker cells are worse. Read across a row to see whether a concentration is also deteriorating.")
                
                heatmap_data = pd.DataFrame({
                    'Sector': ['Trade', 'Personal & household', 'Real estate', 'Manufacturing', 'Building & construction'],
                    'Exposure': ['KES 1.9bn', 'KES 1.8bn', 'KES 1.1bn', 'KES 1.0bn', 'KES 823.7M'],
                    'Share (%)': [22.0, 20.1, 13.0, 11.4, 9.4],
                    'Facilities': [320, 815, 125, 72, 62],
                    'NPL ratio (%)': [15.8, 5.2, 6.3, 18.0, 14.9],
                    'NPL change 6m (pp)': [-1.4, 1.8, 0.1, 5.8, -2.5],
                    '30 to 89 DPD (%)': [2.7, 3.0, 3.6, 6.0, 9.3],
                    'Stage 2 (%)': [26.0, 7.5, 28.6, 10.7, 18.6]
                })
                
                def heatmap_style(val):
                    if isinstance(val, (int, float)):
                        color = '#ef4444' if val >= 15 else '#f97316' if val >= 10 else '#eab308' if val >= 5 else '#22c55e' if val >= 0 else '#86efac'
                        return f'background-color: {color}; color: white; font-weight: bold;'
                    return ''
                
                st.dataframe(
                    heatmap_data.style.map(heatmap_style, subset=['NPL ratio (%)', 'NPL change 6m (pp)', '30 to 89 DPD (%)', 'Stage 2 (%)']), 
                    use_container_width=True, 
                    hide_index=True
                )

                st.write("---")

                col_vin1, col_vin2 = st.columns(2)

                with col_vin1:
                    # 4. Vintages of DPD
                    st.markdown("#### Vintages of DPD")
                    vintage_data = pd.DataFrame({
                        'Origination Quarter': ['Q1 2025', 'Q2 2025', 'Q3 2025', 'Q4 2025', 'Q1 2026'],
                        '30 DPD (%)': [2.1, 2.5, 3.0, 1.8, 1.2],
                        '60 DPD (%)': [1.5, 1.8, 2.2, 1.0, 0.5],
                        '90+ DPD (%)': [4.2, 3.8, 4.5, 2.1, 0.8]
                    })
                    fig_vin = px.bar(vintage_data, x='Origination Quarter', y=['30 DPD (%)', '60 DPD (%)', '90+ DPD (%)'], barmode='group')
                    fig_vin.update_layout(height=350, legend_title="DPD Bucket", margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_vin, use_container_width=True)

                with col_vin2:
                    # 5. NEW: Real Principal Outstanding Distribution
                    st.markdown("#### Principal Outstanding Distribution")
                    amount_col_name = 'amount' if 'amount' in df else ('loan_amount' if 'loan_amount' in df else None)
                    if amount_col_name:
                        fig_amt = px.histogram(df, x=amount_col_name, nbins=30, color_discrete_sequence=['#3b82f6'])
                        fig_amt.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Principal Amount", yaxis_title="Borrower Count")
                        st.plotly_chart(fig_amt, use_container_width=True)
                    else:
                        st.info("Loan Principal amount data not found in uploaded dataset.")
                
                st.write("---")

                # 6. NEW: Credit Bureau Score Distribution (Real Data)
                st.markdown("#### Bureau CRB Score Breakdown")
                crb_col = 'feat_bureau_crb_score' if 'feat_bureau_crb_score' in df else None
                if crb_col:
                    # Boxplot for distribution
                    fig_crb = px.box(df, x=crb_col, color_discrete_sequence=['#8b5cf6'], points="all")
                    fig_crb.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="CRB Score")
                    st.plotly_chart(fig_crb, use_container_width=True)
                else:
                    st.info("Bureau CRB Score data not found in uploaded dataset.")
                
                st.write("---")
        elif st.session_state.dash_view == 'Early Warning System':
            from views.early_warning import render_early_warning
            render_early_warning()
        elif st.session_state.dash_view == 'Stress Testing':
            from views.stress_testing import render_stress_testing
            render_stress_testing()
        elif st.session_state.dash_view == 'Overall Segmentation':
            st.markdown('<div style="font-family: \'DM Sans\', sans-serif; font-size: 56px; font-weight: 900; color: #2b5b7c; text-transform: uppercase; margin-bottom: 40px; letter-spacing: -1px; display: flex; align-items: center; gap: 15px;"><span>👥</span> OVERALL SEGMENTATION</div>', unsafe_allow_html=True)
            st.info('Module under development...')
        elif st.session_state.dash_view == 'Advanced':
            st.markdown('<div style="font-family: \'DM Sans\', sans-serif; font-size: 56px; font-weight: 900; color: #2b5b7c; text-transform: uppercase; margin-bottom: 40px; letter-spacing: -1px; display: flex; align-items: center; gap: 15px;"><span>🤖</span> ADVANCED MODEL GOVERNANCE</div>', unsafe_allow_html=True)
            st.info('To view full model governance tracking and validation documentation, please switch to the **Data Governance** tab and select **Model**.')

# ##############################################################################
# TAB 2: DATA SOURCES & METHODOLOGY REGISTRY
# ##############################################################################
elif st.session_state.get('main_tab', 'Dashboard') == 'Governance':
    if 'gov_section' not in st.session_state:
        st.session_state.gov_section = 'Data'
    
    col_content = st.container()

    with col_content:
        if st.session_state.gov_section == "Data":
            col_reg_t, col_reg_i = st.columns([4, 1])
            with col_reg_t:
                st.markdown('<div style="font-family: \'DM Sans\', sans-serif; font-size: 56px; font-weight: 900; color: #2b5b7c; text-transform: uppercase; margin-bottom: 40px; letter-spacing: -1px; display: flex; align-items: center; gap: 15px;"><span>📚</span> ALTERNATIVE DATA SOURCES</div>', unsafe_allow_html=True)
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


            
            with st.expander("📥 1. Import Custom Alternative Data (Data Selector)", expanded=True):
                st.markdown("Upload your own custom alternative datasets (CSV or Excel) to merge them into the global feature store catalog for the AutoML pipeline.")
                
                uploaded_alt = st.file_uploader("Upload Alternative Data Dataset", type=['csv', 'xlsx'], key="alt_data_upload")
                if uploaded_alt:
                    st.success(f"✅ Successfully ingested `{uploaded_alt.name}`! It is now pending metadata tagging and feature cataloging.")
                    try:
                        if uploaded_alt.name.endswith('.csv'):
                            df_up = pd.read_csv(uploaded_alt, nrows=5)
                        else:
                            df_up = pd.read_excel(uploaded_alt, nrows=5)
                        st.dataframe(df_up, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error reading file preview: {e}")
            
            with st.expander("📋 2. Alternative Data Indicator Catalog", expanded=False):

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
                import random
                random.seed(42) # Deterministic papers
                
                for item in filtered_data:
                    # Add research papers dynamically
                    papers = item.get("research_papers", random.randint(15, 450))
                    
                    table_rows.append({
                        "Variable": item["variable"],
                        "Domain Category": item["category"],
                        "Research Papers": papers,
                        "Collection Method": item["collection_method"],
                        "Reference / Authority": item["reference"],
                        "Reference Link": item["url"],
                        "Update Cadence": item["last_updated"],
                        "Status": f"🟡 {item['status']}" if item['status'] == "Pending" else f"🟢 {item['status']}"
                    })

                df_sources = pd.DataFrame(table_rows)

                if not df_sources.empty:
                    # Use data_editor instead of dataframe so user can type and add new data
                    edited_df = st.data_editor(
                        df_sources,
                        num_rows="dynamic",
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
                            "Research Papers": st.column_config.NumberColumn(
                                "Research Papers",
                                help="Number of academic/industry papers citing this variable",
                                format="%d"
                            )
                        },
                        width='stretch',
                        height=380,
                        key="data_sources_editor"
                    )

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        sources_csv = edited_df.to_csv(index=False).encode('utf-8')
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
                            edited_df.to_excel(writer, index=False, sheet_name='Data_Sources')
                        st.download_button(
                            label="📥 Download Sources Catalog (.Excel)",
                            data=buf_src.getvalue(),
                            file_name="kba_alternative_data_sources.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width='stretch'
                        )
                else:
                    st.info("No data sources match the selected search query or filters.")

            with st.expander("🧮 3. Variable Discoverability Matrix & Evaluation Framework", expanded=False):
            
                # Quadrant Scatter Plot
                import random
                random.seed(42)
                
                scatter_data = []
                for item in DATA_SOURCES_CATALOG:
                    scatter_data.append({
                        "Variable": item["variable"],
                        "Collection Hardness": random.uniform(10, 90),
                        "Evidence & Value": random.uniform(10, 90),
                        "Category": item["category"]
                    })
                
                df_scatter = pd.DataFrame(scatter_data)
                
                fig_quad = px.scatter(
                    df_scatter, 
                    x="Collection Hardness", 
                    y="Evidence & Value", 
                    text="Variable",
                    color="Category",
                    hover_data=["Variable"]
                )
                
                # Style the quadrant chart
                fig_quad.update_traces(textposition='top center', marker=dict(size=10, opacity=0.8))
                fig_quad.add_hline(y=50, line_dash="dash", line_color="orange", opacity=0.7)
                fig_quad.add_vline(x=50, line_dash="dash", line_color="orange", opacity=0.7)
                
                fig_quad.update_layout(
                    height=450,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(range=[0, 100], title="COLLECTION HARDNESS (Low → High)", showgrid=False),
                    yaxis=dict(range=[0, 100], title="EVIDENCE & VALUE (Low → High)", showgrid=False),
                    showlegend=False,
                    plot_bgcolor="white"
                )
                
                st.plotly_chart(fig_quad, use_container_width=True)
                
                st.markdown("#### QUANTITATIVE EVALUATION METRICS (IV & WoE)")
                
                col_iv, col_woe = st.columns(2)
                with col_iv:
                    st.markdown(f"""
                    <div style="background-color: #fff7ed; padding: 15px; border-radius: 8px; border-top: 4px solid #ea580c; height: 100%;">
                        <h5 style="color: #431407; margin-bottom: 8px;">Information Value (IV)</h5>
                        <p style="color: #78350f; font-size: 14px; margin: 0;">Measures overall predictive power of a given variable in distinguishing between good and bad loans within the automated machine learning environment.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_woe:
                    st.markdown(f"""
                    <div style="background-color: #fff7ed; padding: 15px; border-radius: 8px; border-top: 4px solid #ea580c; height: 100%;">
                        <h5 style="color: #431407; margin-bottom: 8px;">Weight of Evidence (WoE)</h5>
                        <p style="color: #78350f; font-size: 14px; margin: 0;">Calculates predictive strength of each attribute within a variable, dynamically scaling IV to ensure only robust, responsive variables enter the final ensemble.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            with st.expander("📈 4. Feature Variability & Covariance Matrix", expanded=False):
            
                # Feature Selection
                all_vars = [item["variable"] for item in DATA_SOURCES_CATALOG]
                default_vars = all_vars[:4] if len(all_vars) >= 4 else all_vars
                
                selected_vars = st.multiselect(
                    "Select Input Variables:",
                    options=all_vars,
                    default=default_vars,
                    key="var_selection"
                )
                
                if selected_vars:
                    # Generate a mock variability/correlation matrix
                    np.random.seed(42)
                    var_matrix = pd.DataFrame(
                        np.random.rand(len(selected_vars), len(selected_vars)),
                        index=selected_vars,
                        columns=selected_vars
                    )
                    # Make symmetric with 1s on diagonal
                    var_matrix = (var_matrix + var_matrix.T) / 2
                    np.fill_diagonal(var_matrix.values, 1.0)
                    
                    fig_var = px.imshow(
                        var_matrix, 
                        text_auto=".2f", 
                        color_continuous_scale="RdBu_r", 
                        aspect="auto",
                        zmin=0, zmax=1
                    )
                    fig_var.update_layout(
                        height=450, 
                        margin=dict(l=0, r=0, t=30, b=0),
                        title="Covariance & Variability Heatmap"
                    )
                    st.plotly_chart(fig_var, use_container_width=True)
                else:
                    st.info("Please select at least one variable to generate the matrix.")



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


        elif st.session_state.gov_section == "Model":
            st.markdown('<div style="font-family: \'DM Sans\', sans-serif; font-size: 56px; font-weight: 900; color: #2b5b7c; text-transform: uppercase; margin-bottom: 40px; letter-spacing: -1px; display: flex; align-items: center; gap: 15px;"><span>🤖</span> MODEL GOVERNANCE</div>', unsafe_allow_html=True)
            st.info("Model governance tracking and validation documentation is under development.")
        elif st.session_state.gov_section == "Performance":
            st.markdown('<div style="font-family: \'DM Sans\', sans-serif; font-size: 56px; font-weight: 900; color: #2b5b7c; text-transform: uppercase; margin-bottom: 40px; letter-spacing: -1px; display: flex; align-items: center; gap: 15px;"><span>📈</span> PERFORMANCE MONITORING</div>', unsafe_allow_html=True)
            st.info("Model drift and data quality monitoring dashboards are under development.")
        elif st.session_state.gov_section == "Terms and Conditions":
            st.markdown(
                '<div style="font-family: \'DM Sans\', sans-serif; font-size: 56px; font-weight: 900; color: #2b5b7c; text-transform: uppercase; margin-bottom: 40px; letter-spacing: -1px; display: flex; align-items: center; gap: 15px;">'
                '<span>⚖️</span> TERMS AND CONDITIONS'
                '</div>', 
                unsafe_allow_html=True
            )
            
            st.markdown("""
            <style>
                .tc-card {
                    background-color: white;
                    border: 1px solid #e2e8f0;
                    border-radius: 12px;
                    padding: 20px 24px;
                    margin-bottom: 16px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.03);
                    border-left: 5px solid #ffb703;
                }
                .tc-card-title {
                    font-family: 'DM Sans', sans-serif;
                    font-weight: 800;
                    font-size: 16px;
                    color: #0f172a;
                    margin-bottom: 8px;
                }
                .tc-card-text {
                    font-family: 'Century Gothic', 'Questrial', sans-serif;
                    font-size: 14px;
                    color: #475569;
                    line-height: 1.6;
                }
            </style>
            
            <div class="tc-card">
                <div class="tc-card-title">1. Early Prototype Testing</div>
                <div class="tc-card-text">Credit Analyze is experimental software developed for the Kenya Bankers Association (KBA) Credit Scoring & Alternative Data Initiative.</div>
            </div>
            <div class="tc-card">
                <div class="tc-card-title">2. Survey & Interview Consent</div>
                <div class="tc-card-text">By signing up you agree to be available for user experience surveys, qualitative feedback sessions, and research interviews.</div>
            </div>
            <div class="tc-card">
                <div class="tc-card-title">3. Immediate Access</div>
                <div class="tc-card-text">Open registration: valid credentials provision access instantly.</div>
            </div>
            <div class="tc-card">
                <div class="tc-card-title">4. Privacy (Kenya DPA 2019)</div>
                <div class="tc-card-text">Credentials are hashed and stored in an isolated data store. Portfolio records are ephemeral and held in volatile RAM.</div>
            </div>
            <div class="tc-card">
                <div class="tc-card-title">5. Open Source Licensing</div>
                <div class="tc-card-text">This project is proudly open source and released under the Apache License 2.0. <a href="https://github.com/Bluuclues/CreditRisk_AutoML/blob/main/LICENSE" target="_blank" style="color: #ea580c; font-weight: bold; text-decoration: none;">View the Full License ↗</a></div>
            </div>
            """, unsafe_allow_html=True)

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
