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

# Import custom backend modules
from modules.data_validator import CreditRiskDataValidator
from modules.feature_store import apply_macro_layers, export_parquet_snapshot, export_parquet_bytes, export_csv_bytes
from modules.models.dispatcher import run_automl_pipeline
from modules.models.shap_explainer import CreditRiskExplainer

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
# SECTION 1: HEADER & PORTFOLIO SETUP (TOP CANVAS)
# ==============================================================================
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="massive-title">CREDIT RISK<br>AUTOML</div>', unsafe_allow_html=True)
    st.markdown("""
    **Kenya Bankers Association (KBA) Credit Scoring & Alternative Data Initiative**  
    *Developed by Francis Nyangoma, Linda Kimilu, and Shirleen Chege.*
    
    This end-to-end Automated Machine Learning (AutoML) platform evaluates thin-file and informal sector borrowers by **fusion of panel repayment records with multi-source alternative data**.
    """)

with right_col:
    st.info("🔒 **Privacy Guarantee:** Customer data is processed in ephemeral RAM. Non-reversible SHA-256 pseudonymization protects individual borrower identities (Kenya DPA 2019).")

    with st.expander("📄 View Data Upload Guidelines & Template"):
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
        st.subheader("1. Portfolio Setup & Ingestion")
        
        country_list = list(COUNTRY_MAPPING.keys())
        default_idx = country_list.index("Kenya") if "Kenya" in country_list else 0
        selected_country_name = st.selectbox("Country Jurisdiction:", country_list, index=default_idx)
        selected_country_code = COUNTRY_MAPPING[selected_country_name]

        uploaded_file = st.file_uploader("Upload Panel Loan Data (.CSV)", type=["csv"])

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

                if st.button("⚡ Ingest into DuckDB Memory Store", type="primary", key="ingest_btn"):
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
            if st.button("🔄 Reset Portfolio & Upload New CSV", use_container_width=True):
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
    st.subheader("💡 2. Layer Alternative Data Streams")

    available_files = [f for f in os.listdir(ALTERNATIVE_DATA_DIR) if f.endswith(('.db', '.csv'))] if os.path.exists(ALTERNATIVE_DATA_DIR) else []
    
    col_layer_left, col_layer_right = st.columns([2, 1])

    with col_layer_left:
        selected_layers = []
        if available_files:
            st.write("Select local alternative data feeds to join in RAM:")
            for f in available_files:
                if st.checkbox(f"🌿 Join `{f}` (Macro GCP & County Indicators)", value=(f == 'macro_layer.db'), key=f"chk_{f}"):
                    selected_layers.append(f)
        else:
            st.warning(f"No alternative data fixtures found in `{ALTERNATIVE_DATA_DIR}`.")

    with col_layer_right:
        st.caption("Layering Action:")
        if st.button("⚡ Execute Vectorized Join in DuckDB", type="primary"):
            layered_df = apply_macro_layers(
                st.session_state.duck_conn,
                selected_layers,
                ALTERNATIVE_DATA_DIR
            )
            st.session_state.final_layered_df = layered_df
            st.session_state.layers_applied = True
            st.success(f"Successfully merged features! Matrix shape: {layered_df.shape[0]:,} rows × {layered_df.shape[1]} columns.")
            st.rerun()

        if st.button("Continue Without Layering"):
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
# SECTION 3: TABFM & PYCARET AUTOML ENGINE & DISPATCHER
# ==============================================================================
if st.session_state.layers_applied:
    st.subheader("🤖 3. TabFM & PyCaret AutoML Engine")

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

    with col_cfg1:
        st.markdown("**Task & Optimization Metric**")
        st.caption("Target Detected: `default_flag` (Binary Default Classification)")
        optimize_metric = st.selectbox("Optimization Metric:", ["PR-AUC", "ROC-AUC", "F1", "Accuracy"], index=0)

    with col_cfg2:
        st.markdown("**Hyperparameter Tuning & Ensembling**")
        tune_toggle = st.checkbox("Enable Automated Optuna Hyperparameter Tuning", value=True)
        ensemble_toggle = st.checkbox("Construct Soft-Voting GBDT & TabFM Ensemble", value=True)

    with col_cfg3:
        st.markdown("**Engine Execution**")
        if st.button("🚀 Run TabFM & AutoML Pipeline", type="primary", use_container_width=True):
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

    st.markdown("## 📈 4. Onset Default Risk Screening & Live Analytics")

    # --- 4.1 EXECUTIVE KPI ROW ---
    total_records = len(df)
    hist_default_rate = (df['default_flag'].sum() / total_records * 100.0) if 'default_flag' in df and total_records > 0 else 0.0
    mean_pred_pd = np.mean(probs) * 100.0
    high_risk_count = int(np.sum(probs >= 0.60))

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Evaluated Portfolio Volume</div>
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
            <div class="kpi-title">Predicted Portfolio Mean PD</div>
            <div class="kpi-value" style="color: #d97706;">{mean_pred_pd:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">High Risk Borrowers (PD > 60%)</div>
            <div class="kpi-value" style="color: #dc2626;">{high_risk_count:,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # --- 4.2 THE ONSET DEFAULT DECISION TABLE ---
    st.subheader("📋 Onset Default Risk Decision Table")

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
            badge_html = f'<span class="badge-high-risk">🔴 HIGH {pd_pct:.1f}%</span>'
        elif pd_pct >= 30.0:
            tier = "🟡 Medium Risk (30% ≤ PD < 60%)"
            badge_html = f'<span class="badge-med-risk">🟡 MED {pd_pct:.1f}%</span>'
        else:
            tier = "🟢 Low Risk (PD < 30%)"
            badge_html = f'<span class="badge-low-risk">🟢 LOW {pd_pct:.1f}%</span>'

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

        # --- 4.3 BORROWER RISK DATA CARD DRAWER ---
        st.write("")
        st.markdown(f"### 🎴 Borrower Risk Data Card: `{selected_borrower_id}`")

        card_row = df.iloc[sel_idx]
        card_pd = probs[sel_idx] * 100.0
        card_amt = card_row.get("amount", 50000)
        card_tenure = card_row.get("tenure_days", 30)
        card_county = card_row.get("county_code", "016 (Kiambu)")

        rec_limit = max(10000, int(card_amt * (1.0 - (card_pd / 100.0))))

        col_card1, col_card2 = st.columns([1, 1])

        with col_card1:
            st.markdown(f"""
            <div class="borrower-card">
                <div class="card-header-title">👤 BORROWER PROFILE & DECISION</div>
                <p><b>Borrower ID:</b> {selected_borrower_id} &nbsp;|&nbsp; <b>Jurisdiction:</b> Kenya ({card_county})</p>
                <p><b>Requested Loan Principal:</b> KES {card_amt:,.0f} &nbsp;|&nbsp; <b>Tenor:</b> {card_tenure} Days</p>
                <div style="margin-top: 12px; margin-bottom: 12px;">
                    <b>Predicted Default Probability (PD):</b> <span style="font-size: 20px; font-weight: 800; color: #dc2626;">{card_pd:.1f}%</span>
                    <br><b>Recommended Max Credit Limit:</b> <span style="font-size: 16px; font-weight: 700; color: #16a34a;">KES {rec_limit:,.0f}</span>
                </div>
                <div class="card-section-label">📋 TRADITIONAL PANEL LOAN HISTORY</div>
                <ul>
                    <li><b>Lifetime Total Borrowed:</b> KES {card_row.get('feat_loan_amount_borrowed_sum_lifetime', card_amt*2):,.0f}</li>
                    <li><b>Lifetime Historical Defaults:</b> {card_row.get('feat_loan_prior_default_count_lifetime', 0)} times</li>
                    <li><b>Bureau CRB Score:</b> {card_row.get('feat_bureau_crb_score', 420)}</li>
                </ul>
                <div class="card-section-label">🌿 ALTERNATIVE DATA SIGNALS</div>
                <ul>
                    <li><b>M-Pesa 30d/90d Inflow Velocity:</b> {card_row.get('feat_mpesa_velocity_30d_over_90d', 0.85)}</li>
                    <li><b>Fuliza Overdraft Utilization:</b> {card_row.get('feat_fuliza_util_pct_30d', 0.45)*100:.0f}%</li>
                    <li><b>Regional Crop NDVI Anomaly:</b> {card_row.get('feat_gis_ndvi_anomaly_30d', -0.05)}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_card2:
            st.markdown("**🔍 TreeSHAP Waterfall Explanation Plot**")
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
            st.markdown("#### 🏆 Comparative Model Leaderboard")
            if not leaderboard_df.empty:
                st.dataframe(leaderboard_df, use_container_width=True)
            else:
                st.write("Leaderboard data unavailable.")

        with col_lead2:
            st.markdown("#### 🌟 Portfolio-Wide Feature Importance (SHAP)")
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
                    
                    # Download static high-res SHAP bar plot
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

    # Business Analyst Quick Plug-and-Play Exports
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
            if engine is not None and hasattr(engine, "export_pipeline_bytes"):
                try:
                    pkl_bytes = engine.export_pipeline_bytes()
                except Exception:
                    import pickle
                    pkl_bytes = pickle.dumps(champion_model)
            else:
                import pickle
                pkl_bytes = pickle.dumps(champion_model)

            st.download_button(
                label="💾 Download Champion Pipeline (.pkl)",
                data=pkl_bytes,
                file_name="champion_pipeline.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )

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