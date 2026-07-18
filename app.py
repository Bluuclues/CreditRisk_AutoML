import streamlit as st
import pandas as pd
import duckdb
import sqlite3
import os
import io
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import uuid
import streamlit.components.v1 as components
import io

# --- IMPORT STREAMLIT ELEMENTS ---
from streamlit_elements import elements, dashboard, mui, nivo

# Import your custom backend modules
from modules.feature_store import apply_macro_layers
from modules.models.dispatcher import run_models

# Keep wide layout
st.set_page_config(page_title="Credit Risk AutoML", layout="wide")

# --- CSS INJECTION ---
def load_local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_local_css("style.css")

# --- SESSION STATE INITIALIZATION ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'sqlite_conn' not in st.session_state:
    st.session_state.sqlite_conn = sqlite3.connect(':memory:', check_same_thread=False)
if 'duck_conn' not in st.session_state:
    st.session_state.duck_conn = duckdb.connect(':memory:')

# UI Routing & State Flags
if 'current_page' not in st.session_state:
    st.session_state.current_page = "pipeline"
if 'data_ingested' not in st.session_state:
    st.session_state.data_ingested = False
if 'primary_df' not in st.session_state:
    st.session_state.primary_df = None
if 'final_layered_df' not in st.session_state:
    st.session_state.final_layered_df = None
if 'show_ml_selection' not in st.session_state:
    st.session_state.show_ml_selection = False
if 'trained_results' not in st.session_state:
    st.session_state.trained_results = None

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALTERNATIVE_DATA_DIR = os.path.join(BASE_DIR, "Data", "Alternative_Data")

# --- COUNTRY CODE MAPPING ---
def load_country_mapping():
    db_path = os.path.join(ALTERNATIVE_DATA_DIR, 'countries.db')
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM country_mapping ORDER BY country_name", conn)
        conn.close()
        return dict(zip(df['country_name'], df['country_code']))
    else:
        return {"Kenya": "KEN", "Database Missing": "ERR"}

COUNTRY_MAPPING = load_country_mapping()


# ==========================================
# PAGE 1: THE ML PIPELINE
# ==========================================
if st.session_state.current_page == "pipeline":
    
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown('<div class="massive-title">CREDIT RISK<br>AUTOML</div>', unsafe_allow_html=True)
        st.write("") 
        st.markdown("""
        **Developed by Francis Nyangoma, Linda Kimilu, and Shirleen Chege.**
        
        This open-source Automated Machine Learning (AutoML) tool is designed to analyze your loan panel data. It automatically trains predictive models while allowing you to seamlessly layer on alternative variables of your choosing.
        """)

    with right_col:
        st.info("🔒 **Privacy Guarantee:** All data uploaded here is processed in ephemeral memory. No files are saved to a server.")

        # --- STEP 1: UPLOAD & SETUP ---
        if not st.session_state.data_ingested:
            st.subheader("1. Portfolio Setup")
            
            # Downloadable Guides
            with st.expander("📄 View Data Upload Guidelines & Templates"):
                st.markdown("""
                Your uploaded `.csv` file must contain these standard columns:
                * **borrower_id**: Unique identifier for the client.
                * **borrower_type**: Category (e.g., 'Retail', 'SME', 'Corporate').
                * **loan_no**: Unique loan identifier.
                * **loan_date**: Origination date (Used to map annual macro data).
                * **due_date**: Contractual maturity date.
                * **payoff_date**: Actual repayment date (leave blank if active).
                * **tenure_days**: Total duration in days.
                * **amount_kes**: Principal amount (numeric).
                * **default_flag**: Target Variable (1 = Defaulted, 0 = Performing).
                """)
                
                template_csv = "borrower_id,borrower_type,loan_no,loan_date,due_date,payoff_date,tenure_days,amount_kes,default_flag\n"
                st.download_button(
                    label="⬇️ Download Blank CSV Template",
                    data=template_csv,
                    file_name="credit_risk_template.csv",
                    mime="text/csv"
                )

            # Country Selector
            country_list = list(COUNTRY_MAPPING.keys())
            default_index = country_list.index("Kenya") if "Kenya" in country_list else 0
            
            selected_country_name = st.selectbox("Select the Country of the Loan Portfolio:", country_list, index=default_index)
            selected_country_code = COUNTRY_MAPPING[selected_country_name]

            uploaded_file = st.file_uploader("Upload Panel Loan Data (CSV)", type=["csv"])

            if uploaded_file is not None:
                st.session_state.primary_df = pd.read_csv(uploaded_file)
                
                if 'loan_date' in st.session_state.primary_df.columns:
                    st.session_state.primary_df['loan_date'] = pd.to_datetime(st.session_state.primary_df['loan_date'], errors='coerce', format='mixed', dayfirst=True)
                    st.session_state.primary_df['year'] = st.session_state.primary_df['loan_date'].dt.year
                
                st.session_state.primary_df['session_id'] = st.session_state.session_id
                st.session_state.primary_df['country_code'] = selected_country_code 
                
                st.success(f"Successfully loaded {len(st.session_state.primary_df)} rows securely into memory!")
                st.dataframe(st.session_state.primary_df.head(3))

                if st.button("Ingest into Memory Database", key="ingest_btn"):
                    st.session_state.primary_df.to_sql('loans_raw', st.session_state.sqlite_conn, if_exists='replace', index=False)
                    st.session_state.duck_conn.register('temp_df', st.session_state.primary_df)
                    st.session_state.duck_conn.execute("CREATE OR REPLACE TABLE ml_features AS SELECT * FROM temp_df")
                    
                    st.session_state.data_ingested = True
                    st.session_state.final_layered_df = st.session_state.primary_df.copy()
                    st.rerun()

        # --- STEP 2: ALTERNATIVE DATA LAYERING ---
        if st.session_state.data_ingested and not st.session_state.show_ml_selection:
            st.success("✅ Data securely stored in localized RAM!")
            st.write("---")
            st.subheader("💡 Layer Alternative Macroeconomic Data")

            available_files = [f for f in os.listdir(ALTERNATIVE_DATA_DIR) if f.endswith(('.db', '.csv'))] if os.path.exists(ALTERNATIVE_DATA_DIR) else []
            
            selected_layers = []
            if available_files:
                for file in available_files:
                    if st.checkbox(f"Layer with {file}", key=f"layer_{file}"):
                        selected_layers.append(file)
            else:
                st.warning(f"No alternative datasets found in `{ALTERNATIVE_DATA_DIR}`.")

            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Apply Selected Layering", type="primary"):
                    st.session_state.final_layered_df = apply_macro_layers(
                        st.session_state.duck_conn, 
                        selected_layers, 
                        ALTERNATIVE_DATA_DIR
                    )
                    st.success("Successfully layered macro features inside DuckDB!")
                    st.session_state.show_ml_selection = True
                    st.rerun()

            with col2:
                if st.button("Continue Without Layering"):
                    st.session_state.final_layered_df = st.session_state.duck_conn.execute("SELECT * FROM ml_features").df()
                    st.session_state.show_ml_selection = True
                    st.rerun()

        # --- STEP 3: ML MODELS SELECTION ---
        if st.session_state.show_ml_selection:
            st.info("📊 Final Feature Set Ready for Pipeline Training")
            st.dataframe(st.session_state.final_layered_df.head(3))
            
            st.write("---")
            st.header("🤖 Machine Learning Model Selection")
            
            models = st.multiselect(
                "Select algorithms to train:",
                ["XGBoost Classifier", "Random Forest"],
                default=["XGBoost Classifier"]
            )
            
            col_train, col_reset = st.columns(2)
            with col_train:
                if st.button("Run AutoML Pipeline", type="primary"):
                    with st.spinner("Training models..."):
                        st.session_state.trained_results = run_models(st.session_state.final_layered_df, models)
                    
                    st.success("Training Complete!")
                    st.session_state.current_page = "dashboard"
                    st.rerun()
                    
            with col_reset:
                if st.button("Reset Pipeline", type="secondary"):
                    st.session_state.data_ingested = False
                    st.session_state.show_ml_selection = False
                    st.session_state.trained_results = None
                    st.rerun()


# ==========================================
# PAGE 2: THE ANALYTICS DASHBOARD (REFACETORED)
# ==========================================
elif st.session_state.current_page == "dashboard":
    
    col_title, col_back = st.columns([2, 2])
    with col_title:
        st.title("📈 Portfolio & Model Insights")
    with col_back:
        col_b1, col_b2, col_b3 = st.columns([1,1,1])
        with col_b1:
            if st.button("← Pipeline"):
                st.session_state.current_page = "pipeline"
                st.rerun()
        with col_b2:
            if st.button("View Data"):
                st.session_state.current_page = "data_viewer"
                st.rerun()
        with col_b3:
            components.html(
                """
                <button onclick="window.parent.print()" style="padding:0.4rem 0.8rem; border:1px solid #ccc; border-radius:4px; background-color:#f8f9fa; cursor:pointer;">🖨️ Print</button>
                """,
                height=40
            )
            
    st.write("---")
    
    # --- DATA COMPUTATION ---
    df = st.session_state.final_layered_df
    total_loans = len(df)
    total_defaults = int(df['default_flag'].sum()) if 'default_flag' in df.columns else 0
    default_rate = (total_defaults / total_loans) * 100 if total_loans > 0 else 0
    
    # Prepare data structure for Nivo Pie Chart
    if 'default_flag' in df.columns:
        counts = df['default_flag'].value_counts()
        pie_data = [
            {"id": "Performing (0)", "label": "Performing", "value": int(counts.get(0, 0)), "color": "#2e7bcf"},
            {"id": "Defaulted (1)", "label": "Defaulted", "value": int(counts.get(1, 0)), "color": "#ff4b4b"}
        ]
    else:
        pie_data = []

    # --- 1. SELECT TO BUILD INTERFACE ---
    available_widgets = {
        "metrics": "High-Level Portfolio Metrics",
        "pie_chart": "Default Distribution Breakdown (Nivo Interactive)",
        "summary_table": "Summary Statistics (Traditional Grid)"
    }
    
    selected_widgets = st.multiselect(
        "🛠️ Customize Workspace Layout (Add/Remove Dashboard Elements):",
        options=list(available_widgets.keys()),
        format_func=lambda x: available_widgets[x],
        default=["metrics", "pie_chart"]
    )
    
    # --- 2. DEFINE THE DRAGGABLE CANVAS GRID LAYOUT ---
    # Layout parameters: Item(key, x, y, width, height)
    grid_layout = [
        dashboard.Item("metrics", 0, 0, 12, 2, isResizable=False),
        dashboard.Item("pie_chart", 0, 2, 4, 4),
        dashboard.Item("summary_table", 4, 2, 8, 4)
    ]
    
    st.caption("💡 **Workspace Tip:** Drag components by their top boundaries to reorganize your view. Grab the bottom-right corners to resize components.")
    
    # --- 3. RENDER CANVAS COMPONENT ENVIRONMENT ---
    if selected_widgets:
        with elements("dashboard_workspace"):
            with dashboard.Grid(grid_layout):
                
                # METRICS SECTION
                if "metrics" in selected_widgets:
                    with mui.Paper(key="metrics", elevation=2, sx={"padding": 2, "display": "flex", "justifyContent": "space-around", "alignItems": "center", "backgroundColor": "#1e293b", "color": "#ffffff"}):
                        with mui.Box(sx={"textAlign": "center"}):
                            mui.Typography("Total Loan Records", variant="overline", sx={"color": "#94a3b8"})
                            mui.Typography(f"{total_loans:,}", variant="h4", sx={"fontWeight": "bold"})
                        with mui.Box(sx={"textAlign": "center"}):
                            mui.Typography("Total Defaults Recorded", variant="overline", sx={"color": "#94a3b8"})
                            mui.Typography(f"{total_defaults:,}", variant="h4", sx={"fontWeight": "bold", "color": "#f87171"})
                        with mui.Box(sx={"textAlign": "center"}):
                            mui.Typography("Historical Default Rate", variant="overline", sx={"color": "#94a3b8"})
                            mui.Typography(f"{default_rate:.2f}%", variant="h4", sx={"fontWeight": "bold", "color": "#f87171"})

                # INTERACTIVE NIVO PIE CHART
                if "pie_chart" in selected_widgets:
                    with mui.Paper(key="pie_chart", elevation=3, sx={"padding": 3, "display": "flex", "flexDirection": "column", "height": "100%"}):
                        mui.Typography("Default Distribution", variant="h6", sx={"fontWeight": "600", "marginBottom": 1})
                        if pie_data:
                            nivo.Pie(
                                data=pie_data,
                                margin={"top": 40, "right": 40, "bottom": 40, "left": 40},
                                innerRadius=0.5,
                                padAngle=0.7,
                                cornerRadius=3,
                                activeOuterRadiusOffset=8,
                                borderWidth=1,
                                borderColor={"from": "color", "modifiers": [["darker", 0.2]]},
                                enableArcLinkLabels=True,
                                arcLinkLabelsSkipAngle=10,
                                arcLinkLabelsTextColor="#333333",
                                arcLabelsRadiusOffset=0.5,
                                arcLabelsSkipAngle=10,
                                arcLabelsTextColor="#ffffff"
                            )
                        else:
                            mui.Typography("No default_flag column found to plot.", variant="body2", color="error")

                # REPOSITIONABLE SUMMARY TABLE CARD
                if "summary_table" in selected_widgets:
                    with mui.Paper(key="summary_table", elevation=3, sx={"padding": 3, "overflow": "auto"}):
                        mui.Typography("Summary Statistics (Numeric Features)", variant="h6", sx={"fontWeight": "600", "marginBottom": 2})
                        
                        # We leverage a sub-container block to safely render native elements like DataFrames inside a static frame block
                        cols_to_exclude = ['session_id', 'borrower_id', 'loan_no', 'default_flag', 'year']
                        numeric_df = df.select_dtypes(include=['number']).drop(columns=[c for c in cols_to_exclude if c in df.columns], errors='ignore')
                        summary_stats = numeric_df.describe().T
                        
                        # Use streamlit's container escape tool since nested layout engine handles it safely outside grid rendering loops
                        st.dataframe(summary_stats, use_container_width=True)

    st.write("---")
    
    # --- 4. ADVANCED LOAN ANALYTICS ---
    st.subheader("📊 Advanced Loan Analytics")

    # Add filters for the advanced analytics
    with st.expander("🔍 Filter Analytics Data", expanded=False):
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            if 'borrower_type' in df.columns:
                b_types = ['All'] + list(df['borrower_type'].dropna().unique())
                selected_b_type = st.selectbox("Borrower Type", b_types)
            else:
                selected_b_type = 'All'
        with filter_col2:
            if 'year' in df.columns:
                years = ['All'] + list(df['year'].dropna().astype(int).unique())
                selected_year = st.selectbox("Year", years)
            else:
                selected_year = 'All'
        
        # Apply filters
        analytics_df = df.copy()
        if selected_b_type != 'All':
            analytics_df = analytics_df[analytics_df['borrower_type'] == selected_b_type]
        if selected_year != 'All':
            analytics_df = analytics_df[analytics_df['year'] == selected_year]
            
    def get_matplotlib_img(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', transparent=True)
        return buf.getvalue()
        
    def get_plotly_img(fig):
        try:
            return fig.to_image(format="png")
        except Exception:
            return None

    def render_chart(chart_type, analytics_df, cell_id):
        if chart_type == "None":
            st.write("")
        elif chart_type == "Time Series Analysis":
            if 'loan_date' in analytics_df.columns:
                temp_df = analytics_df.copy()
                temp_df['year_month'] = temp_df['loan_date'].dt.to_period('M').astype(str)
                time_df = temp_df.groupby('year_month').agg(
                    total_loans=('amount_kes', 'count'),
                    total_amount=('amount_kes', 'sum'),
                    defaults=('default_flag', 'sum')
                ).reset_index()
                time_df['default_rate'] = (time_df['defaults'] / time_df['total_loans']) * 100
                
                fig_time = px.line(time_df, x='year_month', y='total_amount', title="Loan Volume Over Time", markers=True)
                st.plotly_chart(fig_time, use_container_width=True, key=f"fig_time_{cell_id}")
                st.download_button("📥 Download", data=get_plotly_img(fig_time), file_name=f"volume_{cell_id}.png", mime="image/png", key=f"dl_time_{cell_id}")
                
                fig_rate = px.bar(time_df, x='year_month', y='default_rate', title="Default Rate Over Time", color='default_rate', color_continuous_scale='Reds')
                st.plotly_chart(fig_rate, use_container_width=True, key=f"fig_rate_{cell_id}")
                st.download_button("📥 Download", data=get_plotly_img(fig_rate), file_name=f"rate_{cell_id}.png", mime="image/png", key=f"dl_rate_{cell_id}")
            else:
                st.warning("No 'loan_date' column available.")
        elif chart_type == "Distribution & Composition":
            if 'amount_kes' in analytics_df.columns and 'default_flag' in analytics_df.columns:
                fig_box, ax_box = plt.subplots(figsize=(5, 3))
                sns.boxplot(data=analytics_df, x='default_flag', y='amount_kes', ax=ax_box, palette="Set2")
                ax_box.set_title("Amount by Default")
                fig_box.patch.set_alpha(0.0)
                st.pyplot(fig_box)
                st.download_button("📥 Download", data=get_matplotlib_img(fig_box), file_name=f"amount_{cell_id}.png", mime="image/png", key=f"dl_amount_{cell_id}")
            if 'tenure_days' in analytics_df.columns and 'default_flag' in analytics_df.columns:
                fig_kde, ax_kde = plt.subplots(figsize=(5, 3))
                sns.kdeplot(data=analytics_df, x='tenure_days', hue='default_flag', fill=True, common_norm=False, palette="Set1", ax=ax_kde)
                ax_kde.set_title("Tenure by Default")
                fig_kde.patch.set_alpha(0.0)
                st.pyplot(fig_kde)
                st.download_button("📥 Download", data=get_matplotlib_img(fig_kde), file_name=f"tenure_{cell_id}.png", mime="image/png", key=f"dl_tenure_{cell_id}")
        elif chart_type == "Categorical Breakdown":
            if 'borrower_type' in analytics_df.columns and 'default_flag' in analytics_df.columns:
                bt_stats = analytics_df.groupby('borrower_type')['default_flag'].mean().reset_index()
                bt_stats['default_rate'] = bt_stats['default_flag'] * 100
                fig_bt = px.bar(bt_stats, x='borrower_type', y='default_rate', title="Default Rate by Borrower Type", color='borrower_type')
                st.plotly_chart(fig_bt, use_container_width=True, key=f"fig_bt_{cell_id}")
                st.download_button("📥 Download", data=get_plotly_img(fig_bt), file_name=f"categorical_{cell_id}.png", mime="image/png", key=f"dl_cat_{cell_id}")
        elif chart_type == "Correlation Heatmap":
            numeric_cols = analytics_df.select_dtypes(include=['number']).columns.tolist()
            cols_to_exclude = ['session_id', 'borrower_id', 'loan_no', 'year']
            numeric_cols = [c for c in numeric_cols if c not in cols_to_exclude]
            if len(numeric_cols) > 1:
                corr_matrix = analytics_df[numeric_cols].corr()
                fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax_corr)
                ax_corr.set_title("Correlation Heatmap")
                fig_corr.patch.set_alpha(0.0)
                st.pyplot(fig_corr)
                st.download_button("📥 Download", data=get_matplotlib_img(fig_corr), file_name=f"corr_{cell_id}.png", mime="image/png", key=f"dl_corr_{cell_id}")
        elif chart_type == "Macroeconomic Impact":
            standard_cols = ['borrower_id', 'borrower_type', 'loan_no', 'loan_date', 'due_date', 'payoff_date', 'tenure_days', 'amount_kes', 'default_flag', 'session_id', 'country_code', 'year', 'year_month']
            macro_cols = [c for c in analytics_df.columns if c not in standard_cols and pd.api.types.is_numeric_dtype(analytics_df[c])]
            if len(macro_cols) > 0:
                selected_macro = st.selectbox("Select Macro Variable:", macro_cols, key=f"macro_sel_{cell_id}")
                if selected_macro:
                    fig_macro_scatter = px.box(analytics_df, x='default_flag', y=selected_macro, color='default_flag', title=f"{selected_macro} by Default")
                    st.plotly_chart(fig_macro_scatter, use_container_width=True, key=f"fig_macro_{cell_id}")
                    st.download_button("📥 Download", data=get_plotly_img(fig_macro_scatter), file_name=f"macro_{cell_id}.png", mime="image/png", key=f"dl_mac_{cell_id}")
            else:
                st.warning("No external macro variables detected.")

    st.write("### 🎛️ Customizable Dashboard Grid")
    chart_options = ["None", "Time Series Analysis", "Distribution & Composition", "Categorical Breakdown", "Correlation Heatmap", "Macroeconomic Impact"]
    
    # ROW 1 (3 Cells)
    row1_cols = st.columns(3)
    with row1_cols[0]:
        with st.container(border=True):
            cell1_choice = st.selectbox("Widget 1", chart_options, index=1, key="c1")
            render_chart(cell1_choice, analytics_df, "1")
    with row1_cols[1]:
        with st.container(border=True):
            cell2_choice = st.selectbox("Widget 2", chart_options, index=2, key="c2")
            render_chart(cell2_choice, analytics_df, "2")
    with row1_cols[2]:
        with st.container(border=True):
            cell3_choice = st.selectbox("Widget 3", chart_options, index=3, key="c3")
            render_chart(cell3_choice, analytics_df, "3")
            
    st.write("---")
    
    # ROW 2 (2 Cells)
    row2_cols = st.columns(2)
    with row2_cols[0]:
        with st.container(border=True):
            cell4_choice = st.selectbox("Widget 4", chart_options, index=4, key="c4")
            render_chart(cell4_choice, analytics_df, "4")
    with row2_cols[1]:
        with st.container(border=True):
            cell5_choice = st.selectbox("Widget 5", chart_options, index=5, key="c5")
            render_chart(cell5_choice, analytics_df, "5")

    st.write("---")
    
    # --- 5. MACHINE LEARNING INSIGHTS ---
    # Kept outside the grid execution stream for specialized Matplotlib SHAP compilation structures
    if st.session_state.trained_results:
        for model_name, results in st.session_state.trained_results.items():
            st.subheader(f"🤖 Model: {model_name}")
            st.markdown(f"**Accuracy:** {results['metrics']['accuracy']:.2%} | **Precision:** {results['metrics']['precision']:.2%} | **Recall:** {results['metrics']['recall']:.2%}")
            st.write("**Feature Importance (SHAP Values)**")
            
            try:
                model = results['model'].named_steps['classifier']
                preprocessor = results['model'].named_steps['preprocessor']
                
                X_test_transformed = preprocessor.transform(results['X_test'])
                if hasattr(X_test_transformed, 'toarray'):
                    X_test_transformed = X_test_transformed.toarray()
                
                feature_names = preprocessor.get_feature_names_out()
                X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_df)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                fig_shap, ax_shap = plt.subplots(figsize=(7, 4))
                shap.summary_plot(shap_values, X_test_df, max_display=10, show=False)
                
                fig_shap.patch.set_alpha(0.0)
                ax_shap.patch.set_alpha(0.0)
                st.pyplot(fig_shap)
                
            except Exception as e:
                st.warning(f"Could not generate SHAP values for {model_name}. Error: {e}")

            st.write("---")
            st.subheader("💼 Credit Analyst Metrics (PD & LGD)")
            
            # Use original dataframe to recover loan IDs and amounts for the test set
            try:
                original_indices = results['X_test'].index
                test_loans = df.loc[original_indices].copy()
                
                # Assign Probability of Default
                test_loans['Probability of Default (PD)'] = results['y_prob']
                
                # User-defined LGD since we do not have an explicit recovery cash flow column
                lgd_val = st.slider(f"Set Baseline Loss Given Default (LGD) for {model_name} (%)", min_value=10, max_value=100, value=45, step=5) / 100.0
                test_loans['Loss Given Default (LGD)'] = lgd_val
                
                if 'amount_kes' in test_loans.columns:
                    test_loans['Exposure at Default (EAD)'] = test_loans['amount_kes']
                    test_loans['Expected Loss (EL)'] = test_loans['Probability of Default (PD)'] * test_loans['Loss Given Default (LGD)'] * test_loans['Exposure at Default (EAD)']
                    
                    total_ead = test_loans['Exposure at Default (EAD)'].sum()
                    total_el = test_loans['Expected Loss (EL)'].sum()
                    portfolio_el_rate = (total_el / total_ead) * 100 if total_ead > 0 else 0
                    
                    st.info(f"**Aggregate Portfolio Metrics (Test Set):** Total EAD: {total_ead:,.2f} KES | Total Expected Loss: {total_el:,.2f} KES | Portfolio EL Rate: {portfolio_el_rate:.2f}%")
                
                # Display output table
                display_cols = ['borrower_id', 'loan_no', 'amount_kes', 'Probability of Default (PD)', 'Loss Given Default (LGD)', 'Expected Loss (EL)']
                display_cols = [c for c in display_cols if c in test_loans.columns]
                
                st.dataframe(test_loans[display_cols].head(50), use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not calculate Credit Metrics for {model_name}. Error: {e}")

            st.write("---")


# ==========================================
# PAGE 3: DATA VIEWER & EXPORT
# ==========================================
elif st.session_state.current_page == "data_viewer":
    st.title("🗂️ Data Viewer & Export")
    
    if st.button("← Back to Dashboard"):
        st.session_state.current_page = "dashboard"
        st.rerun()
        
    st.write("---")
    
    if st.session_state.final_layered_df is not None:
        st.write("### Transformed Data Preview")
        st.dataframe(st.session_state.final_layered_df.head(100))
        
        col_csv, col_excel = st.columns(2)
        
        with col_csv:
            csv_data = st.session_state.final_layered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name="layered_loan_data.csv",
                mime="text/csv"
            )
            
        with col_excel:
            # Generate Excel using openpyxl in memory
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.final_layered_df.to_excel(writer, index=False, sheet_name='Data')
            
            st.download_button(
                label="📥 Download as Excel",
                data=buffer.getvalue(),
                file_name="layered_loan_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("No data available. Please ingest data in the Pipeline first.")