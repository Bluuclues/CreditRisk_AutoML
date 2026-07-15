import streamlit as st
import sqlite3
import duckdb
import pandas as pd
import uuid
import os
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

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
@st.cache_data
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
                    st.session_state.primary_df['loan_date'] = pd.to_datetime(st.session_state.primary_df['loan_date'], errors='coerce')
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
    
    col_title, col_back = st.columns([3, 1])
    with col_title:
        st.title("📈 Portfolio & Model Insights")
    with col_back:
        if st.button("← Back to Pipeline"):
            st.session_state.current_page = "pipeline"
            st.rerun()
            
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
            
    # Tabs for different statistics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Time Series Analysis", 
        "📦 Distribution & Composition", 
        "📊 Categorical Breakdown", 
        "🔥 Correlation & Relationships",
        "🌍 Macroeconomic Impact"
    ])
    
    with tab1:
        st.markdown("### Loan Volume & Defaults Over Time")
        if 'loan_date' in analytics_df.columns:
            # Group by year-month
            analytics_df['year_month'] = analytics_df['loan_date'].dt.to_period('M').astype(str)
            time_df = analytics_df.groupby('year_month').agg(
                total_loans=('amount_kes', 'count'),
                total_amount=('amount_kes', 'sum'),
                defaults=('default_flag', 'sum')
            ).reset_index()
            
            time_df['default_rate'] = (time_df['defaults'] / time_df['total_loans']) * 100
            
            fig_time = px.line(time_df, x='year_month', y='total_amount', 
                               title="Total Loan Amount Over Time", 
                               markers=True,
                               labels={"year_month": "Month", "total_amount": "Total Amount (KES)"})
            st.plotly_chart(fig_time, use_container_width=True)
            
            fig_rate = px.bar(time_df, x='year_month', y='default_rate', 
                              title="Default Rate Over Time (%)",
                              labels={"year_month": "Month", "default_rate": "Default Rate (%)"},
                              color='default_rate', color_continuous_scale='Reds')
            st.plotly_chart(fig_rate, use_container_width=True)
        else:
            st.warning("No 'loan_date' column available for Time Series analysis.")
            
    with tab2:
        st.markdown("### Loan Distribution by Default Status")
        col2a, col2b = st.columns(2)
        
        with col2a:
            st.write("**Loan Amount Distribution (Seaborn)**")
            if 'amount_kes' in analytics_df.columns and 'default_flag' in analytics_df.columns:
                fig_box, ax_box = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=analytics_df, x='default_flag', y='amount_kes', ax=ax_box, palette="Set2")
                ax_box.set_title("Loan Amount by Default Status")
                ax_box.set_xlabel("Default Status (0 = Performing, 1 = Defaulted)")
                ax_box.set_ylabel("Loan Amount (KES)")
                fig_box.patch.set_alpha(0.0)
                ax_box.patch.set_alpha(0.0)
                st.pyplot(fig_box)
            else:
                st.info("Missing required columns for this plot.")
                
        with col2b:
            st.write("**Tenure Distribution (Seaborn KDE)**")
            if 'tenure_days' in analytics_df.columns and 'default_flag' in analytics_df.columns:
                fig_kde, ax_kde = plt.subplots(figsize=(6, 4))
                sns.kdeplot(data=analytics_df, x='tenure_days', hue='default_flag', fill=True, common_norm=False, palette="Set1", ax=ax_kde)
                ax_kde.set_title("Loan Tenure Density by Default Status")
                ax_kde.set_xlabel("Tenure (Days)")
                fig_kde.patch.set_alpha(0.0)
                ax_kde.patch.set_alpha(0.0)
                st.pyplot(fig_kde)
            else:
                st.info("Missing required columns for this plot.")

    with tab3:
        st.markdown("### Default Rate by Borrower Type")
        if 'borrower_type' in analytics_df.columns and 'default_flag' in analytics_df.columns:
            # Calculate default rate by borrower type
            bt_stats = analytics_df.groupby('borrower_type')['default_flag'].mean().reset_index()
            bt_stats['default_rate'] = bt_stats['default_flag'] * 100
            
            fig_bt = px.bar(bt_stats, x='borrower_type', y='default_rate', 
                            title="Default Rate by Borrower Type (%)",
                            color='borrower_type',
                            labels={"borrower_type": "Borrower Type", "default_rate": "Default Rate (%)"})
            st.plotly_chart(fig_bt, use_container_width=True)
            
            # Additional seaborn plot for count
            st.write("**Loan Count by Borrower Type and Default Status (Seaborn)**")
            fig_count, ax_count = plt.subplots(figsize=(8, 4))
            sns.countplot(data=analytics_df, x='borrower_type', hue='default_flag', palette="viridis", ax=ax_count)
            ax_count.set_title("Number of Loans by Borrower Type")
            ax_count.set_xlabel("Borrower Type")
            ax_count.set_ylabel("Count")
            fig_count.patch.set_alpha(0.0)
            ax_count.patch.set_alpha(0.0)
            st.pyplot(fig_count)
        else:
            st.warning("Missing 'borrower_type' or 'default_flag' for this analysis.")

    with tab4:
        st.markdown("### Numeric Features Correlation & Relationships")
        
        # Select numeric columns
        numeric_cols = analytics_df.select_dtypes(include=['number']).columns.tolist()
        cols_to_exclude = ['session_id', 'borrower_id', 'loan_no', 'year']
        numeric_cols = [c for c in numeric_cols if c not in cols_to_exclude]
        
        if len(numeric_cols) > 1:
            st.write("**Correlation Heatmap (Seaborn)**")
            corr_matrix = analytics_df[numeric_cols].corr()
            
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax_corr)
            ax_corr.set_title("Correlation Heatmap")
            fig_corr.patch.set_alpha(0.0)
            ax_corr.patch.set_alpha(0.0)
            st.pyplot(fig_corr)
            
            # Pairplot
            if st.checkbox("Show Pairplot (may be slow for large datasets)"):
                st.write("**Pairwise Relationships (Seaborn)**")
                # Limiting to a few key columns to prevent massive memory/time usage
                pair_cols = [c for c in ['amount_kes', 'tenure_days', 'default_flag'] if c in numeric_cols]
                if len(pair_cols) > 1:
                    fig_pair = sns.pairplot(analytics_df[pair_cols].dropna(), hue='default_flag' if 'default_flag' in pair_cols else None, palette="husl")
                    fig_pair.fig.patch.set_alpha(0.0)
                    st.pyplot(fig_pair.fig)
        else:
            st.warning("Not enough numeric columns to compute correlations.")

    with tab5:
        st.markdown("### Macroeconomic Factors vs Loan Performance")
        
        # Identify macro columns (those that aren't the standard initial columns)
        standard_cols = ['borrower_id', 'borrower_type', 'loan_no', 'loan_date', 'due_date', 
                         'payoff_date', 'tenure_days', 'amount_kes', 'default_flag', 'session_id', 
                         'country_code', 'year', 'year_month']
        macro_cols = [c for c in analytics_df.columns if c not in standard_cols and pd.api.types.is_numeric_dtype(analytics_df[c])]
        
        if len(macro_cols) > 0:
            st.info(f"Detected {len(macro_cols)} layered macro/alternative variables.")
            
            col_macro1, col_macro2 = st.columns([1, 2])
            with col_macro1:
                selected_macro = st.selectbox("Select Macro Variable to Analyze:", macro_cols)
            
            with col_macro2:
                # Plot the selected macro variable distribution and its relation to default
                if selected_macro:
                    st.write(f"**Impact of {selected_macro} on Default (Plotly)**")
                    fig_macro_scatter = px.box(analytics_df, x='default_flag', y=selected_macro,
                                               color='default_flag',
                                               labels={'default_flag': 'Default Status (0=Performing, 1=Defaulted)', selected_macro: selected_macro},
                                               title=f"{selected_macro} Distribution by Default Status")
                    st.plotly_chart(fig_macro_scatter, use_container_width=True)
                    
            if 'loan_date' in analytics_df.columns:
                st.write(f"**{selected_macro} Over Time (Plotly)**")
                # group by year_month if available, else year
                if 'year_month' not in analytics_df.columns:
                    analytics_df['year_month'] = analytics_df['loan_date'].dt.to_period('M').astype(str)
                macro_time_df = analytics_df.groupby('year_month').agg({selected_macro: 'mean', 'default_flag': 'mean'}).reset_index()
                macro_time_df['default_rate'] = macro_time_df['default_flag'] * 100
                
                # Create a Plotly chart with secondary y-axis
                from plotly.subplots import make_subplots
                fig_macro_time = make_subplots(specs=[[{"secondary_y": True}]])
                fig_macro_time.add_trace(go.Scatter(x=macro_time_df['year_month'], y=macro_time_df[selected_macro], name=selected_macro, line=dict(color='blue')), secondary_y=False)
                fig_macro_time.add_trace(go.Scatter(x=macro_time_df['year_month'], y=macro_time_df['default_rate'], name='Default Rate (%)', line=dict(color='red', dash='dot')), secondary_y=True)
                
                fig_macro_time.update_layout(title_text=f"{selected_macro} and Default Rate Over Time")
                fig_macro_time.update_xaxes(title_text="Month")
                fig_macro_time.update_yaxes(title_text=selected_macro, secondary_y=False)
                fig_macro_time.update_yaxes(title_text="Default Rate (%)", secondary_y=True)
                
                st.plotly_chart(fig_macro_time, use_container_width=True)

        else:
            st.warning("No external macro/alternative numerical variables detected. Please ensure you layered alternative data in the pipeline step.")

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