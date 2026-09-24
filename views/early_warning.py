import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

def render_early_warning():
    st.markdown('<div style="font-family: \'DM Sans\', sans-serif; font-size: 56px; font-weight: 900; color: #2b5b7c; text-transform: uppercase; margin-bottom: 40px; letter-spacing: -1px; display: flex; align-items: center; gap: 15px;"><span>🚨</span> EARLY WARNING SYSTEM</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This section highlights facilities demonstrating early warning signals for potential default. 
    The risk scores are computed using alternative data triggers (e.g., missed mobile money payments, drop in transactional frequency).
    """)

    # Mock Data Generation
    @st.cache_data
    def get_mock_data():
        np.random.seed(42)
        facilities = [f"FAC-{np.random.randint(10000, 99999)}" for _ in range(50)]
        segments = np.random.choice(["SME", "Retail", "Corporate"], 50)
        risk_scores = np.random.randint(60, 100, 50) # High risk scores
        days_past_due = np.random.randint(15, 90, 50)
        exposure = np.random.randint(10000, 500000, 50)
        
        df = pd.DataFrame({
            "Facility ID": facilities,
            "Segment": segments,
            "Risk Score": risk_scores,
            "Days Past Due": days_past_due,
            "Total Exposure ($)": exposure
        })
        return df.sort_values("Risk Score", ascending=False).reset_index(drop=True)

    df_risk = get_mock_data()

    # Layout for KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric(label="Total Critical Facilities", value="50", delta="+5 since last week", delta_color="inverse")
    with kpi2:
        st.metric(label="Exposure at Risk", value=f"${df_risk['Total Exposure ($)'].sum():,.2f}", delta="+12%", delta_color="inverse")
    with kpi3:
        st.metric(label="Avg Risk Score", value=f"{df_risk['Risk Score'].mean():.1f}/100")
        
    st.markdown("---")
    
    st.subheader("High-Risk Facilities List")
    
    # Define color styling for risk scores
    def color_risk(val):
        color = '#ef4444' if val >= 90 else '#f97316' if val >= 75 else '#eab308'
        return f'background-color: {color}; color: white; font-weight: bold;'
    
    styled_df = df_risk.style.map(color_risk, subset=['Risk Score'])
    
    st.dataframe(
        styled_df, 
        use_container_width=True, 
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="ew_table_selection"
    )
    
    # Download Button
    csv = df_risk.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download High-Risk List (CSV)",
        data=csv,
        file_name='early_warning_list.csv',
        mime='text/csv',
    )
    
    # Deep Dive Logic (Dialog)
    selected_rows = st.session_state.ew_table_selection.get("selection", {}).get("rows", [])
    if selected_rows:
        row_idx = selected_rows[0]
        selected_facility = df_risk.iloc[row_idx]
        
        @st.dialog(f"🔍 Deep Dive: {selected_facility['Facility ID']}")
        def facility_deep_dive(facility):
            st.write(f"**Segment:** {facility['Segment']}")
            st.write(f"**Risk Score:** {facility['Risk Score']}/100")
            st.write(f"**Days Past Due:** {facility['Days Past Due']} days")
            st.write(f"**Exposure:** ${facility['Total Exposure ($)']:,.2f}")
            
            st.markdown("### ⚠️ Trigger Events")
            events = pd.DataFrame({
                "Date": pd.date_range(end=pd.Timestamp.today(), periods=3, freq='-7D').strftime('%Y-%m-%d'),
                "Event": [
                    "30% drop in MoMo transaction frequency",
                    "Missed utility payment detected",
                    "Unusual large withdrawal flagged"
                ]
            })
            st.table(events)
            
            st.markdown("### 📈 Recent Behavior Trend")
            trend_data = pd.DataFrame({
                "Month": ["M-3", "M-2", "M-1", "Current"],
                "Utilization %": [45, 60, 85, 95]
            })
            fig = px.line(trend_data, x="Month", y="Utilization %", markers=True)
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        facility_deep_dive(selected_facility)
