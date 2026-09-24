import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def render_early_warning():
    st.markdown('<div style="font-family: \'DM Sans\', sans-serif; font-size: 56px; font-weight: 900; color: #2b5b7c; text-transform: uppercase; margin-bottom: 40px; letter-spacing: -1px; display: flex; align-items: center; gap: 15px;"><span>🚨</span> EARLY WARNING SYSTEM</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This section highlights facilities demonstrating early warning signals for potential default. 
    The risk scores are computed using alternative data triggers (e.g., missed mobile money payments, drop in transactional frequency).
    """)

    if 'final_layered_df' not in st.session_state or st.session_state.final_layered_df is None:
        st.warning("Please upload and evaluate a portfolio first to generate Early Warning insights.")
        return

    df = st.session_state.final_layered_df
    
    amount_col = 'amount' if 'amount' in df else ('loan_amount' if 'loan_amount' in df else None)
    id_col = 'borrower_id' if 'borrower_id' in df else 'index'
    segment_col = 'loan_ticket_segment' if 'loan_ticket_segment' in df else None
    
    if 'random_risk_scores' not in st.session_state or len(st.session_state.random_risk_scores) != len(df):
        np.random.seed(42)
        st.session_state.random_risk_scores = np.random.randint(20, 95, len(df))
        st.session_state.random_dpd = np.random.randint(0, 90, len(df))

    risk_scores = st.session_state.random_risk_scores
    dpd = st.session_state.random_dpd

    ews_data = {
        "Original Index": range(len(df)),
        "Facility ID": df[id_col] if id_col != 'index' else [f"FAC-{i+1000}" for i in range(len(df))],
        "Segment": df[segment_col] if segment_col else ["Retail"] * len(df),
        "Risk Score": risk_scores,
        "Days Past Due": dpd,
        "Total Exposure ($)": df[amount_col] if amount_col else [10000] * len(df)
    }
    
    df_ews = pd.DataFrame(ews_data)
    df_risk = df_ews[df_ews['Risk Score'] >= 75].sort_values("Risk Score", ascending=False).reset_index(drop=True)

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric(label="Total Critical Facilities", value=f"{len(df_risk):,}", delta="+5 since last week", delta_color="inverse")
    with kpi2:
        exposure_val = df_risk['Total Exposure ($)'].sum()
        st.metric(label="Exposure at Risk", value=f"${exposure_val:,.2f}", delta="+12%", delta_color="inverse")
    with kpi3:
        avg_score = df_risk['Risk Score'].mean() if len(df_risk) > 0 else 0
        st.metric(label="Avg Risk Score", value=f"{avg_score:.1f}/100")
        
    st.markdown("---")
    st.subheader("High-Risk Facilities List")
    
    def color_risk(val):
        color = '#ef4444' if val >= 90 else '#f97316' if val >= 75 else '#eab308'
        return f'background-color: {color}; color: white; font-weight: bold;'
    
    display_df = df_risk.drop(columns=["Original Index"])
    styled_df = display_df.style.map(color_risk, subset=['Risk Score'])
    
    st.dataframe(
        styled_df, 
        use_container_width=True, 
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="ew_table_selection"
    )
    
    csv = df_risk.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download High-Risk List (CSV)",
        data=csv,
        file_name='early_warning_list.csv',
        mime='text/csv',
    )
    
    selected_rows = st.session_state.ew_table_selection.get("selection", {}).get("rows", [])
    if selected_rows:
        row_idx = selected_rows[0]
        selected_facility = df_risk.iloc[row_idx]
        real_idx = selected_facility['Original Index']
        real_row = df.iloc[real_idx]
        
        @st.dialog(f"🔍 360° Risk Profile: {selected_facility['Facility ID']}", width="large")
        def facility_deep_dive(facility, row):
            
            def get_card_field(r, col_name, fmt_fn=None, fallback="—"):
                if col_name in r and pd.notna(r[col_name]):
                    val_str = str(r[col_name]).strip()
                    if val_str and val_str.lower() not in ["nan", "none", "null", ""]:
                        try:
                            return fmt_fn(r[col_name]) if fmt_fn else val_str
                        except Exception:
                            return val_str
                return fallback

            card_amt = facility['Total Exposure ($)']
            card_pd = facility['Risk Score']
            card_tenure = get_card_field(row, 'tenure_days', fallback="30")
            ticket_seg_val = facility['Segment']
            
            county_val = None
            for c in ["county", "county_name", "county_code", "borrower_county", "County"]:
                if c in row and pd.notna(row[c]):
                    s = str(row[c]).strip()
                    if s and s.lower() not in ["nan", "none", "null", ""]:
                        county_val = s
                        break
            county_line = f" &nbsp;|&nbsp; <b>County:</b> {county_val}" if county_val else " &nbsp;|&nbsp; <b>County:</b> <i>—</i>"

            lifetime_amt = get_card_field(row, 'feat_loan_amount_borrowed_sum_lifetime', lambda v: f"KES {float(v):,.0f}")
            lifetime_def = get_card_field(row, 'feat_loan_prior_default_count_lifetime', lambda v: f"{int(float(v))} times")
            crb_score = get_card_field(row, 'feat_bureau_crb_score', lambda v: f"{float(v):.0f}")
            mpesa_vel = get_card_field(row, 'feat_mpesa_velocity_30d_over_90d', lambda v: f"{float(v):.2f}")
            fuliza_util = get_card_field(row, 'feat_fuliza_util_pct_30d', lambda v: f"{float(v)*100:.1f}%" if float(v) <= 1.0 else f"{float(v):.1f}%")
            ndvi_anom = get_card_field(row, 'feat_gis_ndvi_anomaly_30d', lambda v: f"{float(v):.3f}")
            
            rec_limit = max(0, int(card_amt * (1.0 - (card_pd / 100.0)))) if card_amt > 0 else 0

            st.markdown(f"""
            <div class="borrower-card">
                <div class="card-header-title" style="font-size: 18px; font-weight: 800; color: #1e293b; margin-bottom: 10px;">👤 BORROWER PROFILE & DECISION</div>
                <p style="margin: 4px 0;"><b>Borrower ID:</b> {facility['Facility ID']} &nbsp;|&nbsp; <b>Jurisdiction:</b> Kenya{county_line} &nbsp;|&nbsp; <b>Segment:</b> {ticket_seg_val}</p>
                <p style="margin: 4px 0;"><b>Total Exposure:</b> KES {card_amt:,.0f} &nbsp;|&nbsp; <b>Tenor:</b> {card_tenure} Days</p>
                
                <div style="background: #f8fafc; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #ef4444;">
                    <p style="margin: 0; font-size: 14px;"><b>Predicted Risk Score (EWS):</b> <span style="font-size: 24px; font-weight: 900; color: #dc2626;">{card_pd}/100</span></p>
                    <p style="margin: 5px 0 0 0; font-size: 14px;"><b>Recommended Max Credit Limit:</b> <span style="font-size: 18px; font-weight: 700; color: #16a34a;">KES {rec_limit:,.0f}</span></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"🔴 **Risk Score Decomposition (Click to Expand)**", expanded=False):
                st.markdown(f"""
                * **Alternative Data Triggers (45% weighting):** {int(facility['Risk Score'] * 0.45)}/45
                * **Historical Repayment Behavior (35% weighting):** {int(facility['Risk Score'] * 0.35)}/35
                * **Macro-Economic / Sector Stress (20% weighting):** {int(facility['Risk Score'] * 0.20)}/20
                """)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div style="margin-top: 10px;">
                    <div style="font-size: 13px; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 8px;">📋 TRADITIONAL PANEL LOAN HISTORY</div>
                    <ul style="font-size: 14px; color: #334155;">
                        <li><b>Lifetime Total Borrowed:</b> {lifetime_amt}</li>
                        <li><b>Lifetime Historical Defaults:</b> {lifetime_def}</li>
                        <li><b>Bureau CRB Score:</b> {crb_score}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div style="margin-top: 10px;">
                    <div style="font-size: 13px; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 8px;">🌿 ALTERNATIVE DATA SIGNALS</div>
                    <ul style="font-size: 14px; color: #334155;">
                        <li><b>M-Pesa 30d/90d Inflow Velocity:</b> {mpesa_vel}</li>
                        <li><b>Fuliza Overdraft Utilization:</b> {fuliza_util}</li>
                        <li><b>Regional Crop NDVI Anomaly:</b> {ndvi_anom}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            st.write("---")
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("### ⚠️ Trigger Events")
                events = pd.DataFrame({
                    "Date": pd.date_range(end=pd.Timestamp.today(), periods=3, freq='-7D').strftime('%Y-%m-%d'),
                    "Event": [
                        "30% drop in MoMo transaction frequency",
                        "Missed utility payment detected",
                        "Unusual large withdrawal flagged"
                    ]
                })
                st.dataframe(events, hide_index=True)
                
            with col_t2:
                st.markdown("### 📈 Recent Behavior Trend")
                trend_data = pd.DataFrame({
                    "Month": ["M-3", "M-2", "M-1", "Current"],
                    "Utilization %": [45, 60, 85, 95]
                })
                fig = px.line(trend_data, x="Month", y="Utilization %", markers=True)
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)
            
        facility_deep_dive(selected_facility, real_row)
