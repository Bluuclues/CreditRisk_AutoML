import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_stress_testing():
    st.markdown('<div style="font-family: \'DM Sans\', sans-serif; font-size: 56px; font-weight: 900; color: #2b5b7c; text-transform: uppercase; margin-bottom: 40px; letter-spacing: -1px; display: flex; align-items: center; gap: 15px;"><span>📉</span> STRESS TESTING</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore how macroeconomic and alternative data shocks impact the portfolio. Adjust the sliders below to simulate different stress scenarios.
    """)
    
    col_sliders, col_charts = st.columns([1, 2])
    
    with col_sliders:
        st.subheader("Scenario Parameters")
        st.markdown("Adjust macro variables to simulate shocks:")
        
        cbr_shock = st.slider("Central Bank Rate (CBR) Shock", min_value=-5.0, max_value=10.0, value=2.0, step=0.5, format="%f%%")
        inflation_shock = st.slider("Inflation Rate Shock", min_value=-2.0, max_value=15.0, value=5.0, step=0.5, format="%f%%")
        unemployment_shock = st.slider("Unemployment Rate Shock", min_value=-1.0, max_value=10.0, value=1.0, step=0.5, format="%f%%")
        
        # Calculate Mock Impact
        base_npl = 5.2
        npl_impact = (cbr_shock * 0.4) + (inflation_shock * 0.25) + (unemployment_shock * 0.6)
        stressed_npl = max(0, base_npl + npl_impact)
        
        base_provision = 12.5 # in Millions
        provision_impact = npl_impact * 2.1
        stressed_provision = max(0, base_provision + provision_impact)
        
        st.markdown("---")
        st.metric("Projected NPL Ratio", f"{stressed_npl:.1f}%", f"{npl_impact:+.1f}% from base", delta_color="inverse")
        st.metric("Required Provisions ($M)", f"${stressed_provision:.1f}M", f"${provision_impact:+.1f}M from base", delta_color="inverse")
        
    with col_charts:
        st.subheader("Portfolio Impact Visualization")
        
        # Mock Data for Chart
        scenarios = ['Baseline', 'Stressed Scenario']
        npl_values = [base_npl, stressed_npl]
        prov_values = [base_provision, stressed_provision]
        
        fig = go.Figure(data=[
            go.Bar(name='NPL Ratio (%)', x=scenarios, y=npl_values, marker_color='#3b82f6'),
            go.Bar(name='Provisions ($M)', x=scenarios, y=prov_values, marker_color='#ef4444')
        ])
        
        fig.update_layout(
            barmode='group',
            title='Baseline vs Stressed Portfolio Health',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#334155'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Insight:** The simulated shock in the Central Bank Rate is the primary driver of the increased Non-Performing Loans (NPL) in this scenario.")
