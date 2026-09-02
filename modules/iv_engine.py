"""
iv_engine.py
Calculates Information Value (IV) and Weight of Evidence (WoE) for feature screening.
"""

import numpy as np
import pandas as pd
import plotly.express as px

def calculate_portfolio_iv(
    df: pd.DataFrame, target: str = "default_flag", n_bins: int = 10
) -> pd.DataFrame:
    """Computes Information Value (IV) across all candidate features."""
    exclude_cols = [
        target,
        "loan_id",
        "loan_no",
        "session_id",
        "borrower_id",
        "national_id",
        "country_code",
        "loan_date",
        "due_date",
        "payoff_date",
    ]
    features = [
        col
        for col in df.columns
        if col not in exclude_cols and not col.endswith("_id")
    ]

    total_goods = (df[target] == 0).sum()
    total_bads = (df[target] == 1).sum()
    epsilon = 1e-6

    records = []
    for col in features:
        try:
            temp = df[[col, target]].copy()
            if (
                pd.api.types.is_numeric_dtype(temp[col])
                and temp[col].nunique() > n_bins
            ):
                temp["bin"] = pd.qcut(temp[col], q=n_bins, duplicates="drop")
            else:
                temp["bin"] = temp[col].fillna("Missing").astype(str)

            agg = (
                temp.groupby("bin", observed=False)[target]
                .agg(
                    bads="sum",
                    goods=lambda x: (x == 0).sum(),
                )
                .reset_index()
            )

            dist_good = (agg["goods"] + epsilon) / (
                total_goods + epsilon * len(agg)
            )
            dist_bad = (agg["bads"] + epsilon) / (
                total_bads + epsilon * len(agg)
            )
            woe = np.log(dist_good / dist_bad)
            iv = ((dist_good - dist_bad) * woe).sum()

            if iv >= 0.50:
                band = "Very Strong"
            elif iv >= 0.30:
                band = "Strong"
            elif iv >= 0.10:
                band = "Medium"
            elif iv >= 0.02:
                band = "Weak"
            else:
                band = "Noise / Prune"

            records.append(
                {
                    "Feature Name": col,
                    "Information Value (IV)": round(iv, 4),
                    "Predictive Power": band,
                    "Action": (
                        "Keep" if iv >= 0.02 else "Prune (Auto-Filtered)"
                    ),
                }
            )
        except Exception:
            continue

    # Create the DataFrame
    df_out = pd.DataFrame(records)
    
    # Generate deterministic mock values for Collection Hardness and Evidence x IV for the quadrant chart
    import hashlib
    def get_pseudo_score(seed_str, min_val, max_val):
        hash_val = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
        return min_val + (hash_val % (max_val - min_val + 1))
        
    df_out["Collection Hardness"] = df_out["Feature Name"].apply(lambda x: get_pseudo_score(x + "hard", 1, 10))
    df_out["Frequency"] = df_out["Feature Name"].apply(lambda x: get_pseudo_score(x + "freq", 5, 10))
    df_out["Evidence x IV"] = df_out["Frequency"] * df_out["Information Value (IV)"]

    return df_out.sort_values(by="Information Value (IV)", ascending=False).reset_index(drop=True)

def plot_iv_chart(iv_df: pd.DataFrame):
    """Generates an interactive Plotly horizontal bar chart of Information Value."""
    # Reverse so highest IV is at top
    chart_df = iv_df.sort_values(by="Information Value (IV)", ascending=True)
    
    fig = px.bar(
        chart_df,
        x="Information Value (IV)",
        y="Feature Name",
        orientation="h",
        color="Information Value (IV)",
        color_continuous_scale="YlGn",
        title="Feature Information Value (IV) Ranking",
        labels={"Information Value (IV)": "IV Score", "Feature Name": ""}
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_showscale=False
    )
    return fig

def plot_iv_quadrant_chart(iv_df: pd.DataFrame):
    """Generates a quadrant scatter plot for Variable Discoverability (Collection Hardness vs Evidence x IV)."""
    if iv_df.empty:
        return None
        
    mid_x = 5.5  # Hardness scale is 1-10
    mid_y = iv_df["Evidence x IV"].median() if not pd.isna(iv_df["Evidence x IV"].median()) else 0.5

    fig = px.scatter(
        iv_df,
        x="Collection Hardness",
        y="Evidence x IV",
        text="Feature Name",
        color="Predictive Power",
        color_discrete_map={
            "Very Strong": "#1a9641",
            "Strong": "#a6d96a",
            "Medium": "#fdae61",
            "Weak": "#d7191c",
            "Noise / Prune": "#cccccc"
        },
        title="Variable Discoverability Matrix"
    )
    
    fig.update_traces(
        textposition='top center', 
        marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')),
        textfont=dict(size=10)
    )
    
    # Add Quadrant Lines
    fig.add_vline(x=mid_x, line_width=2, line_dash="dash", line_color="black")
    fig.add_hline(y=mid_y, line_width=2, line_dash="dash", line_color="black")
    
    # Add Annotations for Quadrants
    max_y = iv_df["Evidence x IV"].max()
    min_y = iv_df["Evidence x IV"].min()
    y_range = max_y - min_y if max_y > min_y else 1
    
    fig.add_annotation(x=3, y=max_y, text="<b>Q1<br>Low Hardness<br>High Evidence x IV</b>", showarrow=False, font=dict(color="black", size=11), align="center")
    fig.add_annotation(x=8, y=max_y, text="<b>Q2<br>High Hardness<br>High Evidence x IV</b>", showarrow=False, font=dict(color="black", size=11), align="center")
    fig.add_annotation(x=3, y=min_y, text="<b>Q3<br>Low Hardness<br>Low Evidence x IV</b>", showarrow=False, font=dict(color="black", size=11), align="center")
    fig.add_annotation(x=8, y=min_y, text="<b>Q4<br>High Hardness<br>Low Evidence x IV</b>", showarrow=False, font=dict(color="black", size=11), align="center")
    
    fig.update_layout(
        xaxis_title="Collection Hardness (Low ➔ High)",
        yaxis_title="Frequency x IV (Low ➔ High)",
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(range=[0, 11], showgrid=False),
        yaxis=dict(showgrid=False),
        showlegend=False
    )
    
    # Add bounding box
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    
    return fig
