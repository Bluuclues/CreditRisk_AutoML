"""
eda_visualizer.py
Comprehensive Exploratory Data Analysis (EDA) & Descriptive Statistics Generator.
Generates automated summary statistics, correlation heatmaps, feature distribution histograms,
and target separation boxplots with full PNG, CSV, and Excel export support.
"""

from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import io


class CreditRiskEDA:
    """
    Automated EDA engine for credit risk panel and alternative data matrices.
    """

    @staticmethod
    def generate_descriptive_stats_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes detailed summary statistics including central tendency,
        spread, missing rates, and skewness for all numerical and categorical features.
        """
        records = []
        total_rows = len(df)

        for col in df.columns:
            s = df[col]
            dtype = str(s.dtype)
            missing_cnt = int(s.isna().sum())
            missing_pct = round((missing_cnt / total_rows) * 100.0, 2) if total_rows > 0 else 0.0
            unique_cnt = int(s.nunique(dropna=True))

            if pd.api.types.is_numeric_dtype(s):
                valid_s = s.dropna()
                mean_val = round(float(valid_s.mean()), 2) if len(valid_s) > 0 else np.nan
                std_val = round(float(valid_s.std()), 2) if len(valid_s) > 1 else np.nan
                min_val = round(float(valid_s.min()), 2) if len(valid_s) > 0 else np.nan
                p25_val = round(float(valid_s.quantile(0.25)), 2) if len(valid_s) > 0 else np.nan
                median_val = round(float(valid_s.median()), 2) if len(valid_s) > 0 else np.nan
                p75_val = round(float(valid_s.quantile(0.75)), 2) if len(valid_s) > 0 else np.nan
                max_val = round(float(valid_s.max()), 2) if len(valid_s) > 0 else np.nan
                skew_val = round(float(valid_s.skew()), 2) if len(valid_s) > 2 else np.nan

                records.append({
                    "Feature": col,
                    "Type": dtype,
                    "Count": len(valid_s),
                    "Missing (%)": f"{missing_pct}%",
                    "Unique": unique_cnt,
                    "Mean": mean_val,
                    "Std": std_val,
                    "Min": min_val,
                    "25%": p25_val,
                    "Median (50%)": median_val,
                    "75%": p75_val,
                    "Max": max_val,
                    "Skewness": skew_val
                })
            else:
                top_val = str(s.mode().iloc[0]) if not s.mode().empty else "N/A"
                records.append({
                    "Feature": col,
                    "Type": dtype,
                    "Count": total_rows - missing_cnt,
                    "Missing (%)": f"{missing_pct}%",
                    "Unique": unique_cnt,
                    "Mean": top_val,
                    "Std": np.nan,
                    "Min": np.nan,
                    "25%": np.nan,
                    "Median (50%)": np.nan,
                    "75%": np.nan,
                    "Max": np.nan,
                    "Skewness": np.nan
                })

        return pd.DataFrame(records)

    @staticmethod
    def generate_correlation_matrix(df: pd.DataFrame, max_cols: int = 16) -> pd.DataFrame:
        """Computes Pearson correlation matrix on numeric predictive features."""
        ignore_cols = ["session_id", "borrower_id", "loan_no", "loan_date", "due_date", "payoff_date", "country_code"]
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ignore_cols][:max_cols]
        if not num_cols:
            return pd.DataFrame()
        return df[num_cols].corr().round(3)

    @staticmethod
    def generate_correlation_heatmap_fig(df: pd.DataFrame, max_cols: int = 16) -> Optional[go.Figure]:
        """Generates an interactive Plotly correlation heatmap."""
        corr_df = CreditRiskEDA.generate_correlation_matrix(df, max_cols=max_cols)
        if corr_df.empty:
            return None

        # Clean display labels
        clean_labels = [c.replace("feat_", "").replace("_", " ").title() for c in corr_df.columns]

        fig = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=clean_labels,
            y=clean_labels,
            colorscale='RdBu_r',
            zmin=-1.0,
            zmax=1.0,
            text=np.around(corr_df.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(title="Pearson r")
        ))

        fig.update_layout(
            title=dict(text="<b>Feature Collinearity & Cross-Correlation Matrix</b>", font=dict(size=14)),
            xaxis=dict(tickangle=-45),
            margin=dict(l=20, r=20, t=40, b=80),
            height=480,
            template="plotly_white"
        )
        return fig

    @staticmethod
    def generate_correlation_heatmap_bytes(df: pd.DataFrame, max_cols: int = 16) -> Optional[bytes]:
        """Generates a high-res Seaborn/Matplotlib correlation heatmap PNG buffer."""
        corr_df = CreditRiskEDA.generate_correlation_matrix(df, max_cols=max_cols)
        if corr_df.empty:
            return None

        try:
            fig, ax = plt.subplots(figsize=(9, 7))
            clean_labels = [c.replace("feat_", "").replace("_", " ").title() for c in corr_df.columns]
            sns.heatmap(
                corr_df, 
                annot=True, 
                fmt=".2f", 
                cmap="vlag", 
                vmin=-1, 
                vmax=1, 
                xticklabels=clean_labels, 
                yticklabels=clean_labels, 
                ax=ax,
                annot_kws={"size": 8}
            )
            plt.title("Feature Collinearity & Cross-Correlation Matrix", fontsize=12, fontweight='bold', pad=12)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=180, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()
        except Exception:
            plt.close('all')
            return None

    @staticmethod
    def generate_feature_distributions_fig(df: pd.DataFrame, target_col: str = "default_flag", max_cols: int = 6) -> Optional[go.Figure]:
        """Generates interactive distribution histograms for key features segmented by default flag."""
        ignore_cols = ["session_id", "borrower_id", "loan_no", "loan_date", "due_date", "payoff_date", "country_code", target_col]
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ignore_cols][:max_cols]
        
        if not numeric_cols:
            return None

        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        titles = [c.replace("feat_", "").replace("_", " ").title() for c in numeric_cols]
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles)

        for i, col in enumerate(numeric_cols):
            r = (i // n_cols) + 1
            c = (i % n_cols) + 1

            if target_col in df.columns:
                for target_val, color, name in [(0, '#10b981', 'Performing (0)'), (1, '#ef4444', 'Defaulted (1)')]:
                    subset = df[df[target_col] == target_val][col].dropna()
                    fig.add_trace(
                        go.Histogram(
                            x=subset, 
                            name=name, 
                            marker_color=color, 
                            opacity=0.6, 
                            showlegend=(i == 0),
                            nbinsx=20
                        ),
                        row=r, col=c
                    )
            else:
                fig.add_trace(
                    go.Histogram(x=df[col].dropna(), marker_color='#3b82f6', opacity=0.7, showlegend=False, nbinsx=20),
                    row=r, col=c
                )

        fig.update_layout(
            title_text="<b>Key Feature Value Distributions by Default Status</b>",
            barmode='overlay',
            height=260 * n_rows,
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=30)
        )
        return fig

    @staticmethod
    def generate_feature_distributions_bytes(df: pd.DataFrame, target_col: str = "default_flag", max_cols: int = 6) -> Optional[bytes]:
        """Generates a high-res static PNG of feature distribution histograms."""
        ignore_cols = ["session_id", "borrower_id", "loan_no", "loan_date", "due_date", "payoff_date", "country_code", target_col]
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ignore_cols][:max_cols]
        
        if not numeric_cols:
            return None

        try:
            n_cols = 3
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5 * n_rows))
            axes = np.array(axes).flatten()

            for i, col in enumerate(numeric_cols):
                ax = axes[i]
                if target_col in df.columns:
                    sns.histplot(data=df, x=col, hue=target_col, kde=True, ax=ax, palette={0: "#10b981", 1: "#ef4444"}, alpha=0.5)
                else:
                    sns.histplot(data=df, x=col, kde=True, ax=ax, color="#3b82f6")

                pretty_name = col.replace("feat_", "").replace("_", " ").title()
                ax.set_title(pretty_name, fontsize=10, fontweight='bold')
                ax.set_xlabel("")
                ax.set_ylabel("Count")

            # Hide unused axes
            for j in range(len(numeric_cols), len(axes)):
                axes[j].set_visible(False)

            plt.suptitle("Feature Value Distributions & Default Separation", fontsize=12, fontweight='bold', y=1.02)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()
        except Exception:
            plt.close('all')
            return None

    @staticmethod
    def generate_boxplots_by_target_fig(df: pd.DataFrame, target_col: str = "default_flag", max_cols: int = 6) -> Optional[go.Figure]:
        """Generates boxplots showing risk separation across key continuous features."""
        ignore_cols = ["session_id", "borrower_id", "loan_no", "loan_date", "due_date", "payoff_date", "country_code", target_col]
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ignore_cols][:max_cols]
        
        if not numeric_cols or target_col not in df.columns:
            return None

        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        titles = [c.replace("feat_", "").replace("_", " ").title() for c in numeric_cols]
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles)

        for i, col in enumerate(numeric_cols):
            r = (i // n_cols) + 1
            c = (i % n_cols) + 1

            for target_val, color, name in [(0, '#10b981', 'Performing'), (1, '#ef4444', 'Defaulted')]:
                subset = df[df[target_col] == target_val][col].dropna()
                fig.add_trace(
                    go.Box(y=subset, name=name, marker_color=color, showlegend=(i == 0)),
                    row=r, col=c
                )

        fig.update_layout(
            title_text="<b>Continuous Feature Quantiles & Outliers by Target Class</b>",
            height=260 * n_rows,
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=30)
        )
        return fig
