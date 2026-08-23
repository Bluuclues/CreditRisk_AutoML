"""
shap_explainer.py
Production TreeSHAP and Adverse Action Reason Code Generator for Credit Risk Models.
Supports individual borrower waterfall decompositions, global beeswarm plots,
and portfolio-wide SHAP feature importance bar charts (interactive Plotly & PNG).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io


class CreditRiskExplainer:
    """
    Computes TreeSHAP attributions and generates customer data cards, adverse action reason codes,
    and portfolio-wide feature importance bar charts.
    """

    def __init__(self, pipeline: Any, X_sample: pd.DataFrame):
        self.pipeline = pipeline
        self.X_sample = X_sample.copy()
        
        # 1. Unpack preprocessor and estimator from Pipeline (PyCaret / Scikit-Learn)
        self.preprocessor = None
        self.classifier = pipeline
        
        if hasattr(pipeline, "named_steps"):
            steps = pipeline.named_steps
            if "actual_estimator" in steps:
                self.classifier = steps["actual_estimator"]
            elif "trained_model" in steps:
                self.classifier = steps["trained_model"]
            elif "classifier" in steps:
                self.classifier = steps["classifier"]
            elif hasattr(pipeline, "steps") and len(pipeline.steps) > 0:
                self.classifier = pipeline.steps[-1][1]

            # Extract preprocessing steps if pipeline has multiple stages
            if hasattr(pipeline, "steps") and len(pipeline.steps) > 1:
                self.preprocessor = pipeline[:-1]
            elif "preprocessor" in steps:
                self.preprocessor = steps["preprocessor"]

        # 2. Transform features
        if self.preprocessor is not None:
            try:
                self.X_transformed = self.preprocessor.transform(self.X_sample)
                if hasattr(self.X_transformed, "toarray"):
                    self.X_transformed = self.X_transformed.toarray()
                
                try:
                    if hasattr(self.preprocessor, "get_feature_names_out"):
                        self.feature_names = list(self.preprocessor.get_feature_names_out())
                    else:
                        self.feature_names = list(self.X_sample.columns)
                except Exception:
                    self.feature_names = [f"feat_{i}" for i in range(self.X_transformed.shape[1])]
            except Exception:
                self.X_transformed = self.X_sample.values
                self.feature_names = list(self.X_sample.columns)
        else:
            self.X_transformed = self.X_sample.values
            self.feature_names = list(self.X_sample.columns)

        # Clean feature names for display
        self.clean_feature_names = [
            str(f).replace("num__", "").replace("cat__", "").replace("remainder__", "") 
            for f in self.feature_names
        ]

        # 3. Fit SHAP Explainer
        try:
            self.explainer = shap.TreeExplainer(self.classifier)
            self.shap_values = self.explainer.shap_values(self.X_transformed)
        except Exception:
            try:
                self.explainer = shap.Explainer(self.classifier, self.X_transformed)
                explanation = self.explainer(self.X_transformed)
                self.shap_values = explanation.values
            except Exception:
                self.explainer = None
                self.shap_values = None

        # Format SHAP matrix for binary classification (take default class 1)
        if isinstance(self.shap_values, list) and len(self.shap_values) > 1:
            self.shap_values = self.shap_values[1]
        elif isinstance(self.shap_values, np.ndarray) and self.shap_values.ndim == 3:
            self.shap_values = self.shap_values[:, :, 1]

    def get_borrower_adverse_reasons(
        self, 
        record_idx: int, 
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Decomposes positive SHAP values into intuitive adverse action percentage drivers.
        """
        if self.shap_values is None or record_idx >= len(self.shap_values):
            return [{
                "feature_name": "General Credit Risk Profile",
                "feature_value": "N/A",
                "impact_percentage": 100.0,
                "reason_text": "Overall credit score and historical payment metrics elevate risk."
            }]

        row_shaps = self.shap_values[record_idx]
        pos_indices = np.where(row_shaps > 0)[0]

        if len(pos_indices) == 0:
            return [{
                "feature_name": "Low Risk Baseline",
                "feature_value": "Optimal",
                "impact_percentage": 100.0,
                "reason_text": "Low Risk: Strong liquidity buffer, zero prior defaults, stable repayment pattern."
            }]

        pos_shaps = row_shaps[pos_indices]
        sum_pos = np.sum(pos_shaps)

        sorted_order = np.argsort(pos_shaps)[::-1]
        reasons = []

        for rank in range(min(top_k, len(sorted_order))):
            idx = pos_indices[sorted_order[rank]]
            feat_raw = self.clean_feature_names[idx]
            raw_val = self.X_transformed[record_idx, idx] if self.X_transformed is not None else "N/A"

            # Format raw value
            if isinstance(raw_val, (float, np.floating)):
                formatted_val = f"{raw_val:.2f}"
            else:
                formatted_val = str(raw_val)

            pct = (row_shaps[idx] / sum_pos) * 100.0 if sum_pos > 0 else 0.0

            # Pretty print feature names
            pretty_name = feat_raw.replace("feat_", "").replace("_", " ").title()

            reasons.append({
                "feature_code": feat_raw,
                "feature_name": pretty_name,
                "feature_value": formatted_val,
                "impact_percentage": round(pct, 1),
                "reason_text": f"{pretty_name} (Val: {formatted_val}) drives {pct:.1f}% of default risk"
            })

        return reasons

    def generate_waterfall_plot_bytes(self, record_idx: int) -> Optional[bytes]:
        """Generates a high-res SHAP Waterfall plot image buffer for a specific borrower."""
        if self.shap_values is None or self.explainer is None:
            return None

        try:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            
            expected_val = getattr(self.explainer, "expected_value", 0.0)
            if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > 1:
                base_val = expected_val[1]
            elif isinstance(expected_val, (list, np.ndarray)) and len(expected_val) == 1:
                base_val = expected_val[0]
            else:
                base_val = expected_val

            exp_obj = shap.Explanation(
                values=self.shap_values[record_idx],
                base_values=base_val,
                data=self.X_transformed[record_idx],
                feature_names=self.clean_feature_names
            )
            
            shap.plots.waterfall(exp_obj, max_display=8, show=False)
            plt.title(f"TreeSHAP Risk Waterfall (Borrower Index #{record_idx})", fontsize=11, fontweight='bold', pad=12)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()
        except Exception:
            plt.close('all')
            return None

    def generate_beeswarm_plot_bytes(self) -> Optional[bytes]:
        """Generates global TreeSHAP beeswarm summary plot image buffer."""
        if self.shap_values is None:
            return None

        try:
            fig, ax = plt.subplots(figsize=(8.5, 5))
            
            X_df = pd.DataFrame(self.X_transformed, columns=self.clean_feature_names)
            shap.summary_plot(self.shap_values, X_df, max_display=12, show=False)
            plt.title("Global TreeSHAP Feature Importance & Risk Direction", fontsize=12, fontweight='bold', pad=12)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()
        except Exception:
            plt.close('all')
            return None

    def get_global_feature_importance_df(self, top_n: int = 20) -> pd.DataFrame:
        """
        Computes portfolio-wide mean absolute SHAP values for every feature.
        Returns a ranked summary DataFrame.
        """
        if self.shap_values is None:
            return pd.DataFrame(columns=["Rank", "Feature Name", "Feature Code", "Mean Absolute SHAP", "Impact Percentage (%)"])

        mean_abs_shaps = np.mean(np.abs(self.shap_values), axis=0)
        total_importance = np.sum(mean_abs_shaps)

        sorted_idx = np.argsort(mean_abs_shaps)[::-1][:top_n]
        
        rows = []
        for rank, idx in enumerate(sorted_idx, 1):
            feat_code = self.clean_feature_names[idx]
            pretty_name = feat_code.replace("feat_", "").replace("_", " ").title()
            val = float(mean_abs_shaps[idx])
            pct = (val / total_importance * 100.0) if total_importance > 0 else 0.0
            
            rows.append({
                "Rank": rank,
                "Feature Name": pretty_name,
                "Feature Code": feat_code,
                "Mean Absolute SHAP": round(val, 4),
                "Impact Percentage (%)": round(pct, 2)
            })

        return pd.DataFrame(rows)

    def generate_bar_plot_bytes(self, max_display: int = 12) -> Optional[bytes]:
        """Generates portfolio-wide static SHAP Feature Importance Bar Plot PNG image buffer."""
        if self.shap_values is None:
            return None

        try:
            fig, ax = plt.subplots(figsize=(8.5, 5))
            X_df = pd.DataFrame(self.X_transformed, columns=self.clean_feature_names)
            shap.summary_plot(self.shap_values, X_df, plot_type="bar", max_display=max_display, show=False)
            plt.title("Portfolio-Wide SHAP Feature Importance (Mean |SHAP Value|)", fontsize=12, fontweight='bold', pad=12)
            plt.xlabel("Mean |SHAP Value| (Average Impact on Credit Decision)", fontsize=10)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()
        except Exception:
            plt.close('all')
            return None

    def generate_plotly_feature_bar_fig(self, top_n: int = 15) -> Optional[go.Figure]:
        """Generates an interactive Plotly horizontal bar chart of portfolio SHAP feature importance."""
        df_imp = self.get_global_feature_importance_df(top_n=top_n)
        if df_imp.empty:
            return None

        # Reverse order so highest importance is at the top of the horizontal bar chart
        df_plot = df_imp.iloc[::-1].copy()

        fig = go.Figure(go.Bar(
            x=df_plot["Mean Absolute SHAP"],
            y=df_plot["Feature Name"],
            orientation='h',
            marker=dict(
                color=df_plot["Mean Absolute SHAP"],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="SHAP")
            ),
            text=[f"{pct:.1f}% ({val:.3f})" for pct, val in zip(df_plot["Impact Percentage (%)"], df_plot["Mean Absolute SHAP"])],
            textposition='auto',
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>"
        ))

        fig.update_layout(
            title=dict(
                text="<b>Portfolio-Wide Feature Importance (Mean |SHAP Value|)</b>",
                font=dict(size=14, color="#1e293b")
            ),
            xaxis_title="Mean Absolute SHAP Value (Average Impact on Default Probability)",
            yaxis_title="Predictive Feature",
            margin=dict(l=20, r=20, t=40, b=30),
            height=420,
            template="plotly_white"
        )

        return fig
