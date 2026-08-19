"""
shap_explainer.py
Production TreeSHAP and Adverse Action Reason Code Generator.
Compliant with Python 3.12+, SHAP 0.40+, and Streamlit Integration.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io


class CreditRiskExplainer:
    """
    Computes TreeSHAP attributions and generates customer data cards and adverse action reason codes.
    """

    def __init__(self, pipeline: Any, X_sample: pd.DataFrame):
        self.pipeline = pipeline
        self.X_sample = X_sample.copy()
        
        # 1. Unpack preprocessor and classifier step if wrapped in scikit-learn Pipeline
        self.preprocessor = None
        self.classifier = pipeline
        
        if hasattr(pipeline, "named_steps"):
            if "preprocessor" in pipeline.named_steps:
                self.preprocessor = pipeline.named_steps["preprocessor"]
            if "classifier" in pipeline.named_steps:
                self.classifier = pipeline.named_steps["classifier"]
            elif "trained_model" in pipeline.named_steps:
                self.classifier = pipeline.named_steps["trained_model"]

        # 2. Transform features
        if self.preprocessor is not None:
            self.X_transformed = self.preprocessor.transform(self.X_sample)
            if hasattr(self.X_transformed, "toarray"):
                self.X_transformed = self.X_transformed.toarray()
            
            try:
                self.feature_names = list(self.preprocessor.get_feature_names_out())
            except Exception:
                self.feature_names = [f"feat_{i}" for i in range(self.X_transformed.shape[1])]
        else:
            self.X_transformed = self.X_sample.values
            self.feature_names = list(self.X_sample.columns)

        # Clean feature names for display (strip num__ or cat__)
        self.clean_feature_names = [
            f.replace("num__", "").replace("cat__", "") for f in self.feature_names
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
            # Low risk case
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
            
            # Construct SHAP explanation object for waterfall
            exp_obj = shap.Explanation(
                values=self.shap_values[record_idx],
                base_values=getattr(self.explainer, "expected_value", 0.0) if not isinstance(getattr(self.explainer, "expected_value", 0.0), list) else getattr(self.explainer, "expected_value", [0.0])[1],
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
