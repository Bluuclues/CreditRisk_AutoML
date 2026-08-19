"""
dispatcher.py
Model Training & Explainability Dispatcher.
Routes requests to PyCaret Engine and fits TreeSHAP Explainers.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from modules.models.pycaret_engine import CreditRiskAutoMLEngine, AutoMLConfig
from modules.models.shap_explainer import CreditRiskExplainer


def run_automl_pipeline(
    df: pd.DataFrame, 
    optimize_metric: str = "PR-AUC",
    tune_hyperparams: bool = True,
    create_ensemble: bool = True,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Orchestrates full AutoML execution and fits TreeSHAP explainer.
    """
    config = AutoMLConfig(
        target_col="default_flag",
        task_type="auto",
        optimize_metric=optimize_metric,
        tune_hyperparameters=tune_hyperparams,
        create_ensemble=create_ensemble
    )

    engine = CreditRiskAutoMLEngine(config)
    results = engine.run_pipeline(df, progress_callback=progress_callback)

    champion_model = results["champion_model"]

    # Compute predictions on dataset
    X_matrix = df.drop(columns=[c for c in ["default_flag", "session_id", "borrower_id", "loan_no", "loan_date", "due_date", "payoff_date", "country_code"] if c in df.columns])
    
    predicted_probs = engine.predict_probabilities(X_matrix)

    # Initialize SHAP explainer
    explainer = None
    try:
        sample_subset = X_matrix.head(200)
        explainer = CreditRiskExplainer(champion_model, sample_subset)
    except Exception:
        explainer = None

    results["predicted_probs"] = predicted_probs
    results["explainer"] = explainer
    results["engine"] = engine
    results["X_matrix"] = X_matrix

    return results


def run_models(df: pd.DataFrame, selected_models: List[str]) -> Dict[str, Any]:
    """Backward compatibility wrapper for app.py."""
    return run_automl_pipeline(df)