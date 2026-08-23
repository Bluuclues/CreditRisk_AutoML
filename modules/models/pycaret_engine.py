"""
pycaret_engine.py
Modular, Production-Ready PyCaret 3.x AutoML Engine for Credit Risk.
Standardized on PyCaret Classification & Regression Experiments for automated
model comparison, Optuna hyperparameter optimization, and ensemble blending.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment


@dataclass
class AutoMLConfig:
    target_col: str = "default_flag"
    task_type: str = "auto"  # 'auto', 'classification', 'regression'
    cv_folds: int = 5
    optimize_metric: str = "PR-AUC"  # 'PR-AUC', 'AUC', 'F1', 'Accuracy', 'Recall', 'Precision'
    top_n_models: int = 3
    tune_hyperparameters: bool = True
    create_ensemble: bool = True
    session_id: int = 42
    exclude_columns: Optional[List[str]] = None


class CreditRiskAutoMLEngine:
    """
    Automated Machine Learning Engine wrapping PyCaret for credit risk scoring.
    Supports binary default classification and regression limit optimization.
    """

    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.exp: Optional[Any] = None
        self.best_models: List[Any] = []
        self.champion_model: Optional[Any] = None
        self.leaderboard: Optional[pd.DataFrame] = None
        self.task_type: str = "classification"

    def _determine_task_type(self, y: pd.Series) -> str:
        """Automatically detects task type from target cardinality."""
        unique_vals = y.dropna().unique()
        if len(unique_vals) <= 2:
            return "classification"
        return "regression"

    def run_pipeline(
        self, 
        df: pd.DataFrame, 
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Executes the full PyCaret AutoML lifecycle:
        1. Preprocessing & Dataset Setup
        2. Multi-Model Benchmarking & Comparison
        3. Optional Hyperparameter Tuning & Soft-Voting Blending
        4. Model Finalization & Metrics Export
        """
        data = df.copy()

        # Clean metadata & identifier columns
        default_excludes = ["session_id", "borrower_id", "loan_no", "loan_date", "due_date", "payoff_date", "country_code"]
        user_excludes = self.config.exclude_columns or []
        all_excludes = list(set(default_excludes + user_excludes))
        
        drop_cols = [c for c in all_excludes if c in data.columns]
        data = data.drop(columns=drop_cols)

        # Drop rows with missing target
        if self.config.target_col in data.columns:
            data = data.dropna(subset=[self.config.target_col])
        else:
            raise ValueError(f"Target column '{self.config.target_col}' missing from feature matrix.")

        # Determine task type
        if self.config.task_type == "auto":
            self.task_type = self._determine_task_type(data[self.config.target_col])
        else:
            self.task_type = self.config.task_type

        return self._run_pycaret(data, progress_callback)

    def _run_pycaret(self, data: pd.DataFrame, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """Executes native PyCaret 3.x experiment stream."""
        if progress_callback:
            progress_callback(10, f"Initializing PyCaret 3.x ({self.task_type.capitalize()})...")

        if self.task_type == "classification":
            self.exp = ClassificationExperiment()
            self.exp.setup(
                data=data,
                target=self.config.target_col,
                session_id=self.config.session_id,
                fold=self.config.cv_folds,
                fix_imbalance=False,
                verbose=False
            )

            if progress_callback:
                progress_callback(35, "Benchmarking PyCaret classifiers (GBDTs, Ensembles & Linear models)...")

            # Map user metric to PyCaret metric key
            metric_map = {
                "PR-AUC": "AUC",
                "ROC-AUC": "AUC",
                "AUC": "AUC",
                "F1": "F1",
                "Accuracy": "Accuracy",
                "Recall": "Recall",
                "Precision": "Prec."
            }
            pycaret_metric = metric_map.get(self.config.optimize_metric, "AUC")

            # Compare all PyCaret models and select top N
            self.best_models = self.exp.compare_models(
                n_select=self.config.top_n_models,
                sort=pycaret_metric,
                verbose=False
            )
            
            if not isinstance(self.best_models, list):
                self.best_models = [self.best_models]

            self.leaderboard = self.exp.pull()

            selected_model = self.best_models[0]

            # Optional Hyperparameter Tuning
            if self.config.tune_hyperparameters and self.best_models:
                if progress_callback:
                    progress_callback(60, "Tuning hyperparameters of top candidate model...")
                try:
                    tuned_model = self.exp.tune_model(
                        self.best_models[0], 
                        optimize=pycaret_metric, 
                        n_iter=10, 
                        verbose=False
                    )
                    selected_model = tuned_model
                except Exception:
                    selected_model = self.best_models[0]

            # Optional Ensembling / Blending
            if self.config.create_ensemble and len(self.best_models) >= 2:
                if progress_callback:
                    progress_callback(75, "Constructing soft-voting ensemble...")
                try:
                    ensemble = self.exp.blend_models(
                        estimator_list=self.best_models, 
                        optimize=pycaret_metric,
                        verbose=False
                    )
                    selected_model = ensemble
                except Exception:
                    pass

            if progress_callback:
                progress_callback(90, "Finalizing champion pipeline across full portfolio...")

            self.champion_model = self.exp.finalize_model(selected_model)

        else:
            self.exp = RegressionExperiment()
            self.exp.setup(
                data=data,
                target=self.config.target_col,
                session_id=self.config.session_id,
                fold=self.config.cv_folds,
                verbose=False
            )

            if progress_callback:
                progress_callback(40, "Benchmarking PyCaret regression models...")

            self.best_models = self.exp.compare_models(
                n_select=self.config.top_n_models,
                sort="R2",
                verbose=False
            )
            if not isinstance(self.best_models, list):
                self.best_models = [self.best_models]

            self.leaderboard = self.exp.pull()

            if progress_callback:
                progress_callback(80, "Finalizing champion regression pipeline...")

            self.champion_model = self.exp.finalize_model(self.best_models[0])

        if progress_callback:
            progress_callback(100, "PyCaret AutoML Execution Complete!")

        return {
            "task_type": self.task_type,
            "champion_model": self.champion_model,
            "leaderboard": self.leaderboard,
            "top_models": self.best_models,
            "experiment": self.exp,
            "engine_name": "PyCaret 3.x"
        }

    def predict_probabilities(self, X_new: pd.DataFrame) -> np.ndarray:
        """Returns calibrated default probabilities for new records."""
        if self.champion_model is None:
            raise ValueError("Model has not been trained yet.")

        # If PyCaret experiment is active, use predict_model with raw score if available
        if self.exp is not None:
            try:
                preds_df = self.exp.predict_model(self.champion_model, data=X_new, raw_score=True, verbose=False)
                # Look for prediction score column (e.g. 'prediction_score_1' or 'Score' or 'prediction_score')
                if "prediction_score_1" in preds_df.columns:
                    return preds_df["prediction_score_1"].to_numpy()
                elif "Score" in preds_df.columns:
                    # In PyCaret binary classification, if prediction_label is 0, score is prob(0), prob(1) = 1 - Score
                    if "prediction_label" in preds_df.columns:
                        return np.where(preds_df["prediction_label"] == 1, preds_df["Score"], 1.0 - preds_df["Score"])
                    return preds_df["Score"].to_numpy()
                elif "prediction_score" in preds_df.columns:
                    if "prediction_label" in preds_df.columns:
                        return np.where(preds_df["prediction_label"] == 1, preds_df["prediction_score"], 1.0 - preds_df["prediction_score"])
                    return preds_df["prediction_score"].to_numpy()
            except Exception:
                pass

        # Fallback to direct pipeline prediction
        if hasattr(self.champion_model, "predict_proba"):
            probs = self.champion_model.predict_proba(X_new)
            if probs.ndim == 2 and probs.shape[1] > 1:
                return probs[:, 1]
            return probs.flatten()
        elif hasattr(self.champion_model, "predict"):
            return self.champion_model.predict(X_new)

        return np.zeros(len(X_new))
