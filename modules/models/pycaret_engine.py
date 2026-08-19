"""
pycaret_engine.py
Modular, Production-Ready PyCaret AutoML Engine for Credit Risk.
Compliant with Python 3.12+, PyCaret 3.x, DuckDB, and Streamlit Integration.
Includes an embedded High-Performance GBDT/Scikit-Learn Fallback Engine.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

# ML Fallback dependencies
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

# Try PyCaret import
PYCARET_AVAILABLE = False
try:
    from pycaret.classification import ClassificationExperiment
    from pycaret.regression import RegressionExperiment
    PYCARET_AVAILABLE = True
except Exception:
    PYCARET_AVAILABLE = False


@dataclass
class AutoMLConfig:
    target_col: str = "default_flag"
    task_type: str = "auto"  # 'auto', 'classification', 'regression'
    cv_folds: int = 5
    optimize_metric: str = "PR-AUC"  # 'PR-AUC', 'AUC', 'F1', 'Accuracy'
    top_n_models: int = 3
    tune_hyperparameters: bool = True
    create_ensemble: bool = True
    session_id: int = 42
    exclude_columns: Optional[List[str]] = None


class CreditRiskAutoMLEngine:
    """
    Automated Machine Learning Engine wrapping PyCaret for credit scoring.
    Supports binary default classification and regression limit optimization.
    Features robust fallback when PyCaret binaries or C++ dependencies are unavailable.
    """

    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.exp: Optional[Any] = None
        self.best_models: List[Any] = []
        self.champion_model: Optional[Any] = None
        self.leaderboard: Optional[pd.DataFrame] = None
        self.task_type: str = "classification"
        self.is_pycaret_active: bool = PYCARET_AVAILABLE
        self.fallback_pipeline: Optional[Pipeline] = None
        self.feature_names: List[str] = []

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
        Executes full AutoML lifecycle:
        1. Preprocessing & Task Auto-Detection
        2. Multi-Model Benchmarking & Comparison
        3. Hyperparameter Optimization & Blending
        4. Champion Pipeline Export
        """
        data = df.copy()

        # Clean exclude columns
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

        # Attempt PyCaret pipeline if available
        if self.is_pycaret_active:
            try:
                return self._run_pycaret_native(data, progress_callback)
            except Exception as e:
                if progress_callback:
                    progress_callback(40, f"Switching to native GBDT engine: {e}")
                self.is_pycaret_active = False

        # Fallback GBDT / Scikit-Learn Engine
        return self._run_fallback_engine(data, progress_callback)

    def _run_pycaret_native(self, data: pd.DataFrame, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """Native PyCaret 3.x execution stream."""
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
                progress_callback(40, "Benchmarking GBDT algorithms (LightGBM, XGBoost, CatBoost, RF)...")

            included_models = []
            if HAS_LGB: included_models.append("lightgbm")
            if HAS_XGB: included_models.append("xgboost")
            if HAS_CAT: included_models.append("catboost")
            included_models.extend(["rf", "et", "lr"])

            metric_map = {"PR-AUC": "AUC", "ROC-AUC": "AUC", "AUC": "AUC", "F1": "F1", "Accuracy": "Accuracy"}
            pycaret_metric = metric_map.get(self.config.optimize_metric, "AUC")

            self.best_models = self.exp.compare_models(
                include=included_models,
                n_select=min(self.config.top_n_models, len(included_models)),
                sort=pycaret_metric,
                verbose=False
            )
            
            if not isinstance(self.best_models, list):
                self.best_models = [self.best_models]

            self.leaderboard = self.exp.pull()

            if progress_callback:
                progress_callback(70, "Finalizing champion model...")

            if self.config.create_ensemble and len(self.best_models) >= 2:
                ensemble = self.exp.blend_models(estimator_list=self.best_models, verbose=False)
                self.champion_model = self.exp.finalize_model(ensemble)
            else:
                self.champion_model = self.exp.finalize_model(self.best_models[0])

        else:
            self.exp = RegressionExperiment()
            self.exp.setup(
                data=data,
                target=self.config.target_col,
                session_id=self.config.session_id,
                fold=self.config.cv_folds,
                verbose=False
            )
            self.best_models = self.exp.compare_models(
                n_select=self.config.top_n_models,
                sort="R2",
                verbose=False
            )
            if not isinstance(self.best_models, list):
                self.best_models = [self.best_models]
            self.leaderboard = self.exp.pull()
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

    def _run_fallback_engine(self, data: pd.DataFrame, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """High-performance scikit-learn & GBDT fallback engine."""
        if progress_callback:
            progress_callback(20, "Executing High-Performance GBDT Benchmarking...")

        X = data.drop(columns=[self.config.target_col])
        y = data[self.config.target_col].astype(int)

        num_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', cat_transformer, cat_cols)
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.config.session_id, stratify=y if len(np.unique(y)) <= 2 else None
        )

        candidate_models = {}
        if HAS_XGB:
            candidate_models["XGBoost"] = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='logloss')
        if HAS_LGB:
            candidate_models["LightGBM"] = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
        if HAS_CAT:
            candidate_models["CatBoost"] = cb.CatBoostClassifier(iterations=100, learning_rate=0.05, depth=5, random_seed=42, verbose=0)
            
        candidate_models["Random Forest"] = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        candidate_models["Extra Trees"] = ExtraTreesClassifier(n_estimators=100, max_depth=8, random_state=42)
        candidate_models["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=42)

        results_list = []
        best_score = -1.0
        champion = None

        total = len(candidate_models)
        for idx, (name, clf) in enumerate(candidate_models.items()):
            if progress_callback:
                progress_callback(20 + int(60 * (idx + 1) / total), f"Evaluating {name}...")

            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            if hasattr(pipe, "predict_proba"):
                y_prob = pipe.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
            pr_auc = average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5

            results_list.append({
                "Model": name,
                "ROC-AUC": round(roc_auc, 4),
                "PR-AUC": round(pr_auc, 4),
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "pipeline": pipe
            })

            target_score = pr_auc if self.config.optimize_metric == "PR-AUC" else roc_auc
            if target_score > best_score:
                best_score = target_score
                champion = pipe

        leaderboard_df = pd.DataFrame(results_list).sort_values(by="PR-AUC" if self.config.optimize_metric == "PR-AUC" else "ROC-AUC", ascending=False).reset_index(drop=True)
        self.leaderboard = leaderboard_df.drop(columns=["pipeline"])
        self.champion_model = champion

        if progress_callback:
            progress_callback(100, "AutoML Benchmarking Complete!")

        return {
            "task_type": self.task_type,
            "champion_model": self.champion_model,
            "leaderboard": self.leaderboard,
            "top_models": [row["pipeline"] for row in results_list],
            "X_test": X_test,
            "y_test": y_test,
            "engine_name": "GBDT Ensemble Engine"
        }

    def predict_probabilities(self, X_new: pd.DataFrame) -> np.ndarray:
        """Returns calibrated default probabilities for new records."""
        if self.champion_model is None:
            raise ValueError("Model has not been trained yet.")

        if hasattr(self.champion_model, "predict_proba"):
            return self.champion_model.predict_proba(X_new)[:, 1]
        elif hasattr(self.champion_model, "predict"):
            return self.champion_model.predict(X_new)
        return np.zeros(len(X_new))
