# 🧩 Modules

This directory contains the core functional components and business logic for the KBA Credit Risk AutoML Engine. 

## Components

- **`ai_assistant.py`**: Handles AI-driven insights and interactive assistance for the user interface.
- **`auth.py`**: Manages cryptographic pseudonymization, secure user authentication, and role-based access.
- **`data_validator.py`**: Implements strict data quality gates and manages the Dead-Letter Queue (DLQ) for quarantining malformed records.
- **`eda_visualizer.py`**: Responsible for rendering interactive Exploratory Data Analysis (EDA) visualizations and portfolio insights.
- **`feature_store.py`**: Manages the ingestion, transformation, and storage of layered alternative and traditional features.
- **`iv_engine.py`**: Calculates Information Value (IV) and Weight of Evidence (WoE) for feature selection and predictive power assessment.
- **`login_page.py`**: Handles the UI and backend logic for the application's secure login screen.
- **`portfolio_balancer.py`**: Implements techniques for handling imbalanced credit portfolios to optimize PR-AUC and Gini metrics.
- **`models/`**: Subdirectory containing individual model architectures, ensemble builders, and evaluation logic.
