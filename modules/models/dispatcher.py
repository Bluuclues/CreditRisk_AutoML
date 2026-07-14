def run_models(df, selected_models):
    """
    Routes the training request to the selected ML model scripts.
    We use lazy-importing inside the function to prevent Streamlit 
    from crashing during its initial boot sequence.
    """
    results = {}
    
    if "XGBoost Classifier" in selected_models:
        from modules.models import xgboost_model
        results["XGBoost Classifier"] = xgboost_model.train_and_evaluate(df)
        
    if "Random Forest" in selected_models:
        from modules.models import random_forest_model
        results["Random Forest"] = random_forest_model.train_and_evaluate(df)
        
    return results