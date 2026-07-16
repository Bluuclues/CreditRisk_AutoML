import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

def train_and_evaluate(df):
    # 1. Define Target and Features
    if 'default_flag' not in df.columns:
        raise ValueError("Critical Error: 'default_flag' column missing. Cannot train model.")
        
    # --- NEW: Drop any rows where the default_flag is missing ---
    # You cannot train a supervised ML model if the answer is blank!
    df = df.dropna(subset=['default_flag'])

    # Drop columns that have no predictive value or would cause data leakage
    cols_to_drop = ['default_flag', 'session_id', 'borrower_id', 'loan_no', 'loan_date', 'payoff_date', 'country_code','due_date']
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = df['default_flag'].astype(int) 
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category', 'string']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probability of Default (PD)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0)
    }

    return {
        'model': model, 
        'metrics': metrics, 
        'X_test': X_test, 
        'y_test': y_test, 
        'y_pred': y_pred,
        'y_prob': y_prob
    }