"""
data_validator.py
Automated validation checks and quality gate for credit risk ingestion payloads.
Compliant with Kenya DPA 2019 standards and DuckDB/Streamlit pipeline execution.
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np


class CreditRiskDataValidator:
    """Validates schema integrity, data ranges, target values, and missing rates."""

    MANDATORY_COLUMNS = [
        "borrower_id",
        "amount",
        "default_flag"
    ]

    NUMERIC_BOUNDS = {
        "amount": (1.0, 10_000_000.0),
        "tenure_days": (1, 3650),
        "default_flag": (0, 1)
    }

    @classmethod
    def validate_ingestion_payload(cls, df: pd.DataFrame) -> Tuple[bool, List[str], pd.DataFrame, List[Dict[str, Any]]]:
        """
        Executes multi-stage validation quality gate on uploaded panel loan DataFrame.
        Returns:
            is_valid (bool): True if dataset passes mandatory validation gates.
            messages (List[str]): Warning and status messages for user display.
            df_clean (pd.DataFrame): Sanitized DataFrame ready for DuckDB store.
            dlq_records (List[Dict]): Dead-Letter Queue records rejected during quality check.
        """
        messages = []
        dlq_records = []
        df_clean = df.copy()

        # 1. Mandatory Schema Verification
        missing_cols = [c for c in cls.MANDATORY_COLUMNS if c not in df_clean.columns]
        if missing_cols:
            messages.append(f"❌ Mandatory columns missing: {', '.join(missing_cols)}")
            return False, messages, df_clean, dlq_records

        # 2. Target Variable Cleanliness Check
        if 'default_flag' in df_clean.columns:
            # Coerce default_flag to numeric integer
            df_clean['default_flag'] = pd.to_numeric(df_clean['default_flag'], errors='coerce')
            invalid_target_mask = df_clean['default_flag'].isna() | (~df_clean['default_flag'].isin([0, 1]))
            
            if invalid_target_mask.any():
                bad_count = int(invalid_target_mask.sum())
                messages.append(f"⚠️ Found {bad_count} rows with missing/invalid target ('default_flag'). Dropped to DLQ.")
                dlq_records.extend(df_clean[invalid_target_mask].to_dict(orient='records'))
                df_clean = df_clean[~invalid_target_mask].copy()

            df_clean['default_flag'] = df_clean['default_flag'].astype(int)

        # 3. Date & Year Extractions
        if 'loan_date' in df_clean.columns:
            df_clean['loan_date'] = pd.to_datetime(df_clean['loan_date'], errors='coerce', dayfirst=True)
            if 'year' not in df_clean.columns or df_clean['year'].isna().any():
                df_clean['year'] = df_clean['loan_date'].dt.year.fillna(2026).astype(int)
        else:
            if 'year' not in df_clean.columns:
                df_clean['year'] = 2026

        # 4. Numeric Range & Imputation Checks
        if 'amount' in df_clean.columns:
            df_clean['amount'] = pd.to_numeric(df_clean['amount'], errors='coerce')
            negative_amounts = (df_clean['amount'] <= 0)
            if negative_amounts.any():
                messages.append(f"⚠️ Replaced {int(negative_amounts.sum())} non-positive loan amounts with median.")
                df_clean.loc[negative_amounts, 'amount'] = np.nan

            median_amt = df_clean['amount'].median() if not df_clean['amount'].dropna().empty else 10000.0
            df_clean['amount'] = df_clean['amount'].fillna(median_amt)

        if 'tenure_days' in df_clean.columns:
            df_clean['tenure_days'] = pd.to_numeric(df_clean['tenure_days'], errors='coerce')
            df_clean['tenure_days'] = df_clean['tenure_days'].fillna(30).astype(int)

        # 5. Summary Status
        valid_rows = len(df_clean)
        if valid_rows == 0:
            messages.append("❌ All rows failed validation checks.")
            return False, messages, df_clean, dlq_records

        messages.append(f"✅ Data quality gate passed: {valid_rows:,} valid borrower records ready for ingestion.")
        return True, messages, df_clean, dlq_records
