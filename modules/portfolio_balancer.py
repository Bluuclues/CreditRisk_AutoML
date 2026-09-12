"""
portfolio_balancer.py
Automated Portfolio Segmentation & Ratio Balancing Engine.
Performs adaptive tertile loan ticket sizing, tenor categorization,
and proportional stratified defaulter cutoff sampling.
"""

from typing import Tuple, Dict, Any, Optional, List
import pandas as pd
import numpy as np


def _stratified_sample(
    df_subset: pd.DataFrame, 
    n_samples: int, 
    stratify_col: Optional[str], 
    random_state: int
) -> pd.DataFrame:
    """
    Proportionally samples n_samples records from df_subset stratified across categories in stratify_col.
    Guarantees exact quota allocation matching n_samples without dropping or overflowing categories.
    """
    if n_samples >= len(df_subset):
        return df_subset
    if n_samples <= 0:
        return df_subset.head(0)
    if not stratify_col or stratify_col not in df_subset.columns or df_subset[stratify_col].nunique() <= 1:
        return df_subset.sample(n=n_samples, random_state=random_state)
    
    quotas = {}
    total_avail = len(df_subset)
    allocated = 0
    categories = list(df_subset[stratify_col].unique())
    
    for cat in categories:
        cat_count = len(df_subset[df_subset[stratify_col] == cat])
        q = int(round(n_samples * (cat_count / total_avail)))
        q = max(0, min(q, cat_count))
        quotas[cat] = q
        allocated += q
        
    diff = n_samples - allocated
    if diff != 0:
        if diff > 0:
            for cat in categories:
                cat_count = len(df_subset[df_subset[stratify_col] == cat])
                can_add = cat_count - quotas[cat]
                if can_add > 0:
                    add_n = min(diff, can_add)
                    quotas[cat] += add_n
                    diff -= add_n
                    if diff == 0:
                        break
        else:
            for cat in categories:
                can_sub = quotas[cat]
                if can_sub > 0:
                    sub_n = min(-diff, can_sub)
                    quotas[cat] -= sub_n
                    diff += sub_n
                    if diff == 0:
                        break
                        
    sampled_parts = []
    for i, cat in enumerate(categories):
        sub_df = df_subset[df_subset[stratify_col] == cat]
        q = quotas[cat]
        if q > 0:
            sampled_parts.append(sub_df.sample(n=q, random_state=random_state + i))
            
    if sampled_parts:
        return pd.concat(sampled_parts).sort_index()
    return df_subset.sample(n=n_samples, random_state=random_state)


def generate_automated_segments(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Automates portfolio segmentation based on loan ticket size (principal amount)
    and loan tenor/duration, as well as detecting categorical segment columns.
    Adds 'loan_ticket_segment' (and 'loan_tenor_segment' if applicable) to the DataFrame.
    Returns the enriched DataFrame and a metadata dictionary containing segment dimensions and statistics.
    """
    enriched_df = df.copy()
    meta = {
        "dimensions": {},
        "default_dimension": None
    }
    
    # 1. Ticket Size (Principal Amount)
    amount_col = None
    for cand in ['amount', 'loan_amount', 'principal', 'principal_amount', 'disbursed_amount', 'loan_amt']:
        if cand in enriched_df.columns:
            amount_col = cand
            break
            
    if amount_col is not None:
        numeric_amt = pd.to_numeric(enriched_df[amount_col], errors='coerce').fillna(0)
        enriched_df[amount_col] = numeric_amt
        pos_amt = numeric_amt[numeric_amt > 0]
        
        if len(pos_amt) >= 3:
            q33 = float(pos_amt.quantile(0.333))
            q66 = float(pos_amt.quantile(0.666))
            
            def clean_round(val: float) -> int:
                if val >= 50000:
                    return int(round(val / 5000.0) * 5000)
                elif val >= 10000:
                    return int(round(val / 1000.0) * 1000)
                elif val >= 1000:
                    return int(round(val / 500.0) * 500)
                elif val >= 100:
                    return int(round(val / 50.0) * 50)
                else:
                    return int(round(val))
                    
            t1 = clean_round(q33)
            t2 = clean_round(q66)
            if t2 <= t1:
                t2 = t1 + (1000 if t1 >= 1000 else 100)
                
            labels = [
                f"Micro-Ticket (<= KES {t1:,})",
                f"Mid-Ticket (KES {t1+1:,} - {t2:,})",
                f"Large-Ticket (> KES {t2:,})"
            ]
            
            conditions = [
                numeric_amt <= t1,
                (numeric_amt > t1) & (numeric_amt <= t2),
                numeric_amt > t2
            ]
            enriched_df['loan_ticket_segment'] = np.select(conditions, labels, default=labels[0])
        else:
            enriched_df['loan_ticket_segment'] = "Standard Ticket"
            labels = ["Standard Ticket"]
            
        breakdown = []
        for lbl in labels:
            sub = enriched_df[enriched_df['loan_ticket_segment'] == lbl]
            if len(sub) > 0:
                tot = len(sub)
                d_cnt = int(sub['default_flag'].sum()) if 'default_flag' in sub.columns else 0
                rate = (d_cnt / tot * 100.0) if tot > 0 else 0.0
                breakdown.append({
                    "segment": lbl,
                    "count": tot,
                    "defaulters": d_cnt,
                    "performing": tot - d_cnt,
                    "default_rate": round(rate, 1)
                })
                
        meta["dimensions"]["Ticket Size (Principal Amount)"] = {
            "col": "loan_ticket_segment",
            "source_col": amount_col,
            "breakdown": breakdown
        }
        meta["default_dimension"] = "Ticket Size (Principal Amount)"

    # 2. Tenor / Duration
    tenure_col = None
    for cand in ['tenure_days', 'term_days', 'tenor', 'tenure']:
        if cand in enriched_df.columns:
            tenure_col = cand
            break
            
    if tenure_col is not None:
        numeric_ten = pd.to_numeric(enriched_df[tenure_col], errors='coerce').fillna(30)
        enriched_df[tenure_col] = numeric_ten
        
        ten_labels = [
            "Short-Tenor (<= 30 Days)",
            "Medium-Tenor (31 - 90 Days)",
            "Long-Tenor (> 90 Days)"
        ]
        conditions = [
            numeric_ten <= 30,
            (numeric_ten > 30) & (numeric_ten <= 90),
            numeric_ten > 90
        ]
        enriched_df['loan_tenor_segment'] = np.select(conditions, ten_labels, default=ten_labels[0])
        
        ten_breakdown = []
        for lbl in ten_labels:
            sub = enriched_df[enriched_df['loan_tenor_segment'] == lbl]
            if len(sub) > 0:
                tot = len(sub)
                d_cnt = int(sub['default_flag'].sum()) if 'default_flag' in sub.columns else 0
                rate = (d_cnt / tot * 100.0) if tot > 0 else 0.0
                ten_breakdown.append({
                    "segment": lbl,
                    "count": tot,
                    "defaulters": d_cnt,
                    "performing": tot - d_cnt,
                    "default_rate": round(rate, 1)
                })
                
        meta["dimensions"]["Loan Tenor (Duration)"] = {
            "col": "loan_tenor_segment",
            "source_col": tenure_col,
            "breakdown": ten_breakdown
        }
        if meta["default_dimension"] is None:
            meta["default_dimension"] = "Loan Tenor (Duration)"

    # 3. Existing categorical columns
    for cand in ['borrower_type', 'loan_type', 'sector', 'business_category']:
        if cand in enriched_df.columns:
            uniq_vals = enriched_df[cand].dropna().unique()
            if 1 < len(uniq_vals) <= 10:
                c_breakdown = []
                for val in uniq_vals:
                    val_str = str(val)
                    sub = enriched_df[enriched_df[cand] == val]
                    tot = len(sub)
                    d_cnt = int(sub['default_flag'].sum()) if 'default_flag' in sub.columns else 0
                    rate = (d_cnt / tot * 100.0) if tot > 0 else 0.0
                    c_breakdown.append({
                        "segment": val_str,
                        "count": tot,
                        "defaulters": d_cnt,
                        "performing": tot - d_cnt,
                        "default_rate": round(rate, 1)
                    })
                dim_title = cand.replace('_', ' ').title()
                meta["dimensions"][dim_title] = {
                    "col": cand,
                    "source_col": cand,
                    "breakdown": c_breakdown
                }

    return enriched_df, meta


def balance_portfolio_by_defaulter_pct(
    df: pd.DataFrame, 
    target_pct: Optional[float], 
    stratify_col: Optional[str] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Adjusts the portfolio DataFrame so that defaulters (default_flag == 1) make up 
    exactly target_pct% of the analytical dataset, cutting off excess records.
    If stratify_col is provided, samples non-defaulters proportionally across segments.
    If target_pct is None or <= 0 or >= 100, returns the dataset untouched.
    """
    if 'default_flag' not in df.columns:
        return df, {
            "original_total": len(df), "original_def": 0, "original_non_def": len(df),
            "kept_def": 0, "kept_non_def": len(df), "kept_total": len(df),
            "cut_off": 0, "target_pct": 0.0, "actual_pct": 0.0, "segment_breakdown": []
        }

    df_clean = df.copy()
    df_clean['default_flag'] = pd.to_numeric(df_clean['default_flag'], errors='coerce').fillna(0).astype(int)

    defaulters = df_clean[df_clean['default_flag'] == 1]
    non_defaulters = df_clean[df_clean['default_flag'] == 0]

    n_def = len(defaulters)
    n_non_def = len(non_defaulters)
    n_total = len(df_clean)
    orig_pct = (n_def / n_total * 100.0) if n_total > 0 else 0.0

    if target_pct is None or target_pct <= 0 or target_pct >= 100 or n_def == 0 or n_non_def == 0:
        return df_clean, {
            "original_total": n_total,
            "original_def": n_def,
            "original_non_def": n_non_def,
            "kept_def": n_def,
            "kept_non_def": n_non_def,
            "kept_total": n_total,
            "cut_off": 0,
            "target_pct": round(target_pct if target_pct is not None else orig_pct, 1),
            "actual_pct": round(orig_pct, 2),
            "segment_breakdown": []
        }

    p = float(target_pct) / 100.0

    # Desired equation: kept_def / (kept_def + kept_non_def) = p
    # Try keeping all defaulters and cutting off excess non-defaulters:
    needed_non_def = int(round(n_def * (1.0 - p) / p))

    if 0 < needed_non_def <= n_non_def:
        kept_def_df = defaulters
        kept_non_def_df = _stratified_sample(non_defaulters, needed_non_def, stratify_col, random_state)
    else:
        # If target default % is higher than available non-defaulters can support or inverted:
        needed_def = int(round(n_non_def * p / (1.0 - p)))
        needed_def = max(1, min(needed_def, n_def))
        kept_def_df = _stratified_sample(defaulters, needed_def, stratify_col, random_state)
        kept_non_def_df = non_defaulters

    balanced_df = pd.concat([kept_def_df, kept_non_def_df]).sort_index()
    actual_pct = (len(kept_def_df) / len(balanced_df) * 100.0) if len(balanced_df) > 0 else 0.0

    seg_breakdown = []
    if stratify_col and stratify_col in balanced_df.columns:
        for cat in balanced_df[stratify_col].unique():
            cat_sub = balanced_df[balanced_df[stratify_col] == cat]
            c_tot = len(cat_sub)
            c_def = int(cat_sub['default_flag'].sum())
            seg_breakdown.append({
                "segment": str(cat),
                "count": c_tot,
                "defaulters": c_def,
                "performing": c_tot - c_def,
                "default_rate": round(c_def / c_tot * 100.0, 1) if c_tot > 0 else 0.0
            })

    stats = {
        "original_total": n_total,
        "original_def": n_def,
        "original_non_def": n_non_def,
        "kept_def": len(kept_def_df),
        "kept_non_def": len(kept_non_def_df),
        "kept_total": len(balanced_df),
        "cut_off": n_total - len(balanced_df),
        "target_pct": round(target_pct, 1),
        "actual_pct": round(actual_pct, 2),
        "segment_breakdown": seg_breakdown
    }
    return balanced_df, stats
