"""
Credit Analyze - User Authentication & Isolated Data Store Manager
==================================================================
Handles cryptographic user registration, PBKDF2 password hashing with individual salts,
immediate access provisioning, survey/interview consent tracking, and Google Sheets persistence.
Stored via Streamlit's st-gsheets-connection to ensure cloud persistence.
"""

import os
import hashlib
import secrets
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple, List

try:
    from streamlit_gsheets import GSheetsConnection
except ImportError:
    GSheetsConnection = None

# Cryptographic parameters
HASH_NAME = "sha256"
ITERATIONS = 100_000

# Required Columns
DB_COLUMNS = [
    "id", "email", "password_hash", "salt", "other_details",
    "agreed_terms", "terms_version", "terms_timestamp",
    "survey_interview_consent", "access_status", "role",
    "created_at", "last_login"
]


def _get_gsheets_connection():
    """Returns a connection to the Google Sheets users database via st.connection."""
    if GSheetsConnection is None:
        raise ImportError("st-gsheets-connection is not installed. Please add it to requirements.txt")
    
    return st.connection("gsheets", type=GSheetsConnection)


def init_auth_db() -> None:
    """Initializes the users sheet if it does not already exist."""
    conn = _get_gsheets_connection()
    try:
        df = conn.read(ttl=0)
        # If the dataframe is completely empty or missing our core columns, initialize it
        if df.empty or 'email' not in df.columns:
            empty_df = pd.DataFrame(columns=DB_COLUMNS)
            conn.update(data=empty_df)
    except Exception as e:
        # If read fails (e.g. brand new uninitialized sheet), we force update with headers
        empty_df = pd.DataFrame(columns=DB_COLUMNS)
        conn.update(data=empty_df)


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hashes a password using PBKDF2-HMAC-SHA256 with a unique salt.
    Returns (hex_hash, hex_salt).
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    hash_bytes = hashlib.pbkdf2_hmac(
        HASH_NAME,
        password.encode("utf-8"),
        bytes.fromhex(salt),
        ITERATIONS
    )
    return hash_bytes.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies a password against the stored PBKDF2 hash using constant-time comparison."""
    calculated_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(calculated_hash, stored_hash)


def register_user(
    email: str,
    password: str,
    other_details: str = "",
    agreed_terms: bool = True,
    survey_consent: bool = True,
    role: str = "user"
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Registers a new user into the Google Sheets database.
    Per user request: No tail gate for now - immediate access is granted!
    """
    init_auth_db()
    
    clean_email = email.strip().lower()
    if not clean_email or "@" not in clean_email:
        return False, "Please provide a valid email address.", None
        
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long.", None
        
    if not agreed_terms:
        return False, "You must accept the Terms and Conditions to proceed.", None
        
    if not survey_consent:
        return False, "You must consent to participating in research surveys and interviews.", None

    conn = _get_gsheets_connection()
    df = conn.read(ttl=0)
    
    # Check if email already exists
    if not df.empty and 'email' in df.columns:
        if clean_email in df['email'].astype(str).str.lower().values:
            return False, f"An account with email '{clean_email}' already exists. Please log in.", None

    pwd_hash, salt = hash_password(password)
    now_iso = datetime.now(timezone(timedelta(hours=3))).isoformat()
    
    # Calculate new ID safely
    if df.empty or 'id' not in df.columns or df['id'].dropna().empty:
        new_id = 1
    else:
        try:
            new_id = int(pd.to_numeric(df['id']).max()) + 1
        except Exception:
            new_id = 1
    
    new_row = pd.DataFrame([{
        "id": new_id,
        "email": clean_email,
        "password_hash": pwd_hash,
        "salt": salt,
        "other_details": other_details.strip(),
        "agreed_terms": 1 if agreed_terms else 0,
        "terms_version": "v1.0",
        "terms_timestamp": now_iso,
        "survey_interview_consent": 1 if survey_consent else 0,
        "access_status": "granted",
        "role": role,
        "created_at": now_iso,
        "last_login": now_iso
    }])
    
    # Append and update Google Sheet
    df_updated = pd.concat([df, new_row], ignore_index=True)
    conn.update(data=df_updated)
        
    user_dict = {
        "id": new_id,
        "email": clean_email,
        "other_details": other_details.strip(),
        "access_status": "granted",
        "role": role,
        "survey_interview_consent": True,
        "created_at": now_iso
    }
    return True, "Sign up successful! Immediate access granted.", user_dict


def authenticate_user(email: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validates user credentials against Google Sheets users.db and verifies access status.
    Returns (success, message, user_dict).
    """
    init_auth_db()
    
    clean_email = email.strip().lower()
    if not clean_email or not password:
        return False, "Email and password are required.", None

    conn = _get_gsheets_connection()
    df = conn.read(ttl=0)
    
    if df.empty or 'email' not in df.columns:
        return False, "Invalid email or password.", None
        
    # Find user row
    user_df = df[df['email'].astype(str).str.lower() == clean_email]
    
    if user_df.empty:
        return False, "Invalid email or password.", None
        
    user_row = user_df.iloc[0]

    stored_hash = str(user_row.get("password_hash", ""))
    salt = str(user_row.get("salt", ""))
    access_status = str(user_row.get("access_status", ""))

    if not stored_hash or not salt or not verify_password(password, stored_hash, salt):
        return False, "Invalid email or password.", None

    # Check access status
    if access_status != "granted":
        return False, f"Account access is currently '{access_status}'. Please contact the KBA administrator.", None

    # Update last login timestamp
    now_iso = datetime.now(timezone(timedelta(hours=3))).isoformat()
    
    # Ensure index alignment to update the exact row
    df.loc[df['email'].astype(str).str.lower() == clean_email, 'last_login'] = now_iso
    conn.update(data=df)

    user_dict = {
        "id": int(user_row.get("id", 0)),
        "email": str(user_row.get("email", "")),
        "other_details": str(user_row.get("other_details", "")),
        "access_status": access_status,
        "role": str(user_row.get("role", "")),
        "survey_interview_consent": bool(user_row.get("survey_interview_consent", False)),
        "created_at": str(user_row.get("created_at", "")),
        "last_login": now_iso
    }
    return True, "Login successful! Welcome back to Credit Analyze.", user_dict


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Retrieves safe user metadata by email."""
    init_auth_db()
    conn = _get_gsheets_connection()
    df = conn.read(ttl=0)
    
    if df.empty or 'email' not in df.columns:
        return None
        
    clean_email = email.strip().lower()
    user_df = df[df['email'].astype(str).str.lower() == clean_email]
    
    if user_df.empty:
        return None
        
    row = user_df.iloc[0]
    return {
        "id": int(row.get("id", 0)),
        "email": str(row.get("email", "")),
        "other_details": str(row.get("other_details", "")),
        "access_status": str(row.get("access_status", "")),
        "role": str(row.get("role", "")),
        "survey_interview_consent": bool(row.get("survey_interview_consent", False)),
        "created_at": str(row.get("created_at", "")),
        "last_login": str(row.get("last_login", ""))
    }


def get_all_users() -> List[Dict[str, Any]]:
    """Retrieves all registered users (excluding password hashes and salts) for admin review."""
    init_auth_db()
    conn = _get_gsheets_connection()
    df = conn.read(ttl=0)
    
    if df.empty or 'email' not in df.columns:
        return []
        
    # Sort by created_at DESC
    if 'created_at' in df.columns:
        df = df.sort_values(by='created_at', ascending=False)
        
    users_list = []
    for _, row in df.iterrows():
        users_list.append({
            "id": int(row.get("id", 0)),
            "email": str(row.get("email", "")),
            "other_details": str(row.get("other_details", "")),
            "access_status": str(row.get("access_status", "")),
            "role": str(row.get("role", "")),
            "survey_interview_consent": bool(row.get("survey_interview_consent", False)),
            "agreed_terms": bool(row.get("agreed_terms", False)),
            "created_at": str(row.get("created_at", "")),
            "last_login": str(row.get("last_login", ""))
        })
    return users_list


def update_user_email(user_id: int, new_email: str) -> Tuple[bool, str]:
    """Updates the user's email in the Google Sheets database."""
    init_auth_db()
    clean_email = new_email.strip().lower()
    if not clean_email or "@" not in clean_email:
        return False, "Please provide a valid email address."
        
    conn = _get_gsheets_connection()
    df = conn.read(ttl=0)
    
    if df.empty or 'email' not in df.columns:
        return False, "Database error."
        
    # Check if new email is already taken by another user
    existing_users = df[df['email'].astype(str).str.lower() == clean_email]
    if not existing_users.empty and existing_users.iloc[0]['id'] != user_id:
        return False, f"An account with email '{clean_email}' already exists."

    # Update email for the given user_id
    mask = df['id'] == user_id
    if not mask.any():
        return False, "User not found."
        
    df.loc[mask, 'email'] = clean_email
    conn.update(data=df)
    return True, "Email updated successfully."


def delete_user_account(user_id: int) -> Tuple[bool, str]:
    """Opts the user out of the prototype by deleting their account from Google Sheets."""
    init_auth_db()
    conn = _get_gsheets_connection()
    df = conn.read(ttl=0)
    
    if df.empty or 'id' not in df.columns:
        return False, "Database error."
        
    mask = df['id'] == user_id
    if not mask.any():
        return False, "User not found."
        
    # Remove the user's row
    df = df[~mask]
    conn.update(data=df)
    return True, "Account deleted successfully."
