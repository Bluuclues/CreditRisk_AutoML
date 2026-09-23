"""
Credit Analyze - User Authentication & Isolated Data Store Manager
==================================================================
Handles cryptographic user registration, PBKDF2 password hashing with individual salts,
immediate access provisioning, survey/interview consent tracking, and SQLite persistence.
Stored in an isolated directory with its own .gitignore to protect privacy and credentials.
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

# Path to the isolated authentication database
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MODULE_DIR)
AUTH_STORE_DIR = os.path.join(BASE_DIR, "auth_store")
DB_PATH = os.path.join(AUTH_STORE_DIR, "users.db")

# Cryptographic parameters
HASH_NAME = "sha256"
ITERATIONS = 100_000


def get_db_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite users database, creating directory if missing."""
    os.makedirs(AUTH_STORE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db() -> None:
    """Initializes the users table if it does not already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL COLLATE NOCASE,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        other_details TEXT,
        agreed_terms INTEGER NOT NULL DEFAULT 1,
        terms_version TEXT DEFAULT 'v1.0',
        terms_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        survey_interview_consent INTEGER NOT NULL DEFAULT 1,
        access_status TEXT NOT NULL DEFAULT 'granted',
        role TEXT DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
    conn.commit()
    conn.close()


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
    Registers a new user into the database.
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

    pwd_hash, salt = hash_password(password)
    now_iso = datetime.utcnow().isoformat()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO users (
            email, password_hash, salt, other_details, 
            agreed_terms, terms_version, terms_timestamp,
            survey_interview_consent, access_status, role, 
            created_at, last_login
        ) VALUES (?, ?, ?, ?, ?, 'v1.0', ?, ?, 'granted', ?, ?, ?)
        """, (
            clean_email, pwd_hash, salt, other_details.strip(),
            1 if agreed_terms else 0,
            now_iso,
            1 if survey_consent else 0,
            role,
            now_iso,
            now_iso
        ))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        user_dict = {
            "id": user_id,
            "email": clean_email,
            "other_details": other_details.strip(),
            "access_status": "granted",
            "role": role,
            "survey_interview_consent": True,
            "created_at": now_iso
        }
        return True, "Sign up successful! Immediate access granted.", user_dict
        
    except sqlite3.IntegrityError:
        conn.close()
        return False, f"An account with email '{clean_email}' already exists. Please log in.", None
    except Exception as e:
        conn.close()
        return False, f"Database error: {str(e)}", None


def authenticate_user(email: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validates user credentials against users.db and verifies access status.
    Returns (success, message, user_dict).
    """
    init_auth_db()
    
    clean_email = email.strip().lower()
    if not clean_email or not password:
        return False, "Email and password are required.", None

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, email, password_hash, salt, other_details, 
           access_status, role, survey_interview_consent, created_at
    FROM users 
    WHERE email = ?
    """, (clean_email,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return False, "Invalid email or password.", None

    stored_hash = row["password_hash"]
    salt = row["salt"]
    access_status = row["access_status"]

    if not verify_password(password, stored_hash, salt):
        conn.close()
        return False, "Invalid email or password.", None

    # Check access status (currently default is 'granted', but supports future access gates)
    if access_status != "granted":
        conn.close()
        return False, f"Account access is currently '{access_status}'. Please contact the KBA administrator.", None

    # Update last login timestamp
    now_iso = datetime.utcnow().isoformat()
    cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (now_iso, row["id"]))
    conn.commit()
    conn.close()

    user_dict = {
        "id": row["id"],
        "email": row["email"],
        "other_details": row["other_details"],
        "access_status": access_status,
        "role": row["role"],
        "survey_interview_consent": bool(row["survey_interview_consent"]),
        "created_at": row["created_at"],
        "last_login": now_iso
    }
    return True, "Login successful! Welcome back to Credit Analyze.", user_dict


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Retrieves safe user metadata by email."""
    init_auth_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, email, other_details, access_status, role, 
           survey_interview_consent, created_at, last_login
    FROM users 
    WHERE email = ?
    """, (email.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_all_users() -> List[Dict[str, Any]]:
    """Retrieves all registered users (excluding password hashes and salts) for admin review."""
    init_auth_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, email, other_details, access_status, role, 
           survey_interview_consent, agreed_terms, created_at, last_login
    FROM users 
    ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
