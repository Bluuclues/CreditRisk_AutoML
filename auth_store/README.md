# Credit Analyze Authentication Data Store

This directory contains the user authentication database for the **Credit Analyze** prototype.

## Security & Privacy Policies
- **Git Excluded**: This directory contains an isolated `.gitignore` to prevent any database files (`*.db`, `*.sqlite`) or credential secrets from ever being tracked or committed to version control.
- **Cryptographic Hashing**: User passwords are never stored in plaintext. They are hashed using **PBKDF2-HMAC-SHA256** with 100,000 iterations and uniquely generated cryptographically secure salts (`secrets.token_hex(16)`).
- **Access Control**: Users registering for the prototype are assigned `access_status = 'granted'` immediately, with records of their survey & interview consent in compliance with user testing terms and the Kenya Data Protection Act (DPA 2019).

## Database Schema (`users` table)
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER PRIMARY KEY | Unique user identifier |
| `email` | TEXT UNIQUE | Registered email address (case-insensitive) |
| `password_hash` | TEXT | PBKDF2-HMAC-SHA256 hex digest |
| `salt` | TEXT | Cryptographic hex salt |
| `other_details` | TEXT | Organization, role, and tester details |
| `agreed_terms` | INTEGER | 1 if agreed to terms and conditions |
| `survey_interview_consent`| INTEGER | 1 if consented to feedback surveys and interviews |
| `access_status` | TEXT | Access grant status (`granted`, `pending`, `revoked`) |
| `role` | TEXT | User role (`user`, `admin`, `tester`) |
| `created_at` | TIMESTAMP | Registration timestamp |
| `last_login` | TIMESTAMP | Most recent successful login timestamp |
