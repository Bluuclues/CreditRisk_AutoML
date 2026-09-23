"""
Credit Analyze - Custom Authentication UI Page
==============================================
Renders the branded Login & Sign-Up interface conforming to the official brand board:
- Fonts: DM Sans for headings, Century Gothic for body text
- Palette: Terracotta / Ochre / Slate Blue / White / Charcoal
- Brand Logo & 5-Stage Methodology Architecture
- Mobile-friendly responsive view with 5 split methodology cards
- Immediate access provisioning upon signup with survey & interview consent
"""

import os
import base64
import streamlit as st
from typing import Optional, Dict, Any

from modules.auth import register_user, authenticate_user, init_auth_db

# Assets paths
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MODULE_DIR)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
METHODOLOGY_PATH = os.path.join(ASSETS_DIR, "methodology.png")


def get_base64_image(image_path: str) -> str:
    """Reads an image file and returns base64 string for clean HTML rendering."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(image_path)[1].lstrip(".").lower()
            mime = "image/png" if ext == "png" else "image/svg+xml" if ext == "svg" else "image/jpeg"
            return f"data:{mime};base64,{encoded}"
    return ""


def render_login_signup_page() -> None:
    """Renders the complete Login & Sign Up page matching the brand design and mobile responsive requirements."""
    init_auth_db()
    
    # Load base64 assets
    logo_b64 = get_base64_image(LOGO_PATH)
    methodology_b64 = get_base64_image(METHODOLOGY_PATH)

    # Global container
    st.markdown('<div class="auth-page-root">', unsafe_allow_html=True)

    # Two main columns: Left = Brand & Methodology, Right = Sign Up & Login
    col_left, col_right = st.columns([1.15, 0.95], gap="large")

    # =========================================================================
    # LEFT COLUMN: BRAND SHOWCASE, VALUE PROPOSITION, METHODOLOGY
    # =========================================================================
    with col_left:
        # Outer Terracotta Card
        st.markdown(f"""
        <div class="brand-showcase-card">
            <!-- Brand Header -->
            <div class="brand-header-flex">
                <div class="brand-logo-wrap">
                    {"<img src='" + logo_b64 + "' class='brand-logo-img' alt='Credit Analyze Logo' />" if logo_b64 else "<div class='brand-logo-fallback'>CA</div>"}
                </div>
                <div class="brand-header-text">
                    <h1 class="brand-main-title">CREDIT ANALYZE</h1>
                    <p class="brand-tagline">
                        A high-performance analytics engine built to accelerate financial inclusion across Kenya and Africa by delivering scalable infrastructure.
                    </p>
                    <a href="https://github.com" target="_blank" class="github-repo-pill" rel="noopener noreferrer">
                        <svg height="16" width="16" viewBox="0 0 16 16" fill="currentColor" style="display:inline-block; vertical-align:text-bottom; margin-right:6px;">
                            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                        </svg>
                        <span>GitHub</span>
                        <span class="github-pill-action">View Repository</span>
                    </a>
                </div>
            </div>

            <!-- 3 Executive Feature Cards -->
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon-badge icon-orange">👥</div>
                    <div class="feature-card-title">Accelerate Inclusion</div>
                    <div class="feature-card-desc">
                        Robust, scalable credit infrastructure tailored for banks, SACCOs, and fintech developers.
                    </div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon-badge icon-cyan">📈</div>
                    <div class="feature-card-title">Deep Risk Assessment</div>
                    <div class="feature-card-desc">
                        Real-time data visualization and intelligent risk metrics for accurate decision-making.
                    </div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon-badge icon-green">🛡️</div>
                    <div class="feature-card-title">Portfolio Monitoring</div>
                    <div class="feature-card-desc">
                        Navigate lending complexities easily while continuously monitoring overall portfolio health.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Methodology Section
        st.markdown("""
        <div class="methodology-outer-wrap">
            <div class="methodology-title-row">
                <h2 class="methodology-heading">Methodology 📝</h2>
            </div>
        """, unsafe_allow_html=True)

        # Methodology View Switcher (Desktop Diagram & Mobile Responsive Split Cards)
        methodology_view = st.radio(
            "Methodology View Format",
            options=["Diagram View", "Mobile-Friendly Split Cards"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="methodology_view_toggle"
        )

        if methodology_view == "Diagram View" and methodology_b64:
            st.markdown(f"""
            <div class="methodology-diagram-card">
                <img src="{methodology_b64}" class="methodology-banner-img" alt="Credit Risk AutoML 5-Stage Methodology" />
            </div>
            """, unsafe_allow_html=True)
        else:
            # Mobile-friendly split cards format
            st.markdown("""
            <div class="methodology-cards-grid">
                <div class="method-card">
                    <div class="method-step-num">Step 1</div>
                    <div class="method-icon">🔲</div>
                    <div class="method-step-title">Variable Discoverability Matrix</div>
                    <div class="method-step-desc">
                        Identifies alternative indicators across collection hardness and literature predictive power.
                    </div>
                </div>
                <div class="method-card">
                    <div class="method-step-num">Step 2</div>
                    <div class="method-icon">🗄️</div>
                    <div class="method-step-title">Data Storage & Metadata Tags</div>
                    <div class="method-step-desc">
                        <strong>Feature Engineering:</strong> Tags loan panel, transactional, behavioral, and GIS layers in DuckDB.
                    </div>
                </div>
                <div class="method-card">
                    <div class="method-step-num">Step 3</div>
                    <div class="method-icon">🕸️</div>
                    <div class="method-step-title">ML Ensemble with Logistic Regression</div>
                    <div class="method-step-desc">
                        Multi-model comparison: LightGBM, CatBoost, XGBoost & Logistic Regression benchmark.
                    </div>
                </div>
                <div class="method-card">
                    <div class="method-step-num">Step 4</div>
                    <div class="method-icon">🎯</div>
                    <div class="method-step-title">Composite Score Generation</div>
                    <div class="method-step-desc">
                        Calibrated actuarial credit risk score (0–100) reflecting default probability.
                    </div>
                </div>
                <div class="method-card">
                    <div class="method-step-num">Step 5</div>
                    <div class="method-icon">📊</div>
                    <div class="method-step-title">Dashboard Output</div>
                    <div class="method-step-desc">
                        Early Warning & Portfolio Health telemetry with TreeSHAP adverse action reason codes.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Simple & Frictionless Integration Dark Callout Card
        st.markdown("""
            <div class="frictionless-callout-card">
                <div class="frictionless-icon">⚡</div>
                <div class="frictionless-content">
                    <div class="frictionless-title">Simple & Frictionless Integration</div>
                    <div class="frictionless-desc">
                        Simply drag and drop your financial data files into the app. Behind the scenes, Credit Analyze automatically maps variables, engineers features, computes composite credit scores, and delivers early warning signals—all without needing a dedicated data science team.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # RIGHT COLUMN: SIGN UP & LOGIN CARDS
    # =========================================================================
    with col_right:

        # ---------------------------------------------------------------------
        # 1. SIGN UP SECTION (Slate Blue Card)
        # ---------------------------------------------------------------------
        st.markdown("""
        <div class="auth-panel-card signup-card">
            <h2 class="auth-panel-title">Sign Up to the Prototype 💻</h2>
            <p class="auth-panel-wip">
                <strong>Work in Progress:</strong> Credit Analyze is in its early stages of active development. Core features, functionality, and the overall design will evolve rapidly as we continue to build. We appreciate your patience and welcome early feedback!
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("signup_form", clear_on_submit=False):
            st.markdown('<div class="pill-form-container signup-form-inputs">', unsafe_allow_html=True)
            
            signup_email = st.text_input(
                "Email",
                placeholder="email",
                label_visibility="collapsed",
                key="signup_email_input"
            )
            
            signup_details = st.text_input(
                "Other Details",
                placeholder="Other Details (e.g. Name, Institution, Role)",
                label_visibility="collapsed",
                key="signup_details_input"
            )

            col_pwd1, col_pwd2 = st.columns(2)
            with col_pwd1:
                signup_pwd = st.text_input(
                    "Password",
                    type="password",
                    placeholder="password (min 6 chars)",
                    label_visibility="collapsed",
                    key="signup_pwd_input"
                )
            with col_pwd2:
                signup_pwd_confirm = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="confirm password",
                    label_visibility="collapsed",
                    key="signup_pwd_confirm_input"
                )

            # Terms & Conditions + Survey/Interview Consent Tick Box
            agree_terms = st.checkbox(
                "I agree to the Terms & Conditions and consent to participating in research surveys and interviews.",
                value=False,
                key="signup_terms_tickbox"
            )

            with st.expander("📄 Review Prototype Terms, Conditions & Research Consent", expanded=False):
                st.markdown("""
                ### 📋 Credit Analyze Prototype Testing Agreement
                
                1. **Early Prototype Testing**:
                   Credit Analyze is an experimental financial technology platform developed for the Kenya Bankers Association (KBA) Credit Scoring & Alternative Data Initiative. Features are subject to iterative enhancements.
                
                2. **Mandatory Survey & Interview Consent**:
                   By signing up, you explicitly agree to make yourself available for brief user experience surveys, qualitative feedback sessions, and user research interviews to guide the evolution of Credit Analyze.
                
                3. **Immediate Access Grant**:
                   Access is currently open without a tail gate. Valid registration instantly provisions active tester credentials.
                
                4. **Privacy & Data Security (Kenya DPA 2019)**:
                   Your personal credentials and email are safely hashed and stored in an isolated data store. Individual client portfolio records processed in the platform are ephemeral and held in volatile RAM.
                """)

            signup_btn = st.form_submit_button("Sign Up to the Prototype 🚀", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if signup_btn:
                if not signup_email or "@" not in signup_email:
                    st.error("Please provide a valid email address.")
                elif not signup_pwd or len(signup_pwd) < 6:
                    st.error("Password must be at least 6 characters long.")
                elif signup_pwd != signup_pwd_confirm:
                    st.error("Passwords do not match.")
                elif not agree_terms:
                    st.warning("⚠️ You must check the tick box agreeing to the Terms and Conditions and consenting to research surveys and interviews.")
                else:
                    success, msg, user_data = register_user(
                        email=signup_email,
                        password=signup_pwd,
                        other_details=signup_details,
                        agreed_terms=True,
                        survey_consent=True
                    )
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.current_user = user_data
                        st.success("🎉 " + msg)
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 2. LOGIN SECTION (Warm Ochre Card)
        # ---------------------------------------------------------------------
        st.markdown("""
        <div class="auth-panel-card login-card">
            <h2 class="auth-panel-title">Already have access, can login</h2>
            <p class="auth-panel-wip">
                <strong>Work in Progress:</strong> Credit Analyze is in its early stages of active development. Core features, functionality, and the overall design will evolve rapidly as we continue to build. We appreciate your patience and welcome early feedback!
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            st.markdown('<div class="pill-form-container login-form-inputs">', unsafe_allow_html=True)
            
            login_email = st.text_input(
                "Email",
                placeholder="email",
                label_visibility="collapsed",
                key="login_email_input"
            )
            
            login_pwd = st.text_input(
                "Password",
                type="password",
                placeholder="password",
                label_visibility="collapsed",
                key="login_pwd_input"
            )

            login_btn = st.form_submit_button("Log In 🔑", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if login_btn:
                if not login_email or not login_pwd:
                    st.error("Please enter both email and password.")
                else:
                    success, msg, user_data = authenticate_user(login_email, login_pwd)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.current_user = user_data
                        st.success("✓ " + msg)
                        st.rerun()
                    else:
                        st.error("⚠️ " + msg)

    st.markdown('</div>', unsafe_allow_html=True)


def render_authenticated_user_bar() -> None:
    """Renders a sleek top navigation banner when the user is logged into Credit Analyze."""
    user = st.session_state.get("current_user", {})
    email = user.get("email", "Authenticated User")
    role = user.get("role", "Tester").title()
    details = user.get("other_details", "")
    
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        details_badge = f"<span class='user-details-tag'>{details}</span>" if details else ""
        st.markdown(f"""
        <div class="user-session-bar">
            <span class="user-badge-icon">👤</span>
            <span class="user-email-text">{email}</span>
            <span class="user-access-pill">🟢 Access: Granted</span>
            <span class="user-role-pill">{role}</span>
            {details_badge}
        </div>
        """, unsafe_allow_html=True)
        
    with top_col2:
        if st.button("🚪 Sign Out", key="sign_out_btn", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.rerun()
