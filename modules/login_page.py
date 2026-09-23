"""
Credit Analyze - Custom Authentication UI Page
==============================================
Card-centric layout:
  LEFT  - One huge, scrollable branded showcase card (terracotta gradient)
  RIGHT - Compact, fixed auth card with Sign Up / Login toggle (no scrolling needed)

Fonts: DM Sans (headings), Century Gothic / Questrial (body)
Mobile: Stacks vertically; left card first, right card below.
"""

import os
import base64
import streamlit as st
from modules.auth import register_user, authenticate_user, init_auth_db

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(MODULE_DIR)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOGO_PATH        = os.path.join(ASSETS_DIR, "logo.png")
METHODOLOGY_PATH = os.path.join(ASSETS_DIR, "methodology.png")


def _b64(path: str) -> str:
    if os.path.exists(path):
        with open(path, "rb") as f:
            enc = base64.b64encode(f.read()).decode()
        ext  = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "svg": "image/svg+xml"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{enc}"
    return ""


_AUTH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Questrial&display=swap');

section.main > div.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 1500px !important;
}

.auth-cards-row {
    display: flex;
    gap: 22px;
    align-items: flex-start;
    width: 100%;
}

.left-showcase-card {
    flex: 1.3;
    background: linear-gradient(148deg, #d97218 0%, #a84903 100%);
    border-radius: 28px;
    padding: 32px 28px 36px 28px;
    color: #ffffff;
    box-shadow: 0 18px 40px -8px rgba(168,73,3,0.40);
    max-height: calc(100vh - 80px);
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.45) transparent;
}
.left-showcase-card::-webkit-scrollbar { width: 7px; }
.left-showcase-card::-webkit-scrollbar-track { background: transparent; }
.left-showcase-card::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.45);
    border-radius: 9999px;
}
.left-showcase-card::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.65); }

.right-auth-card {
    flex: 0.85;
    background: #ffffff;
    border-radius: 28px;
    padding: 28px 24px 24px 24px;
    box-shadow: 0 14px 35px -6px rgba(15,23,42,0.14);
    border: 1px solid #e2e8f0;
    position: sticky;
    top: 16px;
    align-self: flex-start;
}

.brand-hdr {
    display: flex; align-items: flex-start; gap: 18px; margin-bottom: 22px;
}
.brand-logo-box {
    flex-shrink: 0; width: 88px; height: 88px; background: #fff;
    border-radius: 16px; padding: 6px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.18);
    display: flex; align-items: center; justify-content: center;
}
.brand-logo-box img { width:100%; height:100%; object-fit:contain; border-radius:10px; }
.brand-logo-fb { font-family:'DM Sans',sans-serif; font-weight:900; font-size:22px; color:#1e4a68; }
.brand-title-col { flex: 1; }
.brand-name {
    font-family:'DM Sans',sans-serif !important;
    font-weight:800; font-size:34px; letter-spacing:-0.5px;
    color:#fff !important; margin:0 0 6px 0; text-transform:uppercase; line-height:1.05;
}
.brand-tagline {
    font-family:'Century Gothic','Questrial',sans-serif !important;
    font-size:13px; line-height:1.5; color:rgba(255,255,255,0.92) !important; margin:0 0 12px 0;
}
.gh-pill {
    display:inline-flex; align-items:center;
    background:#0f172a; color:#fff !important;
    padding:5px 13px; border-radius:9999px;
    font-family:'DM Sans',sans-serif; font-size:12px; font-weight:600;
    text-decoration:none; box-shadow:0 2px 6px rgba(0,0,0,0.25); transition:all 0.2s;
}
.gh-pill:hover { background:#1e293b; transform:translateY(-1px); }
.gh-pill-action {
    background:#2563eb; color:#fff; font-size:10px;
    padding:2px 7px; border-radius:9999px; margin-left:7px; font-weight:700;
}

.feat-grid {
    display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:22px;
}
.feat-card {
    background:#fff; border-radius:16px; padding:16px 10px; text-align:center;
    box-shadow:0 5px 14px rgba(0,0,0,0.08);
    display:flex; flex-direction:column; align-items:center;
    transition:transform 0.2s, box-shadow 0.2s;
}
.feat-card:hover { transform:translateY(-2px); box-shadow:0 10px 20px rgba(0,0,0,0.12); }
.feat-icon {
    width:40px; height:40px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:17px; margin-bottom:8px;
}
.ic-orange{background:#ffedd5;} .ic-cyan{background:#e0f2fe;} .ic-green{background:#dcfce7;}
.feat-title {
    font-family:'DM Sans',sans-serif !important;
    font-size:12.5px; font-weight:700; color:#0f172a !important; margin-bottom:5px; line-height:1.25;
}
.feat-desc {
    font-family:'Century Gothic','Questrial',sans-serif !important;
    font-size:10.5px; color:#475569 !important; line-height:1.45;
}

.meth-heading {
    font-family:'DM Sans',sans-serif !important;
    font-weight:800; font-size:22px; color:#fff !important; margin:0 0 12px 0;
}
.meth-img-card {
    background:#fff; border-radius:14px; padding:12px;
    box-shadow:0 4px 12px rgba(0,0,0,0.08); margin-bottom:18px; text-align:center;
}
.meth-img-card img { width:100%; height:auto; border-radius:6px; display:block; }

.frict-card {
    background:#1a1e24; border-radius:12px; padding:14px 16px;
    display:flex; align-items:flex-start; gap:12px;
    box-shadow:0 4px 12px rgba(0,0,0,0.20);
}
.frict-icon {
    font-size:20px; background:rgba(245,158,11,0.2);
    width:34px; height:34px; border-radius:9px;
    display:flex; align-items:center; justify-content:center; flex-shrink:0;
}
.frict-title { font-family:'DM Sans',sans-serif; font-size:14px; font-weight:700; color:#fff; margin-bottom:3px; }
.frict-desc { font-family:'Century Gothic','Questrial',sans-serif; font-size:11px; color:#94a3b8; line-height:1.45; }

.auth-card-logo-row { display:flex; align-items:center; gap:10px; margin-bottom:18px; }
.auth-card-logo-box {
    width:44px; height:44px; background:#f8fafc; border-radius:10px; padding:4px;
    display:flex; align-items:center; justify-content:center;
    box-shadow:0 2px 6px rgba(0,0,0,0.08); border:1px solid #e2e8f0;
}
.auth-card-logo-box img { width:100%; height:100%; object-fit:contain; border-radius:7px; }
.auth-card-app-name {
    font-family:'DM Sans',sans-serif !important;
    font-weight:800; font-size:18px; color:#0f172a !important; letter-spacing:-0.3px;
}

div[data-testid="stForm"] { border:none !important; padding:0 !important; background:transparent !important; }
div[data-testid="stForm"] input[type="text"],
div[data-testid="stForm"] input[type="password"] {
    border-radius:9999px !important; background-color:#f8fafc !important;
    color:#0f172a !important;
    font-family:'Century Gothic','Questrial',sans-serif !important;
    font-size:13.5px !important; padding:10px 18px !important;
    border:1px solid #e2e8f0 !important; box-shadow:none !important;
}
div[data-testid="stForm"] input:focus {
    border-color:#2563eb !important;
    box-shadow:0 0 0 3px rgba(37,99,235,0.15) !important; outline:none !important;
}
div[data-testid="stForm"] button {
    border-radius:9999px !important; font-family:'DM Sans',sans-serif !important;
    font-weight:700 !important; font-size:14px !important;
    min-height:44px !important; transition:all 0.2s ease !important; margin-top:6px !important;
}
div[data-testid="stCheckbox"] label span {
    font-family:'Century Gothic','Questrial',sans-serif !important;
    font-size:11.5px !important; color:#475569 !important; line-height:1.4 !important;
}
.auth-toggle-row button {
    border-radius:9999px !important; font-family:'DM Sans',sans-serif !important;
    font-size:13px !important; font-weight:700 !important; min-height:40px !important;
    transition:all 0.2s ease !important;
}
.switch-link-btn button {
    border-radius:9999px !important;
    font-family:'Century Gothic','Questrial',sans-serif !important;
    font-size:12.5px !important; background:#f1f5f9 !important;
    color:#475569 !important; border:1px solid #e2e8f0 !important;
    min-height:38px !important; margin-top:6px !important;
}

@media (max-width:768px) {
    .auth-cards-row { flex-direction:column !important; gap:16px !important; }
    .left-showcase-card { max-height:none !important; overflow-y:visible !important; }
    .right-auth-card { position:static !important; }
    .feat-grid { grid-template-columns:1fr !important; }
    .brand-hdr { flex-direction:column !important; align-items:center !important; text-align:center !important; }
    .brand-name { font-size:26px !important; }
}
</style>
"""


def render_login_signup_page() -> None:
    init_auth_db()

    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "signup"

    logo_b64 = _b64(LOGO_PATH)
    meth_b64 = _b64(METHODOLOGY_PATH)

    st.markdown(_AUTH_CSS, unsafe_allow_html=True)

    logo_img_html = f"<img src='{logo_b64}' alt='Credit Analyze Logo' />" if logo_b64 else "<span class='brand-logo-fb'>CA</span>"
    meth_img_html = f"<img src='{meth_b64}' alt='Methodology Diagram' />" if meth_b64 else ""

    left_card_html = f"""
<div class="auth-cards-row">
  <div class="left-showcase-card">
    <div class="brand-hdr">
      <div class="brand-logo-box">{logo_img_html}</div>
      <div class="brand-title-col">
        <div class="brand-name">CREDIT ANALYZE</div>
        <p class="brand-tagline">A high-performance analytics engine built to accelerate financial inclusion across Kenya and Africa by delivering scalable, intelligent credit infrastructure.</p>
        <a href="https://github.com" target="_blank" class="gh-pill" rel="noopener noreferrer">
          <svg height="14" width="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
          <span style="margin-left:6px">GitHub</span>
          <span class="gh-pill-action">View Repo</span>
        </a>
      </div>
    </div>
    <div class="feat-grid">
      <div class="feat-card"><div class="feat-icon ic-orange">&#128101;</div><div class="feat-title">Accelerate Inclusion</div><div class="feat-desc">Scalable credit infrastructure for banks, SACCOs, and fintech developers.</div></div>
      <div class="feat-card"><div class="feat-icon ic-cyan">&#128200;</div><div class="feat-title">Deep Risk Assessment</div><div class="feat-desc">Real-time visualisation and intelligent risk metrics for accurate decisions.</div></div>
      <div class="feat-card"><div class="feat-icon ic-green">&#128737;</div><div class="feat-title">Portfolio Monitoring</div><div class="feat-desc">Continuous portfolio health monitoring with early warning signals.</div></div>
    </div>
    <div class="meth-heading">Methodology &#128221;</div>
    <div class="meth-img-card">{meth_img_html}</div>
    <div class="frict-card">
      <div class="frict-icon">&#9889;</div>
      <div>
        <div class="frict-title">Simple &amp; Frictionless Integration</div>
        <div class="frict-desc">Drag and drop your financial data files into the app. Credit Analyze automatically maps variables, engineers features, computes composite credit scores, and delivers early warning signals.</div>
      </div>
    </div>
  </div>
  <div class="right-auth-card">
"""

    st.markdown(left_card_html, unsafe_allow_html=True)

    logo_small = f"<img src='{logo_b64}' alt='Logo' />" if logo_b64 else "CA"
    st.markdown(f"""
<div class="auth-card-logo-row">
  <div class="auth-card-logo-box">{logo_small}</div>
  <div class="auth-card-app-name">Credit Analyze</div>
</div>
""", unsafe_allow_html=True)

    is_signup = (st.session_state.auth_mode == "signup")

    st.markdown('<div class="auth-toggle-row">', unsafe_allow_html=True)
    tog1, tog2 = st.columns(2)
    with tog1:
        if st.button("Sign Up", key="btn_tog_signup", use_container_width=True,
                     type="primary" if is_signup else "secondary"):
            st.session_state.auth_mode = "signup"
            st.rerun()
    with tog2:
        if st.button("Log In", key="btn_tog_login", use_container_width=True,
                     type="primary" if not is_signup else "secondary"):
            st.session_state.auth_mode = "login"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if is_signup:
        st.markdown("""
<div style="margin-bottom:12px">
  <div style="font-family:'DM Sans',sans-serif;font-weight:800;font-size:20px;color:#0f172a;margin-bottom:3px">Sign Up to the Prototype</div>
  <div style="font-family:'Century Gothic','Questrial',sans-serif;font-size:11.5px;color:#64748b;line-height:1.4">Credit Analyze is in active early development. Sign up for instant prototype access.</div>
</div>
""", unsafe_allow_html=True)

        with st.form("signup_form", clear_on_submit=False):
            signup_email = st.text_input("Email", placeholder="your@email.com",
                                         label_visibility="collapsed", key="su_email")
            signup_details = st.text_input("Details", placeholder="Name, Institution, Role (optional)",
                                           label_visibility="collapsed", key="su_details")
            c1, c2 = st.columns(2)
            with c1:
                signup_pwd = st.text_input("Password", type="password",
                                           placeholder="password (6+ chars)",
                                           label_visibility="collapsed", key="su_pwd")
            with c2:
                signup_pwd2 = st.text_input("Confirm", type="password",
                                            placeholder="confirm password",
                                            label_visibility="collapsed", key="su_pwd2")

            agree = st.checkbox(
                "I agree to the Terms & Conditions and consent to participating in research surveys and interviews.",
                value=False, key="su_terms"
            )

            with st.expander("Review Terms, Conditions & Research Consent", expanded=False):
                st.markdown("""
**1. Early Prototype Testing** - Credit Analyze is experimental software developed for the Kenya Bankers Association (KBA) Credit Scoring & Alternative Data Initiative.

**2. Survey & Interview Consent** - By signing up you agree to be available for user experience surveys, qualitative feedback sessions, and research interviews.

**3. Immediate Access** - Open registration: valid credentials provision access instantly.

**4. Privacy (Kenya DPA 2019)** - Credentials are hashed and stored in an isolated data store. Portfolio records are ephemeral and held in volatile RAM.
""")

            submitted = st.form_submit_button("Create Account & Access Prototype", use_container_width=True)

            if submitted:
                if not signup_email or "@" not in signup_email:
                    st.error("Please enter a valid email address.")
                elif not signup_pwd or len(signup_pwd) < 6:
                    st.error("Password must be at least 6 characters.")
                elif signup_pwd != signup_pwd2:
                    st.error("Passwords do not match.")
                elif not agree:
                    st.warning("You must accept the Terms & Conditions to continue.")
                else:
                    ok, msg, udata = register_user(
                        email=signup_email, password=signup_pwd,
                        other_details=signup_details,
                        agreed_terms=True, survey_consent=True,
                    )
                    if ok:
                        st.session_state.authenticated = True
                        st.session_state.current_user  = udata
                        st.success("Account created! Welcome to Credit Analyze.")
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown('<div class="switch-link-btn">', unsafe_allow_html=True)
        if st.button("Already have access? Log In", key="sw_to_login", use_container_width=True):
            st.session_state.auth_mode = "login"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
<div style="margin-bottom:12px">
  <div style="font-family:'DM Sans',sans-serif;font-weight:800;font-size:20px;color:#0f172a;margin-bottom:3px">Welcome Back</div>
  <div style="font-family:'Century Gothic','Questrial',sans-serif;font-size:11.5px;color:#64748b;line-height:1.4">Log in with your registered credentials to access Credit Analyze.</div>
</div>
""", unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            login_email = st.text_input("Email", placeholder="your@email.com",
                                        label_visibility="collapsed", key="li_email")
            login_pwd = st.text_input("Password", type="password",
                                      placeholder="password",
                                      label_visibility="collapsed", key="li_pwd")

            login_btn = st.form_submit_button("Log In", use_container_width=True)

            if login_btn:
                if not login_email or not login_pwd:
                    st.error("Please enter both email and password.")
                else:
                    ok, msg, udata = authenticate_user(login_email, login_pwd)
                    if ok:
                        st.session_state.authenticated = True
                        st.session_state.current_user  = udata
                        st.success("Welcome back!")
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown('<div class="switch-link-btn">', unsafe_allow_html=True)
        if st.button("Need an account? Sign Up", key="sw_to_signup", use_container_width=True):
            st.session_state.auth_mode = "signup"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


def render_authenticated_user_bar() -> None:
    user    = st.session_state.get("current_user", {})
    email   = user.get("email", "Authenticated User")
    role    = user.get("role", "Tester").title()
    details = user.get("other_details", "")

    col1, col2 = st.columns([3, 1])
    with col1:
        details_badge = f"<span class='user-details-tag'>{details}</span>" if details else ""
        st.markdown(f"""
<div class="user-session-bar">
    <span class="user-badge-icon">&#128100;</span>
    <span class="user-email-text">{email}</span>
    <span class="user-access-pill">&#128994; Access: Granted</span>
    <span class="user-role-pill">{role}</span>
    {details_badge}
</div>
""", unsafe_allow_html=True)

    with col2:
        if st.button("Sign Out", key="sign_out_btn", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.current_user  = None
            st.rerun()
