"""
Credit Analyze - Custom Authentication UI Page
==============================================
Two-card side-by-side layout using st.columns (the only reliable
method for placing Streamlit widgets side-by-side):
  LEFT  column -> Scrollable branded showcase card (terracotta gradient, pure HTML)
  RIGHT column -> Compact auth card with Sign Up / Login toggle (Streamlit widgets)

Sign Up fields: Full Name, Email, Institution, Role (select), Country, Phone, Password
All profile fields combined into other_details for DB storage.
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

# Role options for the sign-up form
ROLE_OPTIONS = [
    "Select your role...",
    "Banking Professional",
    "SACCO Manager",
    "Fintech Developer",
    "Data Scientist / Analyst",
    "Credit Risk Officer",
    "Researcher / Academic",
    "Regulator / Policy Maker",
    "Student",
    "Other",
]


def _b64(path: str) -> str:
    """Return a base64 data-URI for an image file, or empty string if missing."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            enc = base64.b64encode(f.read()).decode()
        ext  = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "svg": "image/svg+xml"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{enc}"
    return ""


# Page-level CSS injected once
_AUTH_CSS = (
    "<style>"
    "\n@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Questrial&display=swap');"
    "\n"
    "\n.block-container:has(.auth-page-root) {"
    "\n    padding-top: 1rem !important;"
    "\n    padding-bottom: 1rem !important;"
    "\n    max-width: 1520px !important;"
    "\n}"
    "\n.auth-page-root { margin-top: -10px; }"
    "\n"
    "\n/* Desktop: pin right column, scroll left */"
    "\n@media (min-width: 992px) {"
    "\n    div[data-testid='column']:first-child div[data-testid='stVerticalBlockBorderWrapper'] {"
    "\n        border: none !important;"
    "\n    }"
    "\n    div[data-testid='column']:last-child {"
    "\n        position: sticky !important;"
    "\n        top: 12px !important;"
    "\n        align-self: flex-start !important;"
    "\n    }"
    "\n}"
    "\n@media (max-width: 991px) {"
    "\n    div[data-testid='column']:last-child {"
    "\n        position: static !important;"
    "\n        margin-top: 16px !important;"
    "\n    }"
    "\n}"
    "\n"
    "\n/* Left card: terracotta showcase */"
    "\n.brand-showcase-card {"
    "\n    background: linear-gradient(148deg, #d97218 0%, #a84903 100%);"
    "\n    border-radius: 28px;"
    "\n    padding: 30px 26px 34px 26px;"
    "\n    color: #ffffff;"
    "\n    box-shadow: 0 18px 40px -8px rgba(168,73,3,0.40);"
    "\n}"
    "\n.brand-hdr { display:flex; align-items:flex-start; gap:18px; margin-bottom:20px; }"
    "\n.brand-logo-box {"
    "\n    flex-shrink:0; width:84px; height:84px; background:#fff;"
    "\n    border-radius:16px; padding:5px;"
    "\n    box-shadow:0 6px 16px rgba(0,0,0,0.18);"
    "\n    display:flex; align-items:center; justify-content:center;"
    "\n}"
    "\n.brand-logo-box img { width:100%; height:100%; object-fit:contain; border-radius:10px; }"
    "\n.brand-logo-fb { font-family:'DM Sans',sans-serif; font-weight:900; font-size:20px; color:#1e4a68; }"
    "\n.brand-title-col { flex:1; }"
    "\n.brand-name {"
    "\n    font-family:'DM Sans',sans-serif !important;"
    "\n    font-weight:800; font-size:32px; letter-spacing:-0.5px;"
    "\n    color:#fff !important; margin:0 0 5px 0; text-transform:uppercase; line-height:1.05;"
    "\n}"
    "\n.brand-tagline {"
    "\n    font-family:'Century Gothic','Questrial',sans-serif !important;"
    "\n    font-size:12.5px; line-height:1.5; color:rgba(255,255,255,0.92) !important; margin:0 0 10px 0;"
    "\n}"
    "\n.gh-pill {"
    "\n    display:inline-flex; align-items:center;"
    "\n    background:#0f172a; color:#fff !important;"
    "\n    padding:5px 12px; border-radius:9999px;"
    "\n    font-family:'DM Sans',sans-serif; font-size:11.5px; font-weight:600;"
    "\n    text-decoration:none; box-shadow:0 2px 6px rgba(0,0,0,0.25); transition:all 0.2s;"
    "\n}"
    "\n.gh-pill:hover { background:#1e293b; transform:translateY(-1px); }"
    "\n.gh-pill-action {"
    "\n    background:#2563eb; color:#fff; font-size:10px;"
    "\n    padding:2px 7px; border-radius:9999px; margin-left:6px; font-weight:700;"
    "\n}"
    "\n"
    "\n.feat-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:20px; }"
    "\n.feat-card {"
    "\n    background:#fff; border-radius:16px; padding:14px 10px; text-align:center;"
    "\n    box-shadow:0 5px 14px rgba(0,0,0,0.08);"
    "\n    display:flex; flex-direction:column; align-items:center;"
    "\n    transition:transform 0.2s, box-shadow 0.2s;"
    "\n}"
    "\n.feat-card:hover { transform:translateY(-2px); box-shadow:0 10px 20px rgba(0,0,0,0.12); }"
    "\n.feat-icon {"
    "\n    width:38px; height:38px; border-radius:50%;"
    "\n    display:flex; align-items:center; justify-content:center;"
    "\n    font-size:16px; margin-bottom:7px;"
    "\n}"
    "\n.ic-orange{background:#ffedd5;} .ic-cyan{background:#e0f2fe;} .ic-green{background:#dcfce7;}"
    "\n.feat-title {"
    "\n    font-family:'DM Sans',sans-serif !important;"
    "\n    font-size:12px; font-weight:700; color:#0f172a !important; margin-bottom:4px; line-height:1.25;"
    "\n}"
    "\n.feat-desc {"
    "\n    font-family:'Century Gothic','Questrial',sans-serif !important;"
    "\n    font-size:10px; color:#475569 !important; line-height:1.45;"
    "\n}"
    "\n"
    "\n.meth-heading {"
    "\n    font-family:'DM Sans',sans-serif !important;"
    "\n    font-weight:800; font-size:20px; color:#fff !important; margin:0 0 10px 0;"
    "\n}"
    "\n.meth-img-card {"
    "\n    background:#fff; border-radius:14px; padding:10px;"
    "\n    box-shadow:0 4px 12px rgba(0,0,0,0.08); margin-bottom:16px; text-align:center;"
    "\n}"
    "\n.meth-img-card img { width:100%; height:auto; border-radius:6px; display:block; }"
    "\n"
    "\n.frict-card {"
    "\n    background:#1a1e24; border-radius:12px; padding:12px 14px;"
    "\n    display:flex; align-items:flex-start; gap:10px;"
    "\n    box-shadow:0 4px 12px rgba(0,0,0,0.20);"
    "\n}"
    "\n.frict-icon {"
    "\n    font-size:18px; background:rgba(245,158,11,0.2);"
    "\n    width:32px; height:32px; border-radius:8px;"
    "\n    display:flex; align-items:center; justify-content:center; flex-shrink:0;"
    "\n}"
    "\n.frict-title { font-family:'DM Sans',sans-serif; font-size:13px; font-weight:700; color:#fff; margin-bottom:3px; }"
    "\n.frict-desc { font-family:'Century Gothic','Questrial',sans-serif; font-size:10.5px; color:#94a3b8; line-height:1.45; }"
    "\n"
    "\n/* Right card */"
    "\n.auth-card-logo-row { display:flex; align-items:center; gap:10px; margin-bottom:14px; }"
    "\n.auth-card-logo-box {"
    "\n    width:40px; height:40px; background:#f8fafc; border-radius:10px; padding:3px;"
    "\n    display:flex; align-items:center; justify-content:center;"
    "\n    box-shadow:0 2px 6px rgba(0,0,0,0.08); border:1px solid #e2e8f0;"
    "\n}"
    "\n.auth-card-logo-box img { width:100%; height:100%; object-fit:contain; border-radius:7px; }"
    "\n.auth-card-app-name {"
    "\n    font-family:'DM Sans',sans-serif !important;"
    "\n    font-weight:800; font-size:16px; color:#0f172a !important; letter-spacing:-0.3px;"
    "\n}"
    "\n"
    "\n/* Form styling */"
    "\ndiv[data-testid='stForm'] { border:none !important; padding:0 !important; background:transparent !important; }"
    "\ndiv[data-testid='stForm'] input[type='text'],"
    "\ndiv[data-testid='stForm'] input[type='password'] {"
    "\n    border-radius:12px !important; background-color:#f8fafc !important;"
    "\n    color:#0f172a !important;"
    "\n    font-family:'Century Gothic','Questrial',sans-serif !important;"
    "\n    font-size:13px !important; padding:9px 14px !important;"
    "\n    border:1px solid #e2e8f0 !important; box-shadow:none !important;"
    "\n}"
    "\ndiv[data-testid='stForm'] input:focus {"
    "\n    border-color:#2563eb !important;"
    "\n    box-shadow:0 0 0 3px rgba(37,99,235,0.15) !important; outline:none !important;"
    "\n}"
    "\ndiv[data-testid='stForm'] button {"
    "\n    border-radius:9999px !important; font-family:'DM Sans',sans-serif !important;"
    "\n    font-weight:700 !important; font-size:13.5px !important;"
    "\n    min-height:42px !important; transition:all 0.2s ease !important; margin-top:4px !important;"
    "\n}"
    "\ndiv[data-testid='stCheckbox'] label span {"
    "\n    font-family:'Century Gothic','Questrial',sans-serif !important;"
    "\n    font-size:11px !important; color:#475569 !important; line-height:1.4 !important;"
    "\n}"
    "\n.auth-toggle-row button {"
    "\n    border-radius:9999px !important; font-family:'DM Sans',sans-serif !important;"
    "\n    font-size:12.5px !important; font-weight:700 !important; min-height:38px !important;"
    "\n    transition:all 0.2s ease !important;"
    "\n}"
    "\n.switch-link-btn button {"
    "\n    border-radius:9999px !important;"
    "\n    font-family:'Century Gothic','Questrial',sans-serif !important;"
    "\n    font-size:12px !important; background:#f1f5f9 !important;"
    "\n    color:#475569 !important; border:1px solid #e2e8f0 !important;"
    "\n    min-height:36px !important; margin-top:4px !important;"
    "\n}"
    "\n.field-label {"
    "\n    font-family:'DM Sans',sans-serif;"
    "\n    font-size:11.5px; font-weight:700; color:#334155;"
    "\n    margin-bottom:2px; letter-spacing:0.3px; text-transform:uppercase;"
    "\n}"
    "\n"
    "\n@media (max-width:900px) {"
    "\n    .brand-showcase-card { padding:20px 16px !important; border-radius:20px !important; }"
    "\n    .feat-grid { grid-template-columns:1fr !important; }"
    "\n    .brand-hdr { flex-direction:column !important; align-items:center !important; text-align:center !important; }"
    "\n    .brand-name { font-size:24px !important; }"
    "\n}"
    "\n</style>"
)


def render_login_signup_page() -> None:
    """Renders the two-card auth page using st.columns for reliable side-by-side layout."""
    init_auth_db()

    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "signup"

    logo_b64 = _b64(LOGO_PATH)
    meth_b64 = _b64(METHODOLOGY_PATH)

    # Inject auth page CSS
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)
    st.markdown('<div class="auth-page-root">', unsafe_allow_html=True)

    # Two-column layout: LEFT showcase, RIGHT auth form
    col_left, col_right = st.columns([1.25, 0.85], gap="large")

    # ═══════════════════════════════════════════════════════════════════════════
    # LEFT COLUMN: Scrollable branded showcase card (pure HTML)
    # ═══════════════════════════════════════════════════════════════════════════
    with col_left:
        left_box = st.container(height=820, border=False)
        with left_box:
            logo_html = (
                f"<img src='{logo_b64}' alt='Credit Analyze Logo' />"
                if logo_b64 else "<span class='brand-logo-fb'>CA</span>"
            )
            meth_html = (
                f"<img src='{meth_b64}' alt='Methodology Diagram' />"
                if meth_b64 else ""
            )

            st.markdown(
                '<div class="brand-showcase-card">'
                '  <div class="brand-hdr">'
                f'    <div class="brand-logo-box">{logo_html}</div>'
                '    <div class="brand-title-col">'
                '      <div class="brand-name">CREDIT ANALYZE</div>'
                '      <p class="brand-tagline">A high-performance analytics engine built to '
                'accelerate financial inclusion across Kenya and Africa by delivering scalable, '
                'intelligent credit infrastructure.</p>'
                '      <a href="https://github.com" target="_blank" class="gh-pill" rel="noopener noreferrer">'
                '        <svg height="14" width="14" viewBox="0 0 16 16" fill="currentColor">'
                '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
                '0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15'
                '-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51'
                '-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 '
                '0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2'
                '-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29'
                '.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 '
                '8c0-4.42-3.58-8-8-8z"/>'
                '</svg>'
                '        <span style="margin-left:6px">GitHub</span>'
                '        <span class="gh-pill-action">View Repo</span>'
                '      </a>'
                '    </div>'
                '  </div>'
                '  <div class="feat-grid">'
                '    <div class="feat-card"><div class="feat-icon ic-orange">&#128101;</div>'
                '<div class="feat-title">Accelerate Inclusion</div>'
                '<div class="feat-desc">Scalable credit infrastructure for banks, SACCOs, and fintech developers.</div></div>'
                '    <div class="feat-card"><div class="feat-icon ic-cyan">&#128200;</div>'
                '<div class="feat-title">Deep Risk Assessment</div>'
                '<div class="feat-desc">Real-time visualisation and intelligent risk metrics for accurate decisions.</div></div>'
                '    <div class="feat-card"><div class="feat-icon ic-green">&#128737;</div>'
                '<div class="feat-title">Portfolio Monitoring</div>'
                '<div class="feat-desc">Continuous portfolio health monitoring with early warning signals.</div></div>'
                '  </div>'
                '  <div class="meth-heading">Methodology &#128221;</div>'
                f'  <div class="meth-img-card">{meth_html}</div>'
                '  <div class="frict-card">'
                '    <div class="frict-icon">&#9889;</div>'
                '    <div>'
                '      <div class="frict-title">Simple &amp; Frictionless Integration</div>'
                '      <div class="frict-desc">Drag and drop your financial data files into the app. '
                'Credit Analyze automatically maps variables, engineers features, computes composite '
                'credit scores, and delivers early warning signals.</div>'
                '    </div>'
                '  </div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # RIGHT COLUMN: Compact auth card with toggle
    # ═══════════════════════════════════════════════════════════════════════════
    with col_right:
        logo_small = f"<img src='{logo_b64}' alt='Logo' />" if logo_b64 else "CA"
        st.markdown(
            '<div class="auth-card-logo-row">'
            f'  <div class="auth-card-logo-box">{logo_small}</div>'
            '  <div class="auth-card-app-name">Credit Analyze</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Toggle buttons
        is_signup = (st.session_state.auth_mode == "signup")

        st.markdown('<div class="auth-toggle-row">', unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        with t1:
            if st.button(
                "Sign Up", key="btn_tog_signup", use_container_width=True,
                type="primary" if is_signup else "secondary",
            ):
                st.session_state.auth_mode = "signup"
                st.rerun()
        with t2:
            if st.button(
                "Log In", key="btn_tog_login", use_container_width=True,
                type="primary" if not is_signup else "secondary",
            ):
                st.session_state.auth_mode = "login"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # ── SIGN UP VIEW ─────────────────────────────────────────────────────
        if is_signup:
            st.markdown(
                '<div style="margin:8px 0 6px 0">'
                '<div style="font-family:\'DM Sans\',sans-serif;font-weight:800;font-size:18px;'
                'color:#0f172a;margin-bottom:2px">Create Your Account</div>'
                '<div style="font-family:\'Century Gothic\',\'Questrial\',sans-serif;font-size:11px;'
                'color:#64748b;line-height:1.4">Sign up for instant access to the Credit Analyze prototype.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            with st.form("signup_form", clear_on_submit=False):
                # Full Name
                st.markdown('<div class="field-label">Full Name *</div>', unsafe_allow_html=True)
                su_name = st.text_input(
                    "Full Name", placeholder="e.g. Jane Wanjiku",
                    label_visibility="collapsed", key="su_name",
                )

                # Email
                st.markdown('<div class="field-label">Email Address *</div>', unsafe_allow_html=True)
                su_email = st.text_input(
                    "Email", placeholder="jane@institution.co.ke",
                    label_visibility="collapsed", key="su_email",
                )

                # Institution & Role side by side
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown('<div class="field-label">Institution / Company</div>', unsafe_allow_html=True)
                    su_institution = st.text_input(
                        "Institution", placeholder="e.g. KCB Bank",
                        label_visibility="collapsed", key="su_inst",
                    )
                with r2:
                    st.markdown('<div class="field-label">Role</div>', unsafe_allow_html=True)
                    su_role = st.selectbox(
                        "Role", options=ROLE_OPTIONS,
                        label_visibility="collapsed", key="su_role",
                    )

                # Country & Phone side by side
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="field-label">Country</div>', unsafe_allow_html=True)
                    su_country = st.text_input(
                        "Country", placeholder="e.g. Kenya",
                        label_visibility="collapsed", key="su_country",
                    )
                with c2:
                    st.markdown('<div class="field-label">Phone (optional)</div>', unsafe_allow_html=True)
                    su_phone = st.text_input(
                        "Phone", placeholder="+254 7XX XXX XXX",
                        label_visibility="collapsed", key="su_phone",
                    )

                # Password row
                p1, p2 = st.columns(2)
                with p1:
                    st.markdown('<div class="field-label">Password *</div>', unsafe_allow_html=True)
                    su_pwd = st.text_input(
                        "Password", type="password", placeholder="min 6 characters",
                        label_visibility="collapsed", key="su_pwd",
                    )
                with p2:
                    st.markdown('<div class="field-label">Confirm Password *</div>', unsafe_allow_html=True)
                    su_pwd2 = st.text_input(
                        "Confirm", type="password", placeholder="re-enter password",
                        label_visibility="collapsed", key="su_pwd2",
                    )

                # Terms checkbox
                agree = st.checkbox(
                    "I agree to the Terms & Conditions and consent to participating "
                    "in research surveys and interviews.",
                    value=False, key="su_terms",
                )

                with st.expander("Review Terms, Conditions & Research Consent", expanded=False):
                    st.markdown(
                        "**1. Early Prototype Testing** \u2014 Credit Analyze is experimental "
                        "software developed for the Kenya Bankers Association (KBA) Credit "
                        "Scoring & Alternative Data Initiative.\n\n"
                        "**2. Survey & Interview Consent** \u2014 By signing up you agree to "
                        "be available for user experience surveys, qualitative feedback "
                        "sessions, and research interviews.\n\n"
                        "**3. Immediate Access** \u2014 Open registration: valid credentials "
                        "provision access instantly.\n\n"
                        "**4. Privacy (Kenya DPA 2019)** \u2014 Credentials are hashed and "
                        "stored in an isolated data store. Portfolio records are ephemeral "
                        "and held in volatile RAM."
                    )

                submitted = st.form_submit_button(
                    "Create Account & Access Prototype", use_container_width=True,
                )

                if submitted:
                    if not su_name or not su_name.strip():
                        st.error("Please enter your full name.")
                    elif not su_email or "@" not in su_email:
                        st.error("Please enter a valid email address.")
                    elif not su_pwd or len(su_pwd) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif su_pwd != su_pwd2:
                        st.error("Passwords do not match.")
                    elif not agree:
                        st.warning("You must accept the Terms & Conditions to continue.")
                    else:
                        role_str = su_role if su_role != "Select your role..." else ""
                        detail_parts = [f"Name: {su_name.strip()}"]
                        if su_institution and su_institution.strip():
                            detail_parts.append(f"Institution: {su_institution.strip()}")
                        if role_str:
                            detail_parts.append(f"Role: {role_str}")
                        if su_country and su_country.strip():
                            detail_parts.append(f"Country: {su_country.strip()}")
                        if su_phone and su_phone.strip():
                            detail_parts.append(f"Phone: {su_phone.strip()}")
                        combined_details = " | ".join(detail_parts)

                        ok, msg, udata = register_user(
                            email=su_email, password=su_pwd,
                            other_details=combined_details,
                            agreed_terms=True, survey_consent=True,
                        )
                        if ok:
                            st.session_state.authenticated = True
                            st.session_state.current_user = udata
                            st.success("Account created! Welcome to Credit Analyze.")
                            st.rerun()
                        else:
                            st.error(msg)

            st.markdown('<div class="switch-link-btn">', unsafe_allow_html=True)
            if st.button("Already have access? Log In", key="sw_to_login", use_container_width=True):
                st.session_state.auth_mode = "login"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # ── LOGIN VIEW ────────────────────────────────────────────────────────
        else:
            st.markdown(
                '<div style="margin:8px 0 6px 0">'
                '<div style="font-family:\'DM Sans\',sans-serif;font-weight:800;font-size:18px;'
                'color:#0f172a;margin-bottom:2px">Welcome Back</div>'
                '<div style="font-family:\'Century Gothic\',\'Questrial\',sans-serif;font-size:11px;'
                'color:#64748b;line-height:1.4">Log in with your registered credentials to access '
                'Credit Analyze.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            with st.form("login_form", clear_on_submit=False):
                st.markdown('<div class="field-label">Email Address</div>', unsafe_allow_html=True)
                li_email = st.text_input(
                    "Email", placeholder="your@email.com",
                    label_visibility="collapsed", key="li_email",
                )

                st.markdown('<div class="field-label">Password</div>', unsafe_allow_html=True)
                li_pwd = st.text_input(
                    "Password", type="password", placeholder="your password",
                    label_visibility="collapsed", key="li_pwd",
                )

                login_btn = st.form_submit_button("Log In", use_container_width=True)

                if login_btn:
                    if not li_email or not li_pwd:
                        st.error("Please enter both email and password.")
                    else:
                        ok, msg, udata = authenticate_user(li_email, li_pwd)
                        if ok:
                            st.session_state.authenticated = True
                            st.session_state.current_user = udata
                            st.success("Welcome back!")
                            st.rerun()
                        else:
                            st.error(msg)

            st.markdown('<div class="switch-link-btn">', unsafe_allow_html=True)
            if st.button("Need an account? Sign Up", key="sw_to_signup", use_container_width=True):
                st.session_state.auth_mode = "signup"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_authenticated_user_bar() -> None:
    """Sleek session bar shown when a user is logged in."""
    user    = st.session_state.get("current_user", {})
    email   = user.get("email", "Authenticated User")
    role    = user.get("role", "Tester").title()
    details = user.get("other_details", "")

    col1, col2 = st.columns([3, 1])
    with col1:
        details_badge = f"<span class='user-details-tag'>{details}</span>" if details else ""
        st.markdown(
            '<div class="user-session-bar">'
            '  <span class="user-badge-icon">&#128100;</span>'
            f'  <span class="user-email-text">{email}</span>'
            '  <span class="user-access-pill">&#128994; Access: Granted</span>'
            f'  <span class="user-role-pill">{role}</span>'
            f'  {details_badge}'
            '</div>',
            unsafe_allow_html=True,
        )

    with col2:
        if st.button("Sign Out", key="sign_out_btn", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.rerun()
