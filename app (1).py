import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSense AI",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0f1e;
    color: #e8eaf0;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #0d1f3c 0%, #0a0f1e 60%),
                radial-gradient(ellipse at 80% 80%, #0e1a30 0%, transparent 60%);
}

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.app-header { text-align: center; padding: 2.5rem 0 1.5rem 0; }
.app-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -1px;
}
.app-header p { color: #6b7fa3; font-size: 1rem; font-weight: 300; margin-top: 0.5rem; }

.gradient-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, #3b82f6, #8b5cf6, transparent);
    margin: 1rem 0 2rem 0;
    border: none;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

.info-box {
    background: #111827;
    border: 1px dashed #1e3a5f;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    font-size: 0.84rem;
    color: #6b7fa3;
    margin-bottom: 1.2rem;
}
.info-box code {
    color: #60a5fa;
    background: rgba(59,130,246,0.1);
    padding: 1px 5px;
    border-radius: 4px;
}

.derived-box {
    background: #0f1f35;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    font-size: 0.82rem;
    color: #6b7fa3;
    margin-top: 0.5rem;
}
.derived-box span { color: #60a5fa; font-weight: 500; }

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 1px;
    padding: 0.85rem 2rem;
    border: none;
    border-radius: 10px;
    margin-top: 1.5rem;
    text-transform: uppercase;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(37,99,235,0.35);
}

.result-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 2rem;
    animation: fadeInUp 0.5s ease;
}
.result-approved { background: linear-gradient(135deg, #052e16, #064e3b); border: 1px solid #059669; }
.result-rejected  { background: linear-gradient(135deg, #1c0a0a, #3b0f0f); border: 1px solid #dc2626; }
.result-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0.25rem 0;
}
.result-approved .result-title { color: #34d399; }
.result-rejected  .result-title { color: #f87171; }
.result-subtitle { color: #9ca3af; font-size: 0.9rem; margin-top: 0.25rem; }
.result-confidence {
    display: inline-block;
    margin-top: 1rem;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 500;
}
.result-approved .result-confidence { background: rgba(52,211,153,0.15); color: #34d399; }
.result-rejected  .result-confidence { background: rgba(248,113,113,0.15); color: #f87171; }

.feature-summary {
    background: #111827;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-top: 1.5rem;
    font-size: 0.82rem;
    border: 1px solid #1e3a5f;
}
.feature-summary table { width: 100%; border-collapse: collapse; }
.feature-summary td { padding: 4px 8px; }
.feature-summary td:first-child { color: #9ca3af; }
.feature-summary td:last-child  { color: #e8eaf0; font-weight: 500; text-align: right; }
.feature-summary .derived { color: #60a5fa !important; }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>LoanSense AI</h1>
    <p>Intelligent loan approval · powered by machine learning</p>
</div>
<hr class="gradient-line">
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL & SCALER LOADER
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts(mp, sp):
    return joblib.load(mp), joblib.load(sp)

model, scaler = None, None
model_path  = Path("model.pkl")
scaler_path = Path("scaler.pkl")

missing = [f for f, p in [("model.pkl", model_path), ("scaler.pkl", scaler_path)] if not p.exists()]

if missing:
    st.markdown(
        '<div class="info-box">📁 <strong>Missing: '
        + ", ".join(f"<code>{f}</code>" for f in missing)
        + "</strong><br>Export from Colab and upload below, or place in the app folder.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("📋 Export from Google Colab"):
        st.code("""\
import joblib
from google.colab import files

joblib.dump(mod,    "model.pkl");   files.download("model.pkl")
joblib.dump(scaler, "scaler.pkl");  files.download("scaler.pkl")
""", language="python")

    c1, c2 = st.columns(2)
    up_model  = c1.file_uploader("Upload model.pkl",  type=["pkl"])
    up_scaler = c2.file_uploader("Upload scaler.pkl", type=["pkl"])
    if up_model:
        model_path.write_bytes(up_model.read())
    if up_scaler:
        scaler_path.write_bytes(up_scaler.read())
    if up_model or up_scaler:
        st.success("Saved! Reloading…"); st.rerun()
else:
    try:
        model, scaler = load_artifacts(str(model_path), str(scaler_path))
        st.success("✅ Model & scaler loaded successfully")
    except Exception as e:
        st.error(f"Load failed: {e}")


# ─────────────────────────────────────────────
# INPUTS
#
# Exact 10 features the scaler was trained on:
#  1. no_of_dependents
#  2. education
#  3. self_employed
#  4. income_annum
#  5. loan_amount
#  6. loan_term
#  7. cibil_score
#  8. total_assets          = bank + commercial + residential + luxury
#  9. loan_income_ratio     = loan_amount / income_annum  (auto-computed)
# 10. emi                   = loan_amount / loan_term     (auto-computed)
# ─────────────────────────────────────────────

st.markdown('<div class="section-title">Personal Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    no_of_dependents = st.selectbox("No. of Dependants", [0, 1, 2, 3, 4, 5])
with col2:
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
with col3:
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

st.markdown('<div class="section-title">Loan Details</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)
with col4:
    income_annum = st.number_input("Annual Income (₹)", min_value=1,
                                    max_value=10_000_000, value=500_000, step=10_000)
with col5:
    loan_amount = st.number_input("Loan Amount (₹)", min_value=1,
                                   max_value=50_000_000, value=1_000_000, step=50_000)
with col6:
    loan_term = st.number_input("Loan Term (months)", min_value=1,
                                 max_value=360, value=12, step=1)

st.markdown('<div class="section-title">Credit Score</div>', unsafe_allow_html=True)

cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=650,
                         help="300 = poor  ·  900 = excellent")

st.markdown('<div class="section-title">Assets (₹)</div>', unsafe_allow_html=True)

col7, col8 = st.columns(2)
with col7:
    bank_asset_value         = st.number_input("Bank Assets (₹)",          min_value=0, value=100_000, step=10_000)
    commercial_assets_value  = st.number_input("Commercial Assets (₹)",    min_value=0, value=0,       step=10_000)
with col8:
    residential_assets_value = st.number_input("Residential Assets (₹)",   min_value=0, value=500_000, step=10_000)
    luxury_assets_value      = st.number_input("Luxury Assets (₹)",        min_value=0, value=0,       step=10_000)

# ── Engineered features (computed automatically, matching notebook) ──
total_assets       = bank_asset_value + commercial_assets_value + residential_assets_value + luxury_assets_value
loan_income_ratio  = loan_amount / income_annum
emi                = loan_amount / loan_term

st.markdown(f"""
<div class="derived-box">
    ⚙️ Auto-computed features &nbsp;·&nbsp;
    Total Assets: <span>₹{total_assets:,.0f}</span> &nbsp;·&nbsp;
    Loan/Income Ratio: <span>{loan_income_ratio:.3f}</span> &nbsp;·&nbsp;
    EMI: <span>₹{emi:,.0f}/mo</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ENCODE — exactly 10 features in the correct order
# ─────────────────────────────────────────────
def encode_inputs():
    return np.array([[
        no_of_dependents,
        1 if education == "Graduate" else 0,   # Graduate→1, Not Graduate→0
        1 if self_employed == "Yes" else 0,    # Yes→1, No→0
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        total_assets,                           # engineered
        loan_income_ratio,                      # engineered
        emi,                                    # engineered
    ]], dtype=np.float32)


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
if st.button("⚡ Predict Loan Approval"):
    if model is None or scaler is None:
        st.warning("Please load both model.pkl and scaler.pkl first.")
    else:
        with st.spinner("Analysing application…"):
            time.sleep(0.5)
            raw           = encode_inputs()
            scaled        = scaler.transform(raw)
            pred          = model.predict(scaled)[0]
            proba         = model.predict_proba(scaled)[0]
            prob_approved = float(proba[1])

        approved   = pred == 1
        confidence = prob_approved if approved else (1 - prob_approved)

        if approved:
            st.markdown(f"""
            <div class="result-card result-approved">
                <div class="result-icon">✅</div>
                <div class="result-title">Loan Approved</div>
                <div class="result-subtitle">Application meets the approval criteria</div>
                <div class="result-confidence">Confidence: {confidence:.1%}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-rejected">
                <div class="result-icon">❌</div>
                <div class="result-title">Loan Rejected</div>
                <div class="result-subtitle">Application does not meet the approval criteria</div>
                <div class="result-confidence">Confidence: {confidence:.1%}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Approval Probability</div>', unsafe_allow_html=True)
        st.progress(prob_approved, text=f"{prob_approved:.1%} probability of approval")

        st.markdown(f"""
        <div class="feature-summary">
            <table>
                <tr><td>Dependants</td>           <td>{no_of_dependents}</td></tr>
                <tr><td>Education</td>             <td>{education}</td></tr>
                <tr><td>Self Employed</td>         <td>{self_employed}</td></tr>
                <tr><td>Annual Income</td>         <td>₹{income_annum:,.0f}</td></tr>
                <tr><td>Loan Amount</td>           <td>₹{loan_amount:,.0f}</td></tr>
                <tr><td>Loan Term</td>             <td>{loan_term} months</td></tr>
                <tr><td>CIBIL Score</td>           <td>{cibil_score}</td></tr>
                <tr><td>Total Assets</td>          <td class="derived">₹{total_assets:,.0f}</td></tr>
                <tr><td>Loan / Income Ratio</td>   <td class="derived">{loan_income_ratio:.4f}</td></tr>
                <tr><td>EMI</td>                   <td class="derived">₹{emi:,.0f} / month</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)
