import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="CHD Risk · Framingham",
    page_icon="🫀",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117;
    color: #e8e8e8;
}

.main { background-color: #0f1117; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 2rem 4rem 2rem; max-width: 780px; }

/* Hero header */
.hero {
    border-left: 3px solid #e05c5c;
    padding: 1.2rem 1.5rem;
    margin-bottom: 2.5rem;
    background: linear-gradient(135deg, #1a1d27 0%, #161822 100%);
    border-radius: 0 12px 12px 0;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    font-weight: 400;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
    line-height: 1.2;
}
.hero p {
    color: #8b8fa8;
    font-size: 0.82rem;
    margin: 0;
    letter-spacing: 0.03em;
}

/* Section headers */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #e05c5c;
    margin: 2rem 0 0.8rem 0;
}

/* Input card */
.input-card {
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Streamlit widgets inside dark bg */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div {
    background-color: #12141e !important;
    color: #e8e8e8 !important;
    border-color: #2a2d3e !important;
    border-radius: 8px !important;
}

label { color: #a0a4b8 !important; font-size: 0.82rem !important; }

/* Button */
div[data-testid="stButton"] > button {
    background: #e05c5c;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 2.5rem;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    width: 100%;
    margin-top: 1.5rem;
    transition: background 0.2s;
}
div[data-testid="stButton"] > button:hover {
    background: #c94c4c;
}

/* Result cards */
.result-card {
    border-radius: 12px;
    padding: 1.8rem;
    margin: 1rem 0;
    text-align: center;
}
.result-low {
    background: linear-gradient(135deg, #0d2117, #0a1c12);
    border: 1px solid #1e5c38;
}
.result-high {
    background: linear-gradient(135deg, #250d0d, #1c0a0a);
    border: 1px solid #7a1e1e;
}
.result-uncertain {
    background: linear-gradient(135deg, #1c1a0d, #171508);
    border: 1px solid #5c500d;
}
.result-prob {
    font-family: 'DM Serif Display', serif;
    font-size: 3.5rem;
    font-weight: 400;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.result-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.result-verdict {
    font-size: 1rem;
    font-weight: 500;
}

/* Confidence panel */
.conf-panel {
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1.5rem;
}
.conf-panel h4 {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b8fa8;
    margin: 0 0 0.8rem 0;
}
.conf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.35rem 0;
    border-bottom: 1px solid #22253a;
    font-size: 0.82rem;
}
.conf-row:last-child { border-bottom: none; }
.conf-pass { color: #4caf7d; font-weight: 600; }
.caveat {
    font-size: 0.72rem;
    color: #5a5e78;
    margin-top: 0.8rem;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🫀 10-Year CHD Risk</h1>
    <p>FRAMINGHAM HEART STUDY · LOGISTIC REGRESSION + CONFORMAL PREDICTION · AUC 0.748</p>
</div>
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Patient Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age      = st.number_input("Age (years)",              32, 70,  50)
    male     = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
    smoker   = st.selectbox("Current smoker?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    cigs     = st.number_input("Cigarettes per day", 0, 70, 0) if smoker==1 else 0
    bpmeds   = st.selectbox("On BP medication?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    stroke   = st.selectbox("Prior stroke?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    diabetes = st.selectbox("Diabetes?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    hyp      = st.selectbox("Hypertensive?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

with col2:
    education = st.selectbox("Education", [1,2,3,4],
                    format_func=lambda x: {1:"< High school",2:"High school",
                                           3:"Some college",4:"College+"}[x])
    totchol  = st.number_input("Total cholesterol (mg/dL)", 100, 600, 200)
    sysbp    = st.number_input("Systolic BP (mmHg)",         83, 295, 120)
    diabp    = st.number_input("Diastolic BP (mmHg)",        48, 142,  80)
    bmi      = st.number_input("BMI (kg/m²)",              15.0,55.0, 25.0)
    hr       = st.number_input("Heart rate (bpm)",           44, 143,  75)
    glucose  = st.number_input("Glucose (mg/dL)",           40, 400,  80)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Calculate Risk"):

    input_dict = {
        "male":male,"age":age,"education":education,
        "currentSmoker":smoker,"cigsPerDay":cigs,
        "BPMeds":bpmeds,"prevalentStroke":stroke,
        "prevalentHyp":hyp,"diabetes":diabetes,
        "totChol":totchol,"sysBP":sysbp,"diaBP":diabp,
        "BMI":bmi,"heartRate":hr,"glucose":glucose
    }

    feature_names = artifacts["feature_names"]
    median_cols   = artifacts["median_cols"]
    mode_cols     = artifacts["mode_cols"]

    row = pd.DataFrame([input_dict])[feature_names]
    row[median_cols] = artifacts["median_imputer"].transform(row[median_cols])
    row[mode_cols]   = artifacts["mode_imputer"].transform(row[mode_cols])
    row_scaled       = artifacts["scaler"].transform(row)

    proba    = artifacts["model"].predict_proba(row_scaled)[0]
    prob_chd = proba[1]
    scores   = 1 - proba
    pred_set = [c for c in [0,1] if scores[c] <= artifacts["q_hat"]]

    st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)

    if pred_set == [0]:
        card_class, icon, verdict = "result-low", "🟢", "Prediction set {0} — low risk"
        color = "#4caf7d"
    elif pred_set == [1]:
        card_class, icon, verdict = "result-high", "🔴", "Prediction set {1} — high risk"
        color = "#e05c5c"
    else:
        card_class, icon, verdict = "result-uncertain", "🟡", "Prediction set {0,1} — uncertain"
        color = "#d4a017"

    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-prob" style="color:{color}">{prob_chd*100:.1f}%</div>
        <div class="result-label" style="color:{color}">Estimated 10-year CHD probability</div>
        <div class="result-verdict">{icon} {verdict}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="conf-panel">
        <h4>Model Confidence · Empirical Coverage</h4>
        <div class="conf-row"><span>90% target coverage</span>
            <span class="conf-pass">92.1% actual ✓</span></div>
        <div class="conf-row"><span>95% target coverage</span>
            <span class="conf-pass">96.5% actual ✓</span></div>
        <div class="conf-row"><span>Discrimination (AUC)</span>
            <span>0.748</span></div>
        <div class="conf-row"><span>Calibration (ECE)</span>
            <span>0.012 — natively well-calibrated</span></div>
        <p class="caveat">
        Conformal prediction guarantees the true outcome falls within
        the prediction set ≥90% of the time (marginal coverage, held-out test set).
        Not for clinical use. Known limitation: conditional coverage for
        high-risk patients is lower — a documented property of marginal
        conformal prediction with imbalanced data.
        </p>
    </div>
    """, unsafe_allow_html=True)
