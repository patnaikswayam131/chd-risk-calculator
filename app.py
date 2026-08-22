import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ── Load artifacts ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(page_title="CHD Risk Calculator", page_icon="❤️")
st.title("❤️ 10-Year CHD Risk Calculator")
st.caption("Based on the Framingham Heart Study | Logistic Regression + Conformal Prediction")

# ── Input form ──────────────────────────────────────────────────
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age        = st.number_input("Age (years)",            min_value=32,  max_value=70,  value=50)
    male       = st.selectbox("Sex",                       [0, 1],        format_func=lambda x: "Female" if x == 0 else "Male")
    education  = st.selectbox("Education level",           [1, 2, 3, 4],  format_func=lambda x: {1:"< High School", 2:"High School", 3:"Some College", 4:"College+"}[x])
    smoker     = st.selectbox("Current smoker?",           [0, 1],        format_func=lambda x: "No" if x == 0 else "Yes")
    cigs       = st.number_input("Cigarettes per day",     min_value=0,   max_value=70,  value=0) if smoker == 1 else 0
    bpmeds     = st.selectbox("On BP medication?",         [0, 1],        format_func=lambda x: "No" if x == 0 else "Yes")
    stroke     = st.selectbox("Prior stroke?",             [0, 1],        format_func=lambda x: "No" if x == 0 else "Yes")

with col2:
    hyp        = st.selectbox("Hypertensive?",             [0, 1],        format_func=lambda x: "No" if x == 0 else "Yes")
    diabetes   = st.selectbox("Diabetes?",                 [0, 1],        format_func=lambda x: "No" if x == 0 else "Yes")
    totchol    = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    sysbp      = st.number_input("Systolic BP (mmHg)",     min_value=83,  max_value=295, value=120)
    diabp      = st.number_input("Diastolic BP (mmHg)",    min_value=48,  max_value=142, value=80)
    bmi        = st.number_input("BMI (kg/m²)",            min_value=15.0,max_value=55.0,value=25.0)
    hr         = st.number_input("Heart Rate (bpm)",       min_value=44,  max_value=143, value=75)
    glucose    = st.number_input("Glucose (mg/dL)",        min_value=40,  max_value=400, value=80)

# ── Prediction ──────────────────────────────────────────────────
if st.button("Calculate Risk", type="primary"):

    input_dict = {
        "male": male, "age": age, "education": education,
        "currentSmoker": smoker, "cigsPerDay": cigs,
        "BPMeds": bpmeds, "prevalentStroke": stroke,
        "prevalentHyp": hyp, "diabetes": diabetes,
        "totChol": totchol, "sysBP": sysbp, "diaBP": diabp,
        "BMI": bmi, "heartRate": hr, "glucose": glucose
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
    pred_set = [c for c in [0, 1] if scores[c] <= artifacts["q_hat"]]

    # ── Output display ───────────────────────────────────────────
    st.divider()
    st.subheader("Results")

    st.metric("Estimated 10-year CHD probability", f"{prob_chd*100:.1f}%")

    if pred_set == [0]:
        st.success("✅ Prediction set: {0} — Model is confident: LOW risk")
    elif pred_set == [1]:
        st.error("🚨 Prediction set: {1} — Model is confident: HIGH risk")
    elif set(pred_set) == {0, 1}:
        st.warning("⚠️ Prediction set: {0, 1} — Model is uncertain — risk could go either way")
    else:
        st.error("Empty prediction set — unexpected error")

    # ── Confidence context panel ─────────────────────────────────
    st.divider()
    with st.expander("How confident is this model, generally?"):
        st.write("""
        **What is a prediction set?**
        Instead of a single probability, conformal prediction outputs a *set* of possible outcomes
        that is guaranteed to contain the true outcome at least 90% of the time — without
        assuming anything about the data distribution beyond exchangeability.

        **Empirical coverage on held-out test data:**
        - 90% target → 92.1% actual coverage ✅
        - 95% target → 96.5% actual coverage ✅

        **Known limitation:**
        Marginal coverage (overall) is guaranteed. Coverage for high-risk patients specifically
        is lower — a documented limitation of marginal conformal prediction with imbalanced data.
        This calculator should not be used for clinical decisions.
        """)
