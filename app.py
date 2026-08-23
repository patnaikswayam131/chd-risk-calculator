import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="CHD Risk Calculator", page_icon="🫀", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; background-color:#0f1117; color:#e8e8e8; }
.main { background-color:#0f1117; }
#MainMenu, footer { visibility:hidden; }
.block-container { padding:2rem 2rem 4rem 2rem; max-width:820px; }
.hero { border-left:3px solid #e05c5c; padding:1.2rem 1.5rem; margin-bottom:2rem;
        background:linear-gradient(135deg,#1a1d27,#161822); border-radius:0 12px 12px 0; }
.hero h1 { font-family:'DM Serif Display',serif; font-size:1.9rem; font-weight:400; color:#fff; margin:0 0 0.3rem 0; }
.hero p  { color:#8b8fa8; font-size:0.8rem; margin:0; letter-spacing:.03em; }
.section-label { font-size:0.7rem; font-weight:600; letter-spacing:.12em;
                 text-transform:uppercase; color:#e05c5c; margin:1.8rem 0 0.6rem 0; }
div[data-testid="stButton"] > button {
    background:#e05c5c !important; color:white !important; border:none !important;
    border-radius:8px !important; padding:0.65rem 2rem !important;
    font-size:0.9rem !important; font-weight:600 !important; width:100% !important; margin-top:1rem !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifacts = load_artifacts()
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

st.markdown("""
<div class="hero">
  <h1>🫀 10-Year CHD Risk Calculator</h1>
  <p>FRAMINGHAM HEART STUDY · LOGISTIC REGRESSION + CONFORMAL PREDICTION · AUC 0.748</p>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Patient Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age      = st.number_input("Age (years)", 32, 70, 50)
    male     = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
    smoker   = st.selectbox("Current smoker?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    cigs     = st.number_input("Cigarettes per day", 0, 70, 0) if smoker==1 else 0
    bpmeds   = st.selectbox("On BP medication?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    stroke   = st.selectbox("Prior stroke?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    diabetes = st.selectbox("Diabetes?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    hyp      = st.selectbox("Hypertensive?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
with col2:
    education = st.selectbox("Education", [1,2,3,4],
                  format_func=lambda x:{1:"< High School",2:"High School",
                                        3:"Some College",4:"College+"}[x])
    totchol  = st.number_input("Total Cholesterol (mg/dL)", 100, 600, 200)
    sysbp    = st.number_input("Systolic BP (mmHg)", 83, 295, 120)
    diabp    = st.number_input("Diastolic BP (mmHg)", 48, 142, 80)
    bmi      = st.number_input("BMI (kg/m²)", 15.0, 55.0, 25.0)
    hr       = st.number_input("Heart Rate (bpm)", 44, 143, 75)
    glucose  = st.number_input("Glucose (mg/dL)", 40, 400, 80)

def draw_gauge(prob):
    if prob < 0.15:
        bar_color = "#2d6a4f"
    elif prob < 0.30:
        bar_color = "#b5770d"
    else:
        bar_color = "#c0392b"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 42, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b8fa8",
                     "tickfont": {"color": "#8b8fa8", "size": 11}},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "#1a1d27",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  15],  "color": "#1a3d2b"},
                {"range": [15, 30],  "color": "#3d2e09"},
                {"range": [30, 100], "color": "#3d0f0f"},
            ],
            "threshold": {
                "line": {"color": "#aaaaaa", "width": 3},
                "thickness": 0.8,
                "value": 15.2
            }
        },
        title={"text": "10-year CHD probability<br><span style='font-size:11px;color:#8b8fa8'>▲ grey marker = population average (15.2%)</span>",
               "font": {"color": "#e8e8e8", "size": 14}}
    ))
    fig.update_layout(
        paper_bgcolor="#0f1117",
        font={"color": "white"},
        height=300,
        margin=dict(t=60, b=10, l=30, r=30)
    )
    return fig

def draw_coverage():
    targets   = [80, 85, 90, 95]
    empirical = [81.4, 85.9, 92.1, 96.5]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{t}% target" for t in targets],
        y=empirical,
        marker_color="#e05c5c",
        text=[f"{e}%" for e in empirical],
        textposition="outside",
        textfont={"color": "white", "size": 11}
    ))
    for t, label in zip(targets, [f"{t}% target" for t in targets]):
        fig.add_hline(y=t, line_dash="dash", line_color="#aaaaaa",
                      line_width=1.5,
                      annotation_text=f"target {t}%",
                      annotation_font_color="#aaaaaa",
                      annotation_font_size=9)
    fig.update_layout(
        title={"text": "Conformal Coverage — Target vs Actual",
               "font": {"color": "#e8e8e8", "size": 13}},
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d27",
        font={"color": "#8b8fa8"},
        yaxis={"range": [75, 100], "gridcolor": "#2a2d3e",
               "title": "Empirical coverage (%)"},
        xaxis={"gridcolor": "#2a2d3e"},
        height=280,
        margin=dict(t=50, b=20, l=50, r=20),
        showlegend=False
    )
    return fig

if st.button("Calculate Risk"):
    try:
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
        pred_set = [c for c in [0, 1] if scores[c] <= artifacts["q_hat"]]

        st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
        st.plotly_chart(draw_gauge(prob_chd), use_container_width=True)

        if pred_set == [0]:
            st.success("**Prediction set {0}** — Model is confident: LOW risk")
        elif pred_set == [1]:
            st.error("**Prediction set {1}** — Model is confident: HIGH risk")
        else:
            st.warning("**Prediction set {0, 1}** — Model is uncertain — risk could go either way")

        st.markdown('<div class="section-label">Model Confidence</div>', unsafe_allow_html=True)
        st.plotly_chart(draw_coverage(), use_container_width=True)
        st.caption(
            "Dashed lines = target coverage level. Bars = empirical coverage on held-out test set (n=848). "
            "All four alpha levels pass. Marginal coverage only — not for clinical use."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
