import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="CHD Risk Calculator", page_icon="🫀", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0f1117; color: #e8e8e8; }
.main { background-color: #0f1117; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem 2rem; max-width: 820px; }
.hero { border-left: 3px solid #e05c5c; padding: 1.2rem 1.5rem; margin-bottom: 2rem;
        background: linear-gradient(135deg,#1a1d27,#161822); border-radius: 0 12px 12px 0; }
.hero h1 { font-family: 'DM Serif Display',serif; font-size:1.9rem; font-weight:400;
           color:#fff; margin:0 0 0.3rem 0; }
.hero p  { color:#8b8fa8; font-size:0.8rem; margin:0; letter-spacing:.03em; }
.section-label { font-size:0.7rem; font-weight:600; letter-spacing:.12em;
                 text-transform:uppercase; color:#e05c5c; margin:1.8rem 0 0.6rem 0; }
div[data-testid="stButton"] > button {
    background:#e05c5c; color:white; border:none; border-radius:8px;
    padding:0.65rem 2rem; font-size:0.9rem; font-weight:600; width:100%; margin-top:1rem; }
div[data-testid="stButton"] > button:hover { background:#c94c4c; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifacts = load_artifacts()
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🫀 10-Year CHD Risk Calculator</h1>
  <p>FRAMINGHAM HEART STUDY · LOGISTIC REGRESSION + CONFORMAL PREDICTION · AUC 0.748</p>
</div>""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
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

# ── Gauge function ────────────────────────────────────────────────────────────
def draw_gauge(prob):
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.2, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colour bands: green 0–15%, yellow 15–30%, red 30–100%
    bands = [(0, 0.15, "#2d6a4f"), (0.15, 0.30, "#b5770d"), (0.30, 1.0, "#7a1e1e")]
    for lo, hi, color in bands:
        theta1 = 180 - lo * 180
        theta2 = 180 - hi * 180
        arc = Arc((0, 0), 2, 2, angle=0, theta1=theta2, theta2=theta1,
                  color=color, lw=22, solid_capstyle="butt")
        ax.add_patch(arc)

    # Needle
    angle_rad = np.pi * (1 - prob)
    nx = 0.75 * np.cos(angle_rad)
    ny = 0.75 * np.sin(angle_rad)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="white",
                                lw=2.5, mutation_scale=15))
    ax.plot(0, 0, "o", color="white", markersize=8, zorder=5)

    # Population average marker
    avg_angle = np.pi * (1 - 0.152)
    ax.plot(np.cos(avg_angle), np.sin(avg_angle), "|",
            color="#aaaaaa", markersize=14, markeredgewidth=2)
    ax.text(1.05 * np.cos(avg_angle), 1.05 * np.sin(avg_angle),
            "avg", ha="center", va="bottom", color="#aaaaaa", fontsize=7)

    # Percentage label
    ax.text(0, -0.12, f"{prob*100:.1f}%",
            ha="center", va="center", fontsize=28,
            fontweight="bold", color="white")
    ax.text(0, -0.28, "10-year CHD probability",
            ha="center", va="center", fontsize=8,
            color="#8b8fa8")

    # Scale labels
    for pct, label in [(0, "0%"), (0.15, "15%"), (0.30, "30%"), (1.0, "100%")]:
        a = np.pi * (1 - pct)
        ax.text(1.15*np.cos(a), 1.15*np.sin(a), label,
                ha="center", va="center", color="#8b8fa8", fontsize=7)

    plt.tight_layout(pad=0)
    return fig

# ── Coverage bar chart ────────────────────────────────────────────────────────
def draw_coverage():
    fig, ax = plt.subplots(figsize=(6, 2.2), facecolor="#1a1d27")
    ax.set_facecolor("#1a1d27")

    targets   = [80, 85, 90, 95]
    empirical = [81.4, 85.9, 92.1, 96.5]
    x = np.arange(len(targets))

    bars = ax.bar(x, empirical, color="#e05c5c", alpha=0.85, width=0.5, zorder=2)
    for i, (t, e) in enumerate(zip(targets, empirical)):
        ax.axhline(t, xmin=(i)/len(targets)+0.05,
                   xmax=(i+1)/len(targets)-0.05,
                   color="white", lw=1.5, linestyle="--", zorder=3)
        ax.text(i, e + 0.4, f"{e}%", ha="center", va="bottom",
                color="white", fontsize=8, fontweight="600")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}% target" for t in targets], color="#8b8fa8", fontsize=8)
    ax.set_ylim(75, 100)
    ax.set_ylabel("Empirical coverage", color="#8b8fa8", fontsize=8)
    ax.tick_params(colors="#8b8fa8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3e")
    ax.set_title("Conformal Coverage — Target vs Actual", color="#e8e8e8",
                 fontsize=9, fontweight="500", pad=8)
    ax.yaxis.set_tick_params(labelcolor="#8b8fa8")
    ax.grid(axis="y", color="#2a2d3e", zorder=1)
    plt.tight_layout()
    return fig

# ── Predict ───────────────────────────────────────────────────────────────────
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

        st.markdown('<div class="section-label">Result</div>',
                    unsafe_allow_html=True)

        # Gauge
        st.pyplot(draw_gauge(prob_chd))
        plt.close("all")

        # Prediction set verdict
        if pred_set == [0]:
            st.success("**Prediction set {0}** — Model is confident: LOW risk")
        elif pred_set == [1]:
            st.error("**Prediction set {1}** — Model is confident: HIGH risk")
        else:
            st.warning("**Prediction set {0, 1}** — Model is uncertain — risk could go either way")

        # Coverage chart
        st.markdown('<div class="section-label">Model Confidence</div>',
                    unsafe_allow_html=True)
        st.pyplot(draw_coverage())
        plt.close("all")

        st.caption(
            "Dashed lines = target coverage. Bars = empirical coverage on held-out test set. "
            "All four alpha levels pass. Marginal guarantee only — conditional coverage for "
            "high-risk patients is lower. Not for clinical use."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("Check that model_artifacts.pkl is uploaded and scikit-learn==1.6.1 is in requirements.txt")
