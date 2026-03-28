from pathlib import Path
import pandas as pd
import streamlit as st
from joblib import load

# -------------------- Page / Theme --------------------
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="💸",
    layout="wide",
)

st.markdown("""
<style>
/* section titles */
.block-title { font-size: 1.15rem; font-weight: 600; margin-bottom: 0.35rem; }
/* muted caption text */
.muted { color: var(--text-color-secondary); font-size: 0.9rem; }
/* result card */
.result-card {
  border: 1px solid rgba(250,250,250,0.08);
  border-radius: 14px; padding: 18px 20px; margin-top: 8px;
  background: rgba(127,127,127,0.07);
}
.result-value { font-size: 2.2rem; font-weight: 700; line-height: 1.1; }
.badge {
  display:inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 0.8rem; background: rgba(127,127,127,0.15);
  border: 1px solid rgba(127,127,127,0.25);
}
.small { font-size: 0.9rem; }
.hr { border-top: 1px solid rgba(127,127,127,0.2); margin: 10px 0 16px 0; }
</style>
""", unsafe_allow_html=True)

# -------------------- Paths & Utilities --------------------
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

def read_best_model_filename():
    pointer = MODELS_DIR / "best_model.txt"
    if pointer.exists():
        return pointer.read_text().strip()
    # Fallback by scanning lowest RMSE
    csvs = list(MODELS_DIR.glob("*_metrics.csv"))
    if not csvs:
        return None
    dfs = [pd.read_csv(p).assign(file=p.name) for p in csvs]
    ranked = pd.concat(dfs).sort_values("RMSE")
    return ranked.iloc[0]["file"].replace("_metrics.csv", ".joblib")

def load_metrics_for(joblib_name: str) -> pd.DataFrame | None:
    csv = MODELS_DIR / joblib_name.replace(".joblib", "_metrics.csv")
    if csv.exists():
        return pd.read_csv(csv)
    return None

@st.cache_resource
def load_pipeline(joblib_path: Path):
    return load(joblib_path)

# -------------------- Header --------------------
left, right = st.columns([0.75, 0.25])
with left:
    st.title("Medical Insurance Cost Predictor")
    st.caption("Enter a profile or pick a preset. The app loads your best trained model automatically.")
with right:
    st.markdown("<div style='text-align:right'><span class='badge'>v1.0</span></div>", unsafe_allow_html=True)

# -------------------- Model loading --------------------
best_model_file = read_best_model_filename()
if not best_model_file:
    st.error("No trained models found. Please run your training notebooks to generate artifacts in `models/`.")
    st.stop()

pipe = load_pipeline(MODELS_DIR / best_model_file)
metrics_df = load_metrics_for(best_model_file)
model_label = best_model_file.replace("_", " ").replace(".joblib", "").title()

# -------------------- Sidebar: presets & info --------------------
with st.sidebar:
    st.subheader("⚙️ Model")
    st.write(f"**Loaded:** `{best_model_file}`")
    if metrics_df is not None:
        m = metrics_df.iloc[0]
        st.metric("RMSE (test)", f"{m['RMSE']:.0f}")
        st.metric("MAE (test)", f"{m['MAE']:.0f}")
        st.caption(f"R² (test): **{m['R2']:.3f}**")
    else:
        st.caption("_Metrics file not found for this model._")

    st.markdown("---")
    st.subheader("🎛️ Presets")
    preset = st.selectbox(
        "Quick scenarios",
        [
            "Custom (manual inputs)",
            "Young non-smoker (baseline)",
            "Middle-age smoker (higher risk)",
            "Senior non-smoker (moderate risk)",
            "High BMI non-smoker"
        ],
        index=0
    )

# -------------------- Input form --------------------
st.markdown("<div class='block-title'>Profile</div>", unsafe_allow_html=True)
st.markdown("<span class='muted'>Adjust fields or choose a preset from the sidebar.</span>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

# Defaults
defaults = {
    "age": 30, "bmi": 26.5, "children": 0,
    "sex": "male", "smoker": "no", "region": "southwest"
}
if preset == "Young non-smoker (baseline)":
    defaults.update({"age": 24, "bmi": 22.0, "children": 0, "sex": "male", "smoker": "no", "region": "southwest"})
elif preset == "Middle-age smoker (higher risk)":
    defaults.update({"age": 45, "bmi": 28.0, "children": 1, "sex": "female", "smoker": "yes", "region": "southeast"})
elif preset == "Senior non-smoker (moderate risk)":
    defaults.update({"age": 65, "bmi": 27.0, "children": 2, "sex": "female", "smoker": "no", "region": "northeast"})
elif preset == "High BMI non-smoker":
    defaults.update({"age": 35, "bmi": 35.0, "children": 1, "sex": "male", "smoker": "no", "region": "northwest"})

with c1:
    age = st.number_input("Age", min_value=18, max_value=100, value=int(defaults["age"]))
    sex = st.selectbox("Sex", options=["male", "female"], index=["male","female"].index(defaults["sex"]))
with c2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1, value=float(defaults["bmi"]))
    smoker = st.selectbox("Smoker", options=["yes", "no"], index=["yes","no"].index(defaults["smoker"]))
with c3:
    children = st.number_input("Children", min_value=0, max_value=10, value=int(defaults["children"]))
    region = st.selectbox("Region", options=["southwest","southeast","northwest","northeast"],
                          index=["southwest","southeast","northwest","northeast"].index(defaults["region"]))

# -------------------- Predict --------------------
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
colA, colB = st.columns([0.2, 0.8])
with colA:
    predict_btn = st.button("🔮 Predict", type="primary")
with colB:
    st.write("")

if predict_btn:
    row = pd.DataFrame([{
        "age": age, "bmi": bmi, "children": children,
        "sex": sex, "smoker": smoker, "region": region
    }])

    try:
        pred = float(pipe.predict(row)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Result card
    st.markdown("<div class='block-title'>Estimated Annual Charges</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown(
            f"<div class='result-card'><div class='result-value'>${pred:,.2f}</div>"
            f"<div class='muted small'>{model_label}</div></div>",
            unsafe_allow_html=True
        )

    # Compact input echo (helpful in screenshots)
    st.markdown("<div class='block-title' style='margin-top:14px'>Inputs</div>", unsafe_allow_html=True)
    st.dataframe(row.style.format({"bmi": "{:.1f}"}), use_container_width=True)