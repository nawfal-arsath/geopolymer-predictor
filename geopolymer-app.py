"""
Geopolymer Concrete Compressive Strength Predictor
=====================================================
• Dataset: CIVIL_FINAL_DATASET.xlsx (Fly Ash + GGBS mixes, all numerical)
• Features: Fly_Ash_kg_m3, GGBS_kg_m3, NaOH_Molarity_M, Na2SiO3_NaOH_Ratio,
            Water_Binder_Ratio, Curing_Temp_C, Age_days
• Target:   Compressive_Strength_MPa
• 5 ML models: Linear Regression, Ridge, SVR, Random Forest, Gradient Boosting
• Forward prediction (mix → strength) + Inverse prediction (strength → optimal mix)
• EDA, residual analysis, feature importances

Run with:
    streamlit run geopolymer-app.py

Place CIVIL_FINAL_DATASET.xlsx in the same directory as this script.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geopolymer Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Light theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

.stApp { background: #f5f6fa; color: #1a1d27; }

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e5ee;
}

.stNumberInput label, .stSelectbox label { color: #4a5068 !important; font-weight: 500; font-size: 13px; }

.stNumberInput input {
    background: #ffffff !important;
    border: 1.5px solid #d1d5e8 !important;
    border-radius: 8px !important;
    color: #1a1d27 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 15px !important;
    font-weight: 500 !important;
}
.stNumberInput input:focus {
    border-color: #4f6ef7 !important;
    box-shadow: 0 0 0 3px rgba(79,110,247,0.1) !important;
}

.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1.5px solid #d1d5e8 !important;
    border-radius: 8px !important;
    color: #1a1d27 !important;
}

.card {
    background: #ffffff;
    border: 1px solid #e2e5ee;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #e2e5ee;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
    margin-bottom: 10px;
}
.metric-card .mlabel { font-size: 11px; color: #8892b0; letter-spacing: 0.8px; text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }
.metric-card .mvalue { font-size: 26px; font-weight: 700; color: #4f6ef7; font-family: 'DM Mono', monospace; }
.metric-card .msub { font-size: 11px; color: #8892b0; margin-top: 2px; }

.result-banner {
    background: linear-gradient(135deg, #eef2ff, #e8f5e9);
    border: 2px solid #4f6ef7;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 18px 0;
}
.result-banner .rval { font-size: 56px; font-weight: 800; color: #4f6ef7; font-family: 'DM Mono', monospace; }
.result-banner .rlabel { font-size: 14px; color: #6b7aa1; font-weight: 500; margin-top: 4px; }
.result-banner .rgrade { font-size: 19px; font-weight: 700; margin-top: 8px; }

.sec-header {
    font-size: 17px;
    font-weight: 700;
    color: #1a1d27;
    border-bottom: 2px solid #e2e5ee;
    padding-bottom: 8px;
    margin: 20px 0 14px 0;
}

.info-box {
    background: #eef2ff;
    border-left: 4px solid #4f6ef7;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 13px;
    color: #3d4f8a;
}
.warn-box {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 13px;
    color: #78460a;
}

.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #e2e5ee;
}
.stTabs [data-baseweb="tab"] { color: #6b7aa1; font-weight: 600; font-size: 13px; }
.stTabs [aria-selected="true"] {
    background: #4f6ef7 !important;
    color: #ffffff !important;
    border-radius: 7px;
}

.stButton > button {
    background: #4f6ef7;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 14px;
    padding: 11px 28px;
    width: 100%;
    transition: all 0.18s;
}
.stButton > button:hover {
    background: #3d5ce0;
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(79,110,247,0.3);
}

h1, h2, h3 { color: #1a1d27 !important; }

.group-label {
    font-size: 12px;
    font-weight: 600;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 10px;
}

.inv-card {
    background: #f0f4ff;
    border: 1px solid #c7d2fe;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.inv-card-title { font-size: 11px; color: #8892b0; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }
.inv-card-value { font-size: 20px; font-weight: 700; color: #4f6ef7; font-family: 'DM Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib light theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#f8f9fd",
    "axes.edgecolor":    "#d1d5e8",
    "axes.labelcolor":   "#4a5068",
    "xtick.color":       "#6b7aa1",
    "ytick.color":       "#6b7aa1",
    "text.color":        "#1a1d27",
    "grid.color":        "#e2e5ee",
    "grid.alpha":        0.7,
    "patch.edgecolor":   "#d1d5e8",
    "legend.facecolor":  "#ffffff",
    "legend.edgecolor":  "#d1d5e8",
    "legend.labelcolor": "#1a1d27",
    "figure.dpi":        110,
})

PALETTE = ["#4f6ef7", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#f97316", "#ec4899"]

# Column metadata
FEATURE_COLS = [
    "Fly_Ash_kg_m3",
    "GGBS_kg_m3",
    "NaOH_Molarity_M",
    "Na2SiO3_NaOH_Ratio",
    "Water_Binder_Ratio",
    "Curing_Temp_C",
    "Age_days",
]
TARGET_COL = "Compressive_Strength_MPa"

FEATURE_LABELS = {
    "Fly_Ash_kg_m3":       "Fly Ash (kg/m³)",
    "GGBS_kg_m3":          "GGBS (kg/m³)",
    "NaOH_Molarity_M":     "NaOH Molarity (M)",
    "Na2SiO3_NaOH_Ratio":  "Na₂SiO₃/NaOH Ratio",
    "Water_Binder_Ratio":  "Water/Binder Ratio",
    "Curing_Temp_C":       "Curing Temp (°C)",
    "Age_days":            "Age (days)",
}

FEATURE_ICONS = {
    "Fly_Ash_kg_m3":       "🪨",
    "GGBS_kg_m3":          "⚙️",
    "NaOH_Molarity_M":     "⚗️",
    "Na2SiO3_NaOH_Ratio":  "🔬",
    "Water_Binder_Ratio":  "💧",
    "Curing_Temp_C":       "🌡️",
    "Age_days":            "📅",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_dataset():
    try:
        df = pd.read_excel("CIVIL_FINAL_DATASET.xlsx")
    except FileNotFoundError:
        st.error("❌ CIVIL_FINAL_DATASET.xlsx not found. Place it in the same folder as this script.")
        st.stop()

    # Ensure correct dtypes
    for col in FEATURE_COLS + [TARGET_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=[TARGET_COL], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def build_models():
    return {
        "Linear Regression":  LinearRegression(),
        "Ridge Regression":   Ridge(alpha=1.0),
        "SVR (RBF)":          SVR(kernel="rbf", C=10, epsilon=0.1),
        "Random Forest":      RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                                         max_depth=4, random_state=42),
    }


@st.cache_data
def train_pipeline(df_json: str):
    from io import StringIO
    df = pd.read_json(StringIO(df_json), orient="split")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    imputer = SimpleImputer(strategy="mean")
    X_imp   = imputer.fit_transform(X)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models_def = build_models()
    results    = {}
    trained    = {}

    for name, m in models_def.items():
        m.fit(X_train, y_train)
        preds     = m.predict(X_test)
        cv_scores = cross_val_score(m, X_scaled, y, cv=5, scoring="r2")
        results[name] = {
            "R²":           round(r2_score(y_test, preds), 4),
            "MAE":          round(mean_absolute_error(y_test, preds), 3),
            "RMSE":         round(mean_squared_error(y_test, preds) ** 0.5, 3),
            "CV R² (mean)": round(cv_scores.mean(), 4),
            "CV R² (std)":  round(cv_scores.std(), 4),
        }
        trained[name] = m

    results_df = pd.DataFrame(results).T

    return dict(
        trained=trained,
        scaler=scaler,
        imputer=imputer,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        results_df=results_df,
        X_scaled=X_scaled,
        y=y,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_input(row_dict, p):
    row = pd.DataFrame([row_dict])[FEATURE_COLS]
    row_imp    = p["imputer"].transform(row)
    row_scaled = p["scaler"].transform(row_imp)
    return row_scaled


def get_grade(s):
    if   s < 20:  return "M15 – Low",           "#ef4444"
    elif s < 25:  return "M20 – Standard",       "#f97316"
    elif s < 30:  return "M25 – Standard",       "#f59e0b"
    elif s < 40:  return "M30-M35 – Moderate",   "#10b981"
    elif s < 55:  return "M40-M50 – High",       "#059669"
    elif s < 80:  return "M55-M75 – Very High",  "#0d9488"
    else:         return "M80+ – Ultra High",    "#0891b2"


def inverse_predict(target, model, p, df):
    num_rngs = {c: (df[c].dropna().min(), df[c].dropna().max()) for c in FEATURE_COLS}

    def obj(vals):
        row_dict = dict(zip(FEATURE_COLS, vals))
        try:
            x = preprocess_input(row_dict, p)
            return (model.predict(x)[0] - target) ** 2
        except Exception:
            return 1e9

    x0     = [np.mean(num_rngs[c]) for c in FEATURE_COLS]
    bounds = [num_rngs[c] for c in FEATURE_COLS]
    res    = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-9})

    best_combo = dict(zip(FEATURE_COLS, res.x))
    x_best     = preprocess_input(best_combo, p)
    achieved   = model.predict(x_best)[0]
    return best_combo, achieved


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & TRAIN
# ══════════════════════════════════════════════════════════════════════════════

df_raw = load_dataset()

# ── Age filter in sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ Geopolymer Predictor")
    st.markdown("---")

    age_options = sorted(df_raw["Age_days"].dropna().unique().tolist())
    selected_age = st.selectbox(
        "Filter by Curing Age (days)",
        ["All"] + [str(int(a)) for a in age_options],
        index=0,
        help="Filter dataset to a specific curing age, or use all data together."
    )

    mode = st.radio(
        "Mode",
        ["🔮 Predict Strength", "🎯 Target → Mix Design"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**ML Model**")
    model_names = list(build_models().keys())
    selected_model_name = st.selectbox("Model", model_names, index=3, label_visibility="collapsed")
    st.markdown("---")


# Apply age filter
if selected_age == "All":
    df = df_raw.copy()
else:
    df = df_raw[df_raw["Age_days"] == float(selected_age)].copy()

# Train
p              = train_pipeline(df.to_json(orient="split"))
selected_model = p["trained"][selected_model_name]
model_score    = p["results_df"].loc[selected_model_name, "R²"]

with st.sidebar:
    best_name = p["results_df"]["R²"].idxmax()
    st.markdown(f"""
    <div class="metric-card">
        <div class="mlabel">Samples</div>
        <div class="mvalue">{len(df)}</div>
    </div>
    <div class="metric-card">
        <div class="mlabel">Model R²</div>
        <div class="mvalue">{model_score:.4f}</div>
        <div class="msub">20 % hold-out test</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        🏆 Best: <b>{best_name}</b><br>
        R² = {p['results_df'].loc[best_name,'R²']:.4f} &nbsp;|&nbsp;
        MAE = {p['results_df'].loc[best_name,'MAE']:.2f} MPa
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:14px 0 4px 0;">
<h1 style="font-size:28px;font-weight:800;color:#1a1d27;margin:0;">
🏗️ Geopolymer Concrete — Compressive Strength Predictor
</h1>
<p style="color:#8892b0;font-size:13px;margin-top:4px;">
Fly Ash + GGBS blended mixes · ML-powered prediction · Mix design optimizer · Model benchmarking
</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict / Design",
    "📊 EDA",
    "🤖 Model Performance",
    "📂 Dataset",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT / DESIGN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    if mode == "🔮 Predict Strength":
        st.markdown('<div class="sec-header">Input Mix Parameters</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2, gap="large")

        def num_input(col, feat):
            mn  = float(df[feat].dropna().min())
            mx  = float(df[feat].dropna().max())
            med = float(df[feat].dropna().median())
            return col.number_input(
                FEATURE_LABELS[feat],
                min_value=mn, max_value=mx, value=med,
                step=0.001, format="%.3f",
                help=f"Dataset range: {mn:.3f} – {mx:.3f}"
            )

        with col_a:
            st.markdown('<div class="group-label">Binder & Activator</div>', unsafe_allow_html=True)
            fly_ash      = num_input(col_a, "Fly_Ash_kg_m3")
            ggbs         = num_input(col_a, "GGBS_kg_m3")
            naoh_mol     = num_input(col_a, "NaOH_Molarity_M")
            na2sio3_naoh = num_input(col_a, "Na2SiO3_NaOH_Ratio")

        with col_b:
            st.markdown('<div class="group-label">Mix & Curing</div>', unsafe_allow_html=True)
            wb_ratio    = num_input(col_b, "Water_Binder_Ratio")
            curing_temp = num_input(col_b, "Curing_Temp_C")

            age_vals = sorted(df["Age_days"].dropna().unique().tolist())
            age_days = col_b.selectbox(
                FEATURE_LABELS["Age_days"],
                options=[int(a) for a in age_vals],
                index=len(age_vals) - 1,
            )

        st.markdown("---")
        if st.button("🚀 Predict Compressive Strength"):
            row_dict = {
                "Fly_Ash_kg_m3":      fly_ash,
                "GGBS_kg_m3":         ggbs,
                "NaOH_Molarity_M":    naoh_mol,
                "Na2SiO3_NaOH_Ratio": na2sio3_naoh,
                "Water_Binder_Ratio": wb_ratio,
                "Curing_Temp_C":      curing_temp,
                "Age_days":           float(age_days),
            }
            x_in     = preprocess_input(row_dict, p)
            pred_val = selected_model.predict(x_in)[0]
            grade, gc = get_grade(pred_val)

            st.markdown(f"""
            <div class="result-banner">
                <div class="rval">{pred_val:.2f} MPa</div>
                <div class="rlabel">Predicted Compressive Strength</div>
                <div class="rgrade" style="color:{gc}">Grade: {grade}</div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"""<div class="metric-card"><div class="mlabel">Model</div>
                <div class="mvalue" style="font-size:14px;">{selected_model_name}</div></div>""",
                unsafe_allow_html=True)
            m2.markdown(f"""<div class="metric-card"><div class="mlabel">R²</div>
                <div class="mvalue">{model_score:.4f}</div></div>""", unsafe_allow_html=True)
            m3.markdown(f"""<div class="metric-card"><div class="mlabel">MAE</div>
                <div class="mvalue">{p['results_df'].loc[selected_model_name,'MAE']}</div>
                <div class="msub">MPa</div></div>""", unsafe_allow_html=True)
            m4.markdown(f"""<div class="metric-card"><div class="mlabel">RMSE</div>
                <div class="mvalue">{p['results_df'].loc[selected_model_name,'RMSE']}</div>
                <div class="msub">MPa</div></div>""", unsafe_allow_html=True)

            # Strength grade bar
            st.markdown('<div class="sec-header">Strength Grade Context</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(9, 2.2))
            categories  = ["Low\n<20", "Std\n20-30", "Mod\n30-40", "High\n40-55", "V.High\n55-80", "Ultra\n≥80"]
            thresholds  = [20, 30, 40, 55, 80, 165]
            bar_colors  = ["#fca5a5", "#fed7aa", "#fde68a", "#6ee7b7", "#5eead4", "#67e8f9"]
            prev = 0
            for cat, thresh, clr in zip(categories, thresholds, bar_colors):
                ax.barh(0, thresh - prev, left=prev, color=clr, alpha=0.85, height=0.5)
                prev = thresh
            ax.axvline(pred_val, color="#4f6ef7", linewidth=3, linestyle="--",
                       label=f"Predicted: {pred_val:.1f} MPa")
            ax.set_xlim(0, 165); ax.set_yticks([]); ax.set_xlabel("Compressive Strength (MPa)", fontsize=10)
            ax.legend(fontsize=9); ax.grid(False)
            prev2 = 0
            for cat, thresh, _ in zip(categories, thresholds, bar_colors):
                mid = prev2 + (thresh - prev2) / 2
                ax.text(mid, 0, cat, ha="center", va="center", fontsize=7, color="#1a1d27", fontweight="600")
                prev2 = thresh
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # All models comparison
            st.markdown('<div class="sec-header">All Models — Prediction Comparison</div>', unsafe_allow_html=True)
            preds_all  = {nm: m.predict(x_in)[0] for nm, m in p["trained"].items()}
            names_all  = list(preds_all.keys())
            vals_all   = list(preds_all.values())
            colors_all = [PALETTE[0] if n == selected_model_name else "#d1d5e8" for n in names_all]
            fig, ax = plt.subplots(figsize=(9, 3.5))
            bars = ax.bar(names_all, vals_all, color=colors_all, edgecolor="#ffffff", linewidth=1.2, width=0.55)
            ax.axhline(preds_all[selected_model_name], color="#f59e0b", linestyle="--",
                       linewidth=1.5, label="Selected model")
            for bar, v in zip(bars, vals_all):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_ylabel("Predicted Strength (MPa)", fontsize=10)
            ax.set_title("Prediction Across All Models", fontsize=11, fontweight="700")
            ax.set_xticklabels(names_all, rotation=15, ha="right")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── TARGET → MIX DESIGN ──────────────────────────────────────────────────
    else:
        st.markdown('<div class="sec-header">Target Strength → Optimal Mix Design</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
        Enter your <b>desired compressive strength</b>. The optimiser fine-tunes all
        numerical mix parameters using L-BFGS-B to find the mix that best achieves it.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            target_strength = st.number_input(
                "Target Compressive Strength (MPa)",
                min_value=5.0, max_value=200.0, value=100.0, step=0.5, format="%.1f"
            )
        with c2:
            st.markdown(""); st.markdown("")
            find_btn = st.button("⚡ Find Optimal Mix")

        st.markdown(f"""<div class="warn-box">
        ⚠️ Dataset strength range: <b>{df[TARGET_COL].min():.1f} –
        {df[TARGET_COL].max():.1f} MPa</b>.
        Targets outside this range extrapolate beyond training data.
        </div>""", unsafe_allow_html=True)

        if find_btn:
            with st.spinner("Optimising mix design… (~10 s)"):
                best_combo, achieved = inverse_predict(target_strength, selected_model, p, df)

            diff = abs(achieved - target_strength)
            st.markdown(f"""
            <div class="result-banner">
                <div class="rval">{achieved:.2f} MPa</div>
                <div class="rlabel">Achieved Predicted Strength (Target: {target_strength} MPa)</div>
                <div class="rgrade" style="color:{'#059669' if diff < 3 else '#f59e0b'}">
                    {'✅ Excellent match' if diff < 3 else f'⚡ Offset: {diff:.2f} MPa'}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sec-header">Recommended Mix Design</div>', unsafe_allow_html=True)
            rec1, rec2 = st.columns(2)
            for i, (key, val) in enumerate(best_combo.items()):
                cont = rec1 if i % 2 == 0 else rec2
                fmt  = f"{val:.3f}" if key != "Age_days" else f"{int(round(val))} days"
                cont.markdown(f"""<div class="inv-card">
                    <div class="inv-card-title">{FEATURE_ICONS.get(key,'•')} {FEATURE_LABELS.get(key, key)}</div>
                    <div class="inv-card-value">{fmt}</div></div>""",
                    unsafe_allow_html=True)

            # Recommended vs dataset average
            st.markdown('<div class="sec-header">Recommended vs Dataset Average</div>', unsafe_allow_html=True)
            fig, axes = plt.subplots(2, 4, figsize=(14, 7))
            axes_flat = axes.flatten()
            for ax_i, key in enumerate(FEATURE_COLS):
                rv  = best_combo[key]
                avg = df[key].mean()
                bars_ = axes_flat[ax_i].bar(
                    ["Recommended", "Dataset Avg"], [rv, avg],
                    color=["#4f6ef7", "#d1d5e8"], edgecolor="#ffffff", width=0.5
                )
                for bar, v in zip(bars_, [rv, avg]):
                    axes_flat[ax_i].text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01 * abs(v) + 0.1,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
                    )
                axes_flat[ax_i].set_title(FEATURE_LABELS[key], fontsize=9, fontweight="bold")
                axes_flat[ax_i].set_ylabel("Value"); axes_flat[ax_i].grid(axis="y", alpha=0.4)
            # hide last empty panel
            axes_flat[-1].set_visible(False)
            plt.suptitle(f"Recommended Mix vs Dataset Average  (Target: {target_strength} MPa)",
                         fontsize=11, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("**Correlation Heatmap**")
        all_cols = FEATURE_COLS + [TARGET_COL]
        corr  = df[all_cols].corr()
        mask  = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                    mask=mask, linewidths=0.5, ax=ax,
                    annot_kws={"size": 8, "weight": "bold"},
                    cbar_kws={"shrink": 0.8},
                    xticklabels=[FEATURE_LABELS.get(c, c) for c in all_cols],
                    yticklabels=[FEATURE_LABELS.get(c, c) for c in all_cols])
        ax.set_title("Correlation – All Features", fontsize=11, fontweight="bold")
        plt.xticks(rotation=30, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r1c2:
        st.markdown("**Strength Distribution by Curing Age**")
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, age in enumerate(sorted(df["Age_days"].dropna().unique())):
            sub = df[df["Age_days"] == age][TARGET_COL].dropna()
            ax.hist(sub, bins=15, alpha=0.65, label=f"{int(age)} days",
                    color=PALETTE[i % len(PALETTE)], edgecolor="none")
        ax.set_xlabel("Compressive Strength (MPa)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title("Strength Distribution by Age", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Scatter: each feature vs strength
    st.markdown("**Feature vs Compressive Strength**")
    scatter_feats = [f for f in FEATURE_COLS if f != "Age_days"]
    sc_cols = st.columns(len(scatter_feats))
    for i, feat in enumerate(scatter_feats):
        fig, ax = plt.subplots(figsize=(3.5, 3))
        for j, age in enumerate(sorted(df["Age_days"].dropna().unique())):
            sub = df[df["Age_days"] == age][[feat, TARGET_COL]].dropna()
            ax.scatter(sub[feat], sub[TARGET_COL],
                       label=f"{int(age)}d", alpha=0.65, s=24,
                       edgecolors="none", color=PALETTE[j % len(PALETTE)])
        sub_all = df[[feat, TARGET_COL]].dropna()
        if len(sub_all) > 1:
            c_ = np.polyfit(sub_all[feat], sub_all[TARGET_COL], 1)
            xr = np.linspace(sub_all[feat].min(), sub_all[feat].max(), 100)
            ax.plot(xr, np.polyval(c_, xr), color="#1a1d27", linestyle="--", linewidth=1.4)
        ax.set_xlabel(FEATURE_LABELS[feat], fontsize=7)
        ax.set_ylabel("Strength (MPa)", fontsize=7)
        ax.set_title(FEATURE_LABELS[feat], fontsize=8, fontweight="bold")
        ax.legend(fontsize=5); ax.grid(True, alpha=0.4)
        plt.tight_layout()
        sc_cols[i].pyplot(fig); plt.close()

    # Boxplot: strength by curing temp bins
    st.markdown("**Strength by Curing Temperature Band**")
    df_tmp = df.copy()
    df_tmp["Temp_band"] = pd.cut(df_tmp["Curing_Temp_C"], bins=4,
                                  labels=["Low", "Med-Low", "Med-High", "High"])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df_tmp, x="Temp_band", y=TARGET_COL, palette="Set2", ax=ax, linewidth=1.2)
    ax.set_xlabel("Curing Temperature Band", fontsize=10)
    ax.set_ylabel("Compressive Strength (MPa)", fontsize=10)
    ax.set_title("Strength by Curing Temperature Band", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-header">Model Benchmarking & Performance</div>', unsafe_allow_html=True)

    styled_df = p["results_df"].copy()
    styled_df.index.name = "Model"
    st.dataframe(
        styled_df.style
            .background_gradient(cmap="RdYlGn",   subset=["R²", "CV R² (mean)"])
            .background_gradient(cmap="RdYlGn_r", subset=["MAE", "RMSE"])
            .format(precision=4),
        use_container_width=True,
    )

    best_name = p["results_df"]["R²"].idxmax()
    st.markdown(f"""<div class="info-box">
    🏆 Best model (test R²): <b>{best_name}</b> —
    R² = {p['results_df'].loc[best_name,'R²']:.4f},
    MAE = {p['results_df'].loc[best_name,'MAE']:.3f} MPa,
    CV R² = {p['results_df'].loc[best_name,'CV R² (mean)']:.4f} ± {p['results_df'].loc[best_name,'CV R² (std)']:.4f}
    </div>""", unsafe_allow_html=True)

    # Actual vs Predicted — all models
    st.markdown("**Actual vs Predicted — All Models**")
    fig, axes = plt.subplots(1, len(p["trained"]), figsize=(16, 3.8))
    pal_avp = sns.color_palette("tab10", len(p["trained"]))
    for ax, (nm, m), clr in zip(axes, p["trained"].items(), pal_avp):
        yp_i = m.predict(p["X_test"])
        ax.scatter(p["y_test"], yp_i, alpha=0.65, s=30, edgecolors="none", color=clr)
        lims = [min(p["y_test"].min(), yp_i.min()) - 2,
                max(p["y_test"].max(), yp_i.max()) + 2]
        ax.plot(lims, lims, color="#1a1d27", linestyle="--", linewidth=1.2)
        r2_i = r2_score(p["y_test"], yp_i)
        ax.set_title(f"{nm}\nR²={r2_i:.3f}", fontweight="bold", fontsize=8,
                     color=PALETTE[0] if nm == selected_model_name else "#1a1d27")
        ax.set_xlabel("Actual (MPa)", fontsize=8)
        ax.set_ylabel("Predicted (MPa)", fontsize=8)
        ax.grid(True, alpha=0.4)
    plt.suptitle("Actual vs Predicted – All Models", fontsize=11, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Residuals + Feature importances
    ra_c1, ra_c2 = st.columns(2)
    with ra_c1:
        st.markdown(f"**Residual Analysis — {selected_model_name}**")
        y_pred_sel = selected_model.predict(p["X_test"])
        residuals  = p["y_test"].values - y_pred_sel
        fig, axes  = plt.subplots(1, 2, figsize=(9, 3.5))

        axes[0].scatter(y_pred_sel, residuals, alpha=0.7,
                        c=residuals, cmap="RdYlGn", edgecolors="none", s=40)
        axes[0].axhline(0, color="#1a1d27", linewidth=1.5, linestyle="--")
        axes[0].set_xlabel("Predicted (MPa)"); axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs Predicted", fontsize=9, fontweight="bold")
        axes[0].grid(True, alpha=0.4)

        mu_, sd_ = residuals.mean(), residuals.std()
        axes[1].hist(residuals, bins=14, color=PALETTE[0], edgecolor="none", density=True, alpha=0.75)
        xr_r = np.linspace(residuals.min(), residuals.max(), 200)
        axes[1].plot(xr_r, norm.pdf(xr_r, mu_, sd_), color="#f59e0b", linewidth=2)
        axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Density")
        axes[1].set_title(f"Residual Dist. (μ={mu_:.2f}, σ={sd_:.2f})", fontsize=9, fontweight="bold")
        axes[1].grid(True, alpha=0.4)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with ra_c2:
        st.markdown("**Feature Importances (RF & GB)**")
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        feat_labels_list = [FEATURE_LABELS[f] for f in FEATURE_COLS]
        for ax, (nm_, cmap_) in zip(axes, [
            ("Random Forest",     "Blues"),
            ("Gradient Boosting", "Greens"),
        ]):
            m_ = p["trained"][nm_]
            if hasattr(m_, "feature_importances_"):
                imp = m_.feature_importances_
                idx = np.argsort(imp)
                ax.barh([feat_labels_list[i] for i in idx], imp[idx],
                        color=sns.color_palette(cmap_, len(FEATURE_COLS)),
                        edgecolor="none")
                for bar, v in zip(ax.patches, imp[idx]):
                    ax.text(v + 0.004, bar.get_y() + bar.get_height() / 2,
                            f"{v:.3f}", va="center", fontsize=7.5)
                ax.set_title(nm_, fontsize=9, fontweight="bold")
                ax.set_xlabel("Importance")
                ax.grid(axis="x", alpha=0.4)
        plt.suptitle("Feature Importances", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-header">Dataset Overview</div>', unsafe_allow_html=True)

    cs1, cs2, cs3, cs4 = st.columns(4)
    cs1.markdown(f"""<div class="metric-card"><div class="mlabel">Total Samples</div>
        <div class="mvalue">{len(df)}</div></div>""", unsafe_allow_html=True)
    cs2.markdown(f"""<div class="metric-card"><div class="mlabel">Features</div>
        <div class="mvalue">{len(FEATURE_COLS)}</div></div>""", unsafe_allow_html=True)
    cs3.markdown(f"""<div class="metric-card"><div class="mlabel">Avg Strength</div>
        <div class="mvalue">{df[TARGET_COL].mean():.1f}</div>
        <div class="msub">MPa</div></div>""", unsafe_allow_html=True)
    cs4.markdown(f"""<div class="metric-card"><div class="mlabel">Max Strength</div>
        <div class="mvalue">{df[TARGET_COL].max():.1f}</div>
        <div class="msub">MPa</div></div>""", unsafe_allow_html=True)

    st.markdown("**Full Dataset**")
    st.dataframe(df[FEATURE_COLS + [TARGET_COL]], use_container_width=True, height=400)

    st.markdown("**Statistical Summary**")
    st.dataframe(df[FEATURE_COLS + [TARGET_COL]].describe().round(3), use_container_width=True)

    csv_out = df[FEATURE_COLS + [TARGET_COL]].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Dataset as CSV",
        data=csv_out,
        file_name="geopolymer_civil_dataset.csv",
        mime="text/csv",
    )
