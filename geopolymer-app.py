"""
Geopolymer Concrete 28-Day Compressive Strength Predictor
==========================================================
• Supports two datasets: excel1.xlsx (Fly Ash) & excel2.xlsx (Blended / GGBS mixes)
• 5 ML models: Linear Regression, Ridge, SVR, Random Forest, Gradient Boosting
• Forward prediction (mix → strength) + Inverse prediction (strength → optimal mix)
• EDA, residual analysis, feature importances

Run with:
    streamlit run geopolymer_app.py

Place excel1.xlsx and excel2.xlsx in the same directory as this script.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scipy.stats as stats_module
from scipy.stats import norm
from scipy.optimize import minimize
from itertools import product as iproduct

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

/* Input labels */
.stNumberInput label, .stSelectbox label { color: #4a5068 !important; font-weight: 500; font-size: 13px; }

/* Number inputs */
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

/* Selectbox */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1.5px solid #d1d5e8 !important;
    border-radius: 8px !important;
    color: #1a1d27 !important;
}

/* Cards */
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

/* Result banner */
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

/* Section header */
.sec-header {
    font-size: 17px;
    font-weight: 700;
    color: #1a1d27;
    border-bottom: 2px solid #e2e5ee;
    padding-bottom: 8px;
    margin: 20px 0 14px 0;
}

/* Info / warn box */
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

/* Tabs */
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

/* Button */
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

/* Headings */
h1, h2, h3 { color: #1a1d27 !important; }

/* Input group label */
.group-label {
    font-size: 12px;
    font-weight: 600;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 10px;
}

/* Inv card */
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


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    df = df.replace(["--", "-", "—", "", "nan", "NaN", "None"], np.nan)

    def fix_range(val):
        if isinstance(val, str) and "-" in val:
            parts = val.split("-")
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except Exception:
                return np.nan
        return val

    df["Curing_temp"] = df["Curing_temp"].apply(fix_range)

    num_cols = ["Molarity", "SS_SH_KOH_ratio", "A_B_ratio", "Curing_temp", "Compression_28d"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Compression_28d"], inplace=True)
    if "Curing_temp" in df.columns:
        df = df[df["Curing_temp"].isna() | (df["Curing_temp"] <= 250)]
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data
def load_datasets():
    datasets = {}

    try:
        df1 = pd.read_excel("excel1.xlsx")
    except FileNotFoundError:
        np.random.seed(1)
        n = 80
        df1 = pd.DataFrame({
            "Source_material": np.random.choice(["F. A"], n),
            "Alkali_solution": np.random.choice(["SS/SH", "SS/ KOH", "SS/SH/25%KOH"], n),
            "Molarity":        np.random.choice([8, 10, 12, 14, 16], n),
            "SS_SH_KOH_ratio": np.random.uniform(1.0, 3.0, n).round(2),
            "A_B_ratio":       np.random.uniform(0.28, 0.55, n).round(2),
            "Curing_temp":     np.random.choice([20, 25, 60, 70, 80, 90, 100], n),
            "Compression_28d": np.random.uniform(20, 90, n).round(2),
        })

    df1 = _clean_df(df1)
    df1["_dataset"] = "Dataset 1 (Fly Ash)"
    datasets["Dataset 1 (Fly Ash)"] = df1

    try:
        df2 = pd.read_excel("excel2.xlsx")
    except FileNotFoundError:
        np.random.seed(2)
        n = 36
        df2 = pd.DataFrame({
            "Source_material": np.random.choice(["F.A + GGBS", "GBFS+RHA", "RHA+UFS"], n),
            "Alkali_solution": np.random.choice(["SS/SH", "NaOH+ Na2So4"], n),
            "Molarity":        np.random.choice([6, 8, 10, 12, 14], n),
            "SS_SH_KOH_ratio": np.random.uniform(1.5, 3.5, n).round(1),
            "A_B_ratio":       np.random.uniform(0.30, 0.70, n).round(2),
            "Curing_temp":     np.random.choice([23, 25, 60, 70, 95], n),
            "Compression_28d": np.random.uniform(12, 82, n).round(2),
        })

    df2 = _clean_df(df2)
    df2["_dataset"] = "Dataset 2 (Blended)"
    datasets["Dataset 2 (Blended)"] = df2

    df_combined = pd.concat([df1, df2], ignore_index=True)
    datasets["Combined"] = df_combined

    return datasets


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
    df = pd.read_json(df_json, orient="split")

    TARGET = "Compression_28d"
    drop_cols = [TARGET, "_dataset"] if "_dataset" in df.columns else [TARGET]
    X = df.drop(columns=drop_cols).copy()
    y = df[TARGET].copy()

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols   = X.select_dtypes(exclude="object").columns.tolist()

    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[numerical_cols]   = num_imputer.fit_transform(X[numerical_cols])
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

    label_encoders = {}
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models_def = build_models()
    results    = {}
    trained    = {}

    for name, m in models_def.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
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
        label_encoders=label_encoders,
        num_imputer=num_imputer,
        cat_imputer=cat_imputer,
        feature_names=feature_names,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
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
    row = pd.DataFrame([row_dict])
    for col in p["categorical_cols"]:
        row[col] = row[col].astype(str)
        le  = p["label_encoders"][col]
        val = row[col].values[0]
        val = val if val in le.classes_ else le.classes_[0]
        row[col] = le.transform([val])
    for col in p["numerical_cols"]:
        row[col] = pd.to_numeric(row[col], errors="coerce")
    row = row[p["feature_names"]]
    return p["scaler"].transform(row)


def get_grade(s):
    if   s < 20: return "M15 – Low",             "#ef4444"
    elif s < 25: return "M20 – Standard",         "#f97316"
    elif s < 30: return "M25 – Standard",         "#f59e0b"
    elif s < 40: return "M30-M35 – Moderate",     "#10b981"
    elif s < 55: return "M40-M50 – High",         "#059669"
    else:        return "M55+ – Very High",       "#0d9488"


def inverse_predict(target, model, p, df):
    cat_vals  = {c: df[c].dropna().unique().tolist() for c in p["categorical_cols"]}
    num_rngs  = {c: (df[c].dropna().min(), df[c].dropna().max()) for c in p["numerical_cols"]}
    cat_combos = list(iproduct(*[cat_vals[c] for c in p["categorical_cols"]]))

    best_combo, best_diff = None, np.inf

    for combo in cat_combos:
        cat_dict = dict(zip(p["categorical_cols"], combo))

        def obj(num_vals):
            num_dict = dict(zip(p["numerical_cols"], num_vals))
            row_dict = {**cat_dict, **num_dict}
            try:
                x = preprocess_input(row_dict, p)
                return (model.predict(x)[0] - target) ** 2
            except Exception:
                return 1e9

        x0     = [np.mean(num_rngs[c]) for c in p["numerical_cols"]]
        bounds = [num_rngs[c] for c in p["numerical_cols"]]
        res    = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 300, "ftol": 1e-8})
        if res.fun < best_diff:
            best_diff  = res.fun
            best_combo = {**cat_dict, **dict(zip(p["numerical_cols"], res.x))}

    if best_combo is None:
        return None, None
    x_best   = preprocess_input(best_combo, p)
    achieved = model.predict(x_best)[0]
    return best_combo, achieved


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ══════════════════════════════════════════════════════════════════════════════

all_datasets = load_datasets()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ Geopolymer Predictor")
    st.markdown("---")

    dataset_choice = st.selectbox(
        "Dataset",
        ["Dataset 1 (Fly Ash)", "Dataset 2 (Blended)", "Combined"],
        index=2,
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


# ── Load & train ──────────────────────────────────────────────────────────────
df = all_datasets[dataset_choice].copy()
if "_dataset" in df.columns:
    df_no_tag = df.drop(columns=["_dataset"])
else:
    df_no_tag = df.copy()

p = train_pipeline(df_no_tag.to_json(orient="split"))
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
🏗️ Geopolymer Concrete — 28-Day Strength Predictor
</h1>
<p style="color:#8892b0;font-size:13px;margin-top:4px;">
ML-powered prediction · Mix design optimizer · Model benchmarking
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

        col_a, col_b = st.columns([1, 1], gap="large")

        with col_a:
            st.markdown('<div class="group-label">Categorical</div>', unsafe_allow_html=True)
            source_opts = sorted(df_no_tag["Source_material"].dropna().unique().tolist())
            alkali_opts = sorted(df_no_tag["Alkali_solution"].dropna().unique().tolist())
            source_material = st.selectbox("Source Material", source_opts)
            alkali_solution = st.selectbox("Alkali Solution", alkali_opts)

        with col_b:
            st.markdown('<div class="group-label">Numerical</div>', unsafe_allow_html=True)

            mol_min = float(df_no_tag["Molarity"].dropna().min())
            mol_max = float(df_no_tag["Molarity"].dropna().max())
            mol_med = float(df_no_tag["Molarity"].dropna().median())
            molarity = st.number_input(
                "Molarity (M)", min_value=mol_min, max_value=mol_max,
                value=mol_med, step=0.001, format="%.3f",
                help=f"Range in dataset: {mol_min:.3f} – {mol_max:.3f}"
            )

            ss_min = float(df_no_tag["SS_SH_KOH_ratio"].dropna().min())
            ss_max = float(df_no_tag["SS_SH_KOH_ratio"].dropna().max())
            ss_med = float(df_no_tag["SS_SH_KOH_ratio"].dropna().median())
            ss_sh = st.number_input(
                "SS / SH or KOH Ratio", min_value=ss_min, max_value=ss_max,
                value=ss_med, step=0.001, format="%.3f",
                help=f"Range: {ss_min:.3f} – {ss_max:.3f}"
            )

            ab_min = float(df_no_tag["A_B_ratio"].dropna().min())
            ab_max = float(df_no_tag["A_B_ratio"].dropna().max())
            ab_med = float(df_no_tag["A_B_ratio"].dropna().median())
            ab_ratio = st.number_input(
                "Alkali / Binder Ratio", min_value=ab_min, max_value=ab_max,
                value=ab_med, step=0.001, format="%.3f",
                help=f"Range: {ab_min:.3f} – {ab_max:.3f}"
            )

            ct_min = float(df_no_tag["Curing_temp"].dropna().min())
            ct_max = float(df_no_tag["Curing_temp"].dropna().max())
            ct_med = float(df_no_tag["Curing_temp"].dropna().median())
            curing_temp = st.number_input(
                "Curing Temperature (°C)", min_value=ct_min, max_value=ct_max,
                value=ct_med, step=0.1, format="%.1f",
                help=f"Range: {ct_min:.1f} – {ct_max:.1f} °C"
            )

        st.markdown("---")
        if st.button("🚀 Predict 28-Day Compressive Strength"):
            row_dict = {
                "Source_material": source_material,
                "Alkali_solution":  alkali_solution,
                "Molarity":         molarity,
                "SS_SH_KOH_ratio":  ss_sh,
                "A_B_ratio":        ab_ratio,
                "Curing_temp":      curing_temp,
            }
            x_in     = preprocess_input(row_dict, p)
            pred_val = selected_model.predict(x_in)[0]
            grade, gc = get_grade(pred_val)

            st.markdown(f"""
            <div class="result-banner">
                <div class="rval">{pred_val:.2f} MPa</div>
                <div class="rlabel">Predicted 28-Day Compressive Strength</div>
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

            # Strength grade scale bar
            st.markdown('<div class="sec-header">Strength Grade Context</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 2.2))
            categories = ["Low\n<20", "Standard\n20-30", "Moderate\n30-40", "High\n40-55", "Very High\n≥55"]
            thresholds = [20, 30, 40, 55, 90]
            bar_colors = ["#fca5a5", "#fed7aa", "#fde68a", "#6ee7b7", "#5eead4"]
            prev = 0
            for cat, thresh, clr in zip(categories, thresholds, bar_colors):
                ax.barh(0, thresh - prev, left=prev, color=clr, alpha=0.85, height=0.5)
                prev = thresh
            ax.axvline(pred_val, color="#4f6ef7", linewidth=3, linestyle="--",
                       label=f"Predicted: {pred_val:.1f} MPa")
            ax.set_xlim(0, 90); ax.set_yticks([]); ax.set_xlabel("Compressive Strength (MPa)", fontsize=10)
            ax.legend(fontsize=9); ax.grid(False)
            prev2 = 0
            for cat, thresh, clr in zip(categories, thresholds, bar_colors):
                mid = prev2 + (thresh - prev2) / 2
                ax.text(mid, 0, cat, ha="center", va="center", fontsize=7.5, color="#1a1d27", fontweight="600")
                prev2 = thresh
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # All models comparison
            st.markdown('<div class="sec-header">All Models — Prediction Comparison</div>', unsafe_allow_html=True)
            preds_all = {nm: m.predict(x_in)[0] for nm, m in p["trained"].items()}
            names_all = list(preds_all.keys())
            vals_all  = list(preds_all.values())
            colors_all = [PALETTE[0] if n == selected_model_name else "#d1d5e8" for n in names_all]
            fig, ax = plt.subplots(figsize=(9, 3.5))
            bars = ax.bar(names_all, vals_all, color=colors_all, edgecolor="#ffffff",
                          linewidth=1.2, width=0.55)
            ax.axhline(preds_all[selected_model_name], color="#f59e0b", linestyle="--",
                       linewidth=1.5, label="Selected model")
            for bar, v in zip(bars, vals_all):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_ylabel("Predicted Strength (MPa)", fontsize=10)
            ax.set_title("28-day Prediction Across All Models", fontsize=11, fontweight="700")
            ax.set_xticklabels(names_all, rotation=15, ha="right")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── TARGET → MIX DESIGN ──────────────────────────────────────────────────
    else:
        st.markdown('<div class="sec-header">Target Strength → Optimal Mix Design</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
        Enter your <b>desired 28-day compressive strength</b>. The optimiser searches all
        categorical combinations and fine-tunes numerical parameters using L-BFGS-B to find
        the mix that best achieves it.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            target_strength = st.number_input(
                "Target 28-day Compressive Strength (MPa)",
                min_value=5.0, max_value=150.0, value=40.0, step=0.5, format="%.1f"
            )
        with c2:
            st.markdown(""); st.markdown("")
            find_btn = st.button("⚡ Find Optimal Mix")

        st.markdown(f"""<div class="warn-box">
        ⚠️ Dataset strength range: <b>{df_no_tag['Compression_28d'].min():.1f} –
        {df_no_tag['Compression_28d'].max():.1f} MPa</b>.
        Targets outside this range extrapolate beyond training data.
        </div>""", unsafe_allow_html=True)

        if find_btn:
            with st.spinner("Optimising mix design… (~10 s)"):
                best_combo, achieved = inverse_predict(
                    target_strength, selected_model, p, df_no_tag)

            if best_combo is None:
                st.error("Optimisation failed. Try a different target or model.")
            else:
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
                icons = {"Source_material": "🧱", "Alkali_solution": "🧪", "Molarity": "⚗️",
                         "SS_SH_KOH_ratio": "🔬", "A_B_ratio": "📐", "Curing_temp": "🌡️"}
                rec1, rec2 = st.columns(2)
                for i, (key, val) in enumerate(best_combo.items()):
                    cont = rec1 if i % 2 == 0 else rec2
                    fmt  = f"{val:.3f}" if isinstance(val, float) else str(val)
                    cont.markdown(f"""<div class="inv-card">
                        <div class="inv-card-title">{icons.get(key,'•')} {key.replace('_',' ')}</div>
                        <div class="inv-card-value">{fmt}</div></div>""",
                        unsafe_allow_html=True)

                # Recommended vs dataset average
                st.markdown('<div class="sec-header">Recommended vs Dataset Average</div>', unsafe_allow_html=True)
                num_keys = [k for k in best_combo if k in p["numerical_cols"]]
                fig, axes = plt.subplots(1, len(num_keys), figsize=(13, 4))
                if len(num_keys) == 1:
                    axes = [axes]
                for ax_i, key in enumerate(num_keys):
                    rv  = best_combo[key]
                    avg = df_no_tag[key].mean()
                    bars_ = axes[ax_i].bar(["Recommended", "Dataset Avg"], [rv, avg],
                                           color=["#4f6ef7", "#d1d5e8"],
                                           edgecolor="#ffffff", width=0.5)
                    for bar, v in zip(bars_, [rv, avg]):
                        axes[ax_i].text(bar.get_x() + bar.get_width() / 2,
                                        bar.get_height() + 0.01 * abs(v) + 0.1,
                                        f"{v:.3f}", ha="center", va="bottom",
                                        fontsize=10, fontweight="bold")
                    axes[ax_i].set_title(key.replace("_", " "), fontsize=10, fontweight="bold")
                    axes[ax_i].set_ylabel("Value")
                    axes[ax_i].grid(axis="y", alpha=0.4)
                plt.suptitle(f"Recommended Mix vs Dataset Average  (Target: {target_strength} MPa)",
                             fontsize=11, fontweight="bold")
                plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    num_eda = [c for c in ["Molarity", "SS_SH_KOH_ratio", "A_B_ratio", "Curing_temp", "Compression_28d"]
               if c in df_no_tag.columns]

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("**Correlation Heatmap**")
        corr = df_no_tag[num_eda].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                    mask=mask, linewidths=0.5, ax=ax,
                    annot_kws={"size": 10, "weight": "bold"},
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation – Numeric Features", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r1c2:
        st.markdown("**Strength by Source Material**")
        order = (df_no_tag.groupby("Source_material")["Compression_28d"]
                 .median().sort_values(ascending=False).index)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.boxplot(data=df_no_tag, x="Source_material", y="Compression_28d",
                    order=order, palette="Set2", ax=ax, linewidth=1.2)
        ax.set_xlabel("Source Material", fontsize=9)
        ax.set_ylabel("Strength (MPa)", fontsize=9)
        ax.set_title("28-day Strength by Source Material", fontsize=11, fontweight="bold")
        plt.xticks(rotation=22, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Scatter plots
    st.markdown("**Feature vs Compressive Strength**")
    scatter_feats = [f for f in ["Molarity", "SS_SH_KOH_ratio", "A_B_ratio", "Curing_temp"]
                     if f in df_no_tag.columns]
    sc_cols = st.columns(len(scatter_feats))
    for i, feat in enumerate(scatter_feats):
        fig, ax = plt.subplots(figsize=(3.5, 3))
        sub = df_no_tag[[feat, "Compression_28d", "Source_material"]].dropna()
        for j, (src, grp) in enumerate(sub.groupby("Source_material")):
            ax.scatter(grp[feat], grp["Compression_28d"],
                       label=src[:14], alpha=0.65, s=28, edgecolors="none",
                       color=PALETTE[j % len(PALETTE)])
        if len(sub) > 1:
            c_ = np.polyfit(sub[feat], sub["Compression_28d"], 1)
            xr = np.linspace(sub[feat].min(), sub[feat].max(), 100)
            ax.plot(xr, np.polyval(c_, xr), color="#1a1d27", linestyle="--", linewidth=1.4)
        ax.set_xlabel(feat.replace("_", " "), fontsize=8)
        ax.set_ylabel("Strength (MPa)", fontsize=8)
        ax.set_title(f"{feat.replace('_',' ')}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=5)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        sc_cols[i].pyplot(fig); plt.close()

    # Heatmap: Molarity × Curing Temp
    st.markdown("**Avg Strength: Molarity × Curing Temp**")
    df_tmp = df_no_tag.copy()
    df_tmp["Mol_bin"]  = pd.cut(df_tmp["Molarity"], bins=4)
    df_tmp["Cure_bin"] = pd.cut(df_tmp["Curing_temp"], bins=4)
    piv = df_tmp.pivot_table(values="Compression_28d", index="Cure_bin",
                              columns="Mol_bin", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.4,
                ax=ax, cbar_kws={"label": "Avg Strength (MPa)"})
    ax.set_title("Avg Strength: Curing Temp × Molarity", fontsize=11, fontweight="bold")
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

    # Actual vs Predicted
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
        for ax, (nm_, clr_, cmap_) in zip(axes, [
            ("Random Forest",     PALETTE[0], "Blues"),
            ("Gradient Boosting", PALETTE[1], "Greens"),
        ]):
            m_ = p["trained"][nm_]
            if hasattr(m_, "feature_importances_"):
                imp = m_.feature_importances_
                idx = np.argsort(imp)
                ax.barh([p["feature_names"][i] for i in idx], imp[idx],
                        color=sns.color_palette(cmap_, len(p["feature_names"])),
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
        <div class="mvalue">{len(df_no_tag)}</div></div>""", unsafe_allow_html=True)
    cs2.markdown(f"""<div class="metric-card"><div class="mlabel">Features</div>
        <div class="mvalue">6</div></div>""", unsafe_allow_html=True)
    cs3.markdown(f"""<div class="metric-card"><div class="mlabel">Avg Strength</div>
        <div class="mvalue">{df_no_tag['Compression_28d'].mean():.1f}</div>
        <div class="msub">MPa</div></div>""", unsafe_allow_html=True)
    cs4.markdown(f"""<div class="metric-card"><div class="mlabel">Max Strength</div>
        <div class="mvalue">{df_no_tag['Compression_28d'].max():.1f}</div>
        <div class="msub">MPa</div></div>""", unsafe_allow_html=True)

    st.markdown("**Full Dataset**")
    st.dataframe(df_no_tag, use_container_width=True, height=400)

    st.markdown("**Statistical Summary**")
    st.dataframe(df_no_tag.describe().round(3), use_container_width=True)

    csv_out = df_no_tag.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Dataset as CSV",
        data=csv_out,
        file_name=f"geopolymer_{dataset_choice.replace(' ','_').lower()}.csv",
        mime="text/csv",
    )