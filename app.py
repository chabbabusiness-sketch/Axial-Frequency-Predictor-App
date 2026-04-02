import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

GB_MODEL_CANDIDATES = [
    BASE_DIR / "Saved_Models" / "gradient_boosting_model.pkl",
    BASE_DIR / "final_gradientboosting_model.pkl",
]

GAM_MODEL_CANDIDATES = [
    BASE_DIR / "Saved_Models" / "gam_model.pkl",
    BASE_DIR / "best_gam_model.pkl",
]

PIECEWISE_ENET_CANDIDATES = [
    BASE_DIR / "Piecewise_ElasticNet_Equation" / "piecewise_elasticnet_model.pkl",
    BASE_DIR / "piecewise_elasticnet_model.pkl",
]

CONTOUR_DATA_CANDIDATES = [
    BASE_DIR / "Matlab.xlsx",
]

IMAGE_CANDIDATES = [
    BASE_DIR / "Steel and Aluminium.png",
    BASE_DIR / "Physics_Feature_Outputs" / "Steel and Aluminium.png",
]

VIDEO_CANDIDATES = [
    BASE_DIR / "Axial Deformation.mp4",
    BASE_DIR / "Physics_Feature_Outputs" / "Axial Deformation.mp4",
]

# =========================================================
# EXACT BOUNDARIES
# =========================================================
EXACT_BOUNDS_GB = {
    "Phi_Fixed": (438.9572656045681, 12505.054249145147),
    "Phi_Free": (438.9572656045681, 12505.054249145147),
    "eta_Fixed": (0.029, 0.45),
    "eta_Free": (0.029, 0.45),
}

EXACT_BOUNDS_GAM = {
    "Phi_Fixed": (438.9572656045681, 12505.054249145147),
    "Phi_Other": (438.9572656045681, 12505.054249145147),
    "sqrt_chi": (0.035102387951221836, 28.488090365521476),
    "nu_Fixed": (0.029, 0.45),
    "nu_Other": (0.029, 0.45),
}

EXACT_BOUNDS_PW = {
    "Phi_Fixed": (438.9572656045681, 12505.054249145147),
    "Phi_Other": (438.9572656045681, 12505.054249145147),
    "sqrt_chi": (0.035102387951221836, 28.488090365521476),
    "log_Phi_Fixed": (6.084402063472543, 9.433888181498643),
    "log_Phi_Other": (6.084402063472543, 9.433888181498643),
    "log_chi": (-6.6989722360521995, 6.6989722360521995),
    "Phi_Product": (192683.48102703932, 156376381.77406308),
    "Phi_Fixed_sq": (192683.48102703932, 156376381.77406308),
    "Phi_Other_sq": (192683.48102703932, 156376381.77406308),
    "nu_Fixed": (0.029, 0.45),
    "nu_Other": (0.029, 0.45),
    "nu_Fixed_sq": (0.0008410000000000001, 0.2025),
    "nu_Other_sq": (0.0008410000000000001, 0.2025),
    "Phi_Fixed_x_nu_Fixed": (197.53076952205564, 1810.0817878933842),
    "Phi_Other_x_nu_Other": (197.53076952205564, 1810.0817878933842),
    "sqrt_chi_x_nu_Other": (0.0010179692505854333, 12.819640664484664),
    "log_chi_x_nu_Fixed": (-3.01453750622349, 1.8120750472592766),
    "log_chi_x_nu_Other": (-1.8120750472592766, 3.01453750622349),
}

# =========================================================
# PAGE
# =========================================================
st.set_page_config(page_title="Frequency Predictor", layout="wide")

# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
    .pred-card {
        background: linear-gradient(135deg, #0f5132, #146c43);
        border-radius: 14px;
        padding: 22px 20px;
        min-height: 135px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        margin-bottom: 10px;
    }
    .pred-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #d7ffe4;
        margin-bottom: 16px;
    }
    .pred-value {
        font-size: 2.15rem;
        font-weight: 800;
        color: white;
        line-height: 1.1;
    }

    .small-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 95px;
        margin-bottom: 8px;
    }
    .small-title {
        font-size: 0.90rem;
        font-weight: 600;
        color: #cfd8dc;
        margin-bottom: 8px;
    }
    .small-value {
        font-size: 1.00rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.15;
    }

    .info-card {
        background: rgba(33, 150, 243, 0.16);
        border: 1px solid rgba(33, 150, 243, 0.35);
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 88px;
        margin-bottom: 8px;
    }
    .info-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #9fd0ff;
        margin-bottom: 8px;
    }
    .info-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.15;
    }

    .caption-text {
        font-size: 0.95rem;
        color: #c6d0d8;
        margin-top: 2px;
        margin-bottom: 18px;
    }

    .limit-box {
        background: rgba(255, 193, 7, 0.12);
        border: 1px solid rgba(255, 193, 7, 0.35);
        border-radius: 12px;
        padding: 12px 14px;
        margin-top: 8px;
        margin-bottom: 12px;
        color: #ffe082;
        font-size: 0.95rem;
    }

    .formula-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    .section-note {
        font-size: 0.95rem;
        color: #c9d1d9;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DISPLAY NAME MAPPING
# =========================================================
DISPLAY_NAME_MAP = {
    "E_Fixed": "E (Fixed) [N/m²]",
    "E_Other": "E (Free) [N/m²]",
    "E_Free": "E (Free) [N/m²]",
    "rho_Fixed": "ρ (Fixed) [kg/m³]",
    "rho_Other": "ρ (Free) [kg/m³]",
    "rho_Free": "ρ (Free) [kg/m³]",
    "nu_Fixed": "ν (Fixed) [-]",
    "nu_Other": "ν (Free) [-]",
    "eta_Fixed": "ν (Fixed) [-]",
    "eta_Free": "ν (Free) [-]",
    "Phi_Fixed": "ϕ (Fixed) [m/s]",
    "Phi_Other": "ϕ (Free) [m/s]",
    "Phi_Free": "ϕ (Free) [m/s]",
    "sqrt_chi": "√χ [-]",
    "chi": "χ [-]",
    "log_chi": "log(χ) [-]",
    "log_Phi_Fixed": "log(ϕ Fixed) [-]",
    "log_Phi_Other": "log(ϕ Free) [-]",
    "Phi_Product": "ϕ(Fixed) × ϕ(Free) [m²/s²]",
    "Phi_Fixed_sq": "ϕ(Fixed)^2 [m²/s²]",
    "Phi_Other_sq": "ϕ(Free)^2 [m²/s²]",
    "nu_Fixed_sq": "ν(Fixed)^2 [-]",
    "nu_Other_sq": "ν(Free)^2 [-]",
    "Phi_Fixed_x_nu_Fixed": "ϕ(Fixed) × ν(Fixed) [m/s]",
    "Phi_Other_x_nu_Other": "ϕ(Free) × ν(Free) [m/s]",
    "sqrt_chi_x_nu_Other": "√χ × ν(Free) [-]",
    "log_chi_x_nu_Fixed": "log(χ) × ν(Fixed) [-]",
    "log_chi_x_nu_Other": "log(χ) × ν(Free) [-]",
}

def pretty_name(name):
    return DISPLAY_NAME_MAP.get(name, name)

def rename_display_columns(df):
    return df.rename(columns={col: pretty_name(col) for col in df.columns})

# =========================================================
# HELPERS
# =========================================================
def first_existing_path(candidates, label):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"{label} not found. Checked:\n" + "\n".join(str(p) for p in candidates)
    )

def optional_existing_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None

def safe_log(x):
    return math.log(max(x, 1e-12))

def get_model_from_pack(pack, preferred_keys=None):
    if preferred_keys is None:
        preferred_keys = []

    if hasattr(pack, "predict"):
        return pack

    if isinstance(pack, dict):
        for key in preferred_keys:
            if key in pack and hasattr(pack[key], "predict"):
                return pack[key]
        for _, value in pack.items():
            if hasattr(value, "predict"):
                return value

    raise TypeError(f"Could not find a model with predict() inside object of type {type(pack)}")

def get_feature_names(pack, model):
    if isinstance(pack, dict) and "feature_names" in pack:
        return list(pack["feature_names"])
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None

def render_pred_card(title, value):
    st.markdown(
        f"""
        <div class="pred-card">
            <div class="pred-title">{title}</div>
            <div class="pred-value">{value:.6f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_small_card(title, value):
    st.markdown(
        f"""
        <div class="small-card">
            <div class="small-title">{title}</div>
            <div class="small-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_info_card(title, value):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-title">{title}</div>
            <div class="info-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def build_exact_range_table(current_df, exact_bounds):
    current = current_df.iloc[0].to_dict()
    rows = []

    for feat, val in current.items():
        mn, mx = exact_bounds[feat]
        status = "Inside" if mn <= float(val) <= mx else "Outside"
        rows.append({
            "Feature": pretty_name(feat),
            "Current": float(val),
            "Lower boundary": float(mn),
            "Upper boundary": float(mx),
            "Status": status,
        })

    return pd.DataFrame(rows)

# =========================================================
# LOAD ARTIFACTS
# =========================================================
@st.cache_resource
def load_artifacts():
    gb_path = first_existing_path(GB_MODEL_CANDIDATES, "Gradient Boosting model")
    gam_path = first_existing_path(GAM_MODEL_CANDIDATES, "GAM model")
    pw_path = first_existing_path(PIECEWISE_ENET_CANDIDATES, "2-piece ElasticNet model")

    gb_pack = joblib.load(gb_path)
    gam_pack = joblib.load(gam_path)
    pw_pack = joblib.load(pw_path)

    gb_model = get_model_from_pack(gb_pack, preferred_keys=["model", "gb_model"])
    gam_model = get_model_from_pack(gam_pack, preferred_keys=["gam_model", "model"])

    gb_feature_names = get_feature_names(gb_pack, gb_model)
    gam_feature_names = get_feature_names(gam_pack, gam_model)

    return {
        "gb_pack": gb_pack,
        "gam_pack": gam_pack,
        "pw_pack": pw_pack,
        "gb_model": gb_model,
        "gam_model": gam_model,
        "gb_feature_names": gb_feature_names,
        "gam_feature_names": gam_feature_names,
    }

# =========================================================
# BASE FEATURE BUILDING
# =========================================================
def build_base_values(E_Fixed, rho_Fixed, nu_Fixed, E_Other, rho_Other, nu_Other):
    if E_Fixed <= 0 or rho_Fixed <= 0 or E_Other <= 0 or rho_Other <= 0:
        raise ValueError("E and ρ values must be positive.")

    Phi_Fixed = math.sqrt(E_Fixed / rho_Fixed)
    Phi_Other = math.sqrt(E_Other / rho_Other)

    R_E = E_Fixed / E_Other
    R_rho = rho_Fixed / rho_Other
    chi = R_E / R_rho
    sqrt_chi = math.sqrt(chi)
    log_chi = safe_log(chi)

    values = {
        "E_Fixed": E_Fixed,
        "rho_Fixed": rho_Fixed,
        "nu_Fixed": nu_Fixed,
        "E_Other": E_Other,
        "rho_Other": rho_Other,
        "nu_Other": nu_Other,
        "Phi_Fixed": Phi_Fixed,
        "Phi_Other": Phi_Other,
        "Phi_Free": Phi_Other,
        "eta_Fixed": nu_Fixed,
        "eta_Free": nu_Other,
        "R_E": R_E,
        "R_rho": R_rho,
        "chi": chi,
        "sqrt_chi": sqrt_chi,
        "log_Phi_Fixed": safe_log(Phi_Fixed),
        "log_Phi_Other": safe_log(Phi_Other),
        "log_chi": log_chi,
        "Phi_Product": Phi_Fixed * Phi_Other,
        "nu_Fixed_sq": nu_Fixed ** 2,
        "nu_Other_sq": nu_Other ** 2,
        "Phi_Fixed_x_nu_Fixed": Phi_Fixed * nu_Fixed,
        "Phi_Other_x_nu_Other": Phi_Other * nu_Other,
        "sqrt_chi_x_nu_Other": sqrt_chi * nu_Other,
        "log_chi_x_nu_Fixed": log_chi * nu_Fixed,
        "log_chi_x_nu_Other": log_chi * nu_Other,
    }
    return values

# =========================================================
# MODEL INPUT BUILDERS
# =========================================================
PIECEWISE_FEATURE_NAMES = [
    "Phi_Fixed",
    "Phi_Other",
    "sqrt_chi",
    "log_Phi_Fixed",
    "log_Phi_Other",
    "log_chi",
    "Phi_Product",
    "Phi_Fixed_sq",
    "Phi_Other_sq",
    "nu_Fixed",
    "nu_Other",
    "nu_Fixed_sq",
    "nu_Other_sq",
    "Phi_Fixed_x_nu_Fixed",
    "Phi_Other_x_nu_Other",
    "sqrt_chi_x_nu_Other",
    "log_chi_x_nu_Fixed",
    "log_chi_x_nu_Other",
]

def build_piecewise_input_df(v):
    row = {
        "Phi_Fixed": v["Phi_Fixed"],
        "Phi_Other": v["Phi_Other"],
        "sqrt_chi": v["sqrt_chi"],
        "log_Phi_Fixed": v["log_Phi_Fixed"],
        "log_Phi_Other": v["log_Phi_Other"],
        "log_chi": v["log_chi"],
        "Phi_Product": v["Phi_Product"],
        "Phi_Fixed_sq": v["Phi_Fixed"] ** 2,
        "Phi_Other_sq": v["Phi_Other"] ** 2,
        "nu_Fixed": v["nu_Fixed"],
        "nu_Other": v["nu_Other"],
        "nu_Fixed_sq": v["nu_Fixed_sq"],
        "nu_Other_sq": v["nu_Other_sq"],
        "Phi_Fixed_x_nu_Fixed": v["Phi_Fixed_x_nu_Fixed"],
        "Phi_Other_x_nu_Other": v["Phi_Other_x_nu_Other"],
        "sqrt_chi_x_nu_Other": v["sqrt_chi_x_nu_Other"],
        "log_chi_x_nu_Fixed": v["log_chi_x_nu_Fixed"],
        "log_chi_x_nu_Other": v["log_chi_x_nu_Other"],
    }
    return pd.DataFrame([row])[PIECEWISE_FEATURE_NAMES]

def build_gb_input_df(gb_feature_names, v):
    value_map = {
        "Phi_Fixed": v["Phi_Fixed"],
        "Phi_Free": v["Phi_Free"],
        "eta_Fixed": v["eta_Fixed"],
        "eta_Free": v["eta_Free"],
    }

    if gb_feature_names is None:
        gb_feature_names = ["Phi_Fixed", "Phi_Free", "eta_Fixed", "eta_Free"]

    return pd.DataFrame([{name: value_map[name] for name in gb_feature_names}])

def build_gam_input_df(gam_feature_names, v):
    value_map = {
        "Phi_Fixed": v["Phi_Fixed"],
        "Phi_Other": v["Phi_Other"],
        "sqrt_chi": v["sqrt_chi"],
        "nu_Fixed": v["nu_Fixed"],
        "nu_Other": v["nu_Other"],
    }

    if gam_feature_names is None:
        gam_feature_names = ["Phi_Fixed", "Phi_Other", "sqrt_chi", "nu_Fixed", "nu_Other"]

    return pd.DataFrame([[value_map[name] for name in gam_feature_names]], columns=gam_feature_names)

# =========================================================
# MODEL PREDICTIONS
# =========================================================
def predict_two_piece_elasticnet(pw_pack, pw_df, v):
    split_var = pw_pack["split_var"]
    threshold = pw_pack["threshold"]
    split_value = v[split_var]

    if split_value <= threshold:
        scaler = pw_pack["left_scaler"]
        model = pw_pack["left_model"]
        region = "Region 1"
    else:
        scaler = pw_pack["right_scaler"]
        model = pw_pack["right_model"]
        region = "Region 2"

    X_scaled = scaler.transform(pw_df)
    pred = float(np.exp(model.predict(X_scaled))[0])

    return pred, region, split_var, threshold, split_value

def predict_gradient_boosting(gb_model, gb_df):
    return float(gb_model.predict(gb_df)[0])

def predict_gam(gam_model, gam_df):
    gam_array = gam_df.to_numpy(dtype=float)
    raw_gam = float(gam_model.predict(gam_array)[0])
    final_gam = float(np.exp(raw_gam))
    return raw_gam, final_gam

# =========================================================
# CONTOUR PLOT
# =========================================================
@st.cache_data
def load_contour_grid():
    contour_path = first_existing_path(CONTOUR_DATA_CANDIDATES, "Contour Excel file")

    T = pd.read_excel(contour_path)
    T = T.iloc[:, :3].copy()

    original_names = list(T.columns)
    T.columns = ["x", "y", "z"]
    T = T.dropna()

    pivot = T.pivot_table(index="y", columns="x", values="z", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    pivot = pivot.interpolate(axis=0).interpolate(axis=1)
    pivot = pivot.ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1)

    x_vals = pivot.columns.to_numpy(dtype=float)
    y_vals = pivot.index.to_numpy(dtype=float)
    Xg, Yg = np.meshgrid(x_vals, y_vals)
    Zg = pivot.to_numpy(dtype=float)

    return Xg, Yg, Zg, original_names

@st.cache_data
def get_contour_limits():
    Xg, Yg, _, _ = load_contour_grid()
    return float(np.min(Xg)), float(np.max(Xg)), float(np.min(Yg)), float(np.max(Yg))

def create_contour_plot(phi_fixed, phi_other):
    Xg, Yg, Zg, original_names = load_contour_grid()

    fig, ax = plt.subplots(figsize=(9, 6))
    contour = ax.contourf(Xg, Yg, Zg, levels=20)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(r"$f_{\mathrm{axial}}$ frequency")

    ax.scatter(
        [phi_fixed],
        [phi_other],
        s=120,
        facecolors="none",
        edgecolors="white",
        linewidths=2
    )

    ax.set_xlabel("ϕ (Fixed) [m/s]")
    ax.set_ylabel("ϕ (Free) [m/s]")
    ax.set_title("Bi-Material Axial Frequency")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

# =========================================================
# OPTIONAL MEDIA
# =========================================================
def show_output_image():
    img_path = optional_existing_path(IMAGE_CANDIDATES)
    if img_path is not None:
        st.image(str(img_path), use_container_width=True)

def show_output_video():
    video_path = optional_existing_path(VIDEO_CANDIDATES)
    if video_path is not None:
        st.subheader("Axial Deformation")
        st.video(video_path.read_bytes())

# =========================================================
# GUI
# =========================================================
st.title("Frequency Predictor")
st.write("Enter the material properties in SI units and click Predict.")

# SHOW IMAGE IMMEDIATELY UNDER THE TITLE AREA
show_output_image()

with st.sidebar:
    st.header("Input Material Properties")
    st.caption("Use SI units: E in N/m², ρ in kg/m³, and ν is dimensionless [-].")

    st.subheader("Fixed Material")
    E_Fixed = st.number_input("E (Fixed) [N/m²]", min_value=0.0, value=1.970000e11, format="%.6e")
    rho_Fixed = st.number_input("ρ (Fixed) [kg/m³]", min_value=0.0, value=7750.3, format="%.6f")
    nu_Fixed = st.number_input("ν (Fixed) [-]", min_value=0.0, max_value=0.4999, value=0.29, format="%.6f")

    st.subheader("Free Material")
    E_Other = st.number_input("E (Free) [N/m²]", min_value=0.0, value=4.240000e8, format="%.6e")
    rho_Other = st.number_input("ρ (Free) [kg/m³]", min_value=0.0, value=2200.5, format="%.6f")
    nu_Other = st.number_input("ν (Free) [-]", min_value=0.0, max_value=0.4999, value=0.45, format="%.6f")

    predict_btn = st.button("Predict", use_container_width=True)

if predict_btn:
    try:
        artifacts = load_artifacts()

        v = build_base_values(
            E_Fixed, rho_Fixed, nu_Fixed,
            E_Other, rho_Other, nu_Other
        )

        pw_df = build_piecewise_input_df(v)
        gb_df = build_gb_input_df(artifacts["gb_feature_names"], v)
        gam_df = build_gam_input_df(artifacts["gam_feature_names"], v)

        x_min, x_max, y_min, y_max = get_contour_limits()
        inside_contour_limits = (
            x_min <= v["Phi_Fixed"] <= x_max and
            y_min <= v["Phi_Other"] <= y_max
        )

        if not inside_contour_limits:
            st.error("Error - Exceedance of limits")
            st.markdown(
                f"""
                <div class="limit-box">
                    Allowed contour-data limits:<br>
                    ϕ (Fixed) [m/s]: {x_min:.6f} to {x_max:.6f}<br>
                    ϕ (Free) [m/s]: {y_min:.6f} to {y_max:.6f}
                </div>
                """,
                unsafe_allow_html=True,
            )

        enet_pred, region, split_var, threshold, split_value = predict_two_piece_elasticnet(
            artifacts["pw_pack"], pw_df, v
        )

        gb_pred = predict_gradient_boosting(artifacts["gb_model"], gb_df)
        gam_raw, gam_pred = predict_gam(artifacts["gam_model"], gam_df)

        st.subheader("Predicted Frequency (Hz)")
        c1, c2, c3 = st.columns(3)
        with c1:
            render_pred_card("2-piece ElasticNet", enet_pred)
        with c2:
            render_pred_card("GradientBoosting", gb_pred)
        with c3:
            render_pred_card("GAM", gam_pred)

        gam_gap = abs(gam_pred - gb_pred)
        if gam_gap > 10000:
            st.error(
                "Error - GAM differs from GradientBoosting by more than expected. "
                "This usually means the GAM is unstable for this input, often near or beyond its learned boundary, "
                "and the exp() step vary the raw GAM output."
            )
            st.markdown(
                f"""
                <div class="limit-box">
                    GradientBoosting = {gb_pred:.6f}<br>
                    GAM raw output = {gam_raw:.6f}<br>
                    GAM final output = {gam_pred:.6f}<br>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("Derived Features")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            render_small_card("ϕ (Fixed) [m/s]", f"{v['Phi_Fixed']:.6f}")
        with d2:
            render_small_card("ϕ (Free) [m/s]", f"{v['Phi_Other']:.6f}")
        with d3:
            render_small_card("√χ [-]", f"{v['sqrt_chi']:.6f}")
        with d4:
            render_small_card("log(χ) [-]", f"{v['log_chi']:.6f}")

        st.subheader("2-Piece ElasticNet Region")
        pretty_split_var = pretty_name(split_var)

        r1, r2, r3 = st.columns(3)
        with r1:
            render_info_card("Region", region)
        with r2:
            render_info_card("Split variable", pretty_split_var)
        with r3:
            render_info_card("Threshold", f"{threshold:.6g}")

        st.markdown(
            f'<div class="caption-text">Current {pretty_split_var} = {split_value:.6g}</div>',
            unsafe_allow_html=True,
        )

        show_output_video()

        if inside_contour_limits:
            st.subheader("Contour Plot of Axial Frequency")
            fig = create_contour_plot(v["Phi_Fixed"], v["Phi_Other"])
            st.pyplot(fig, use_container_width=True)

        with st.expander("Model inputs used"):
            st.write("2-piece ElasticNet input:")
            st.dataframe(rename_display_columns(pw_df), use_container_width=True, hide_index=True)

            st.write("Gradient Boosting input:")
            st.dataframe(rename_display_columns(gb_df), use_container_width=True, hide_index=True)

            st.write("GAM input:")
            st.dataframe(rename_display_columns(gam_df), use_container_width=True, hide_index=True)

        st.subheader("Meaning of ϕ and χ")
        eq1, eq2 = st.columns(2)
        with eq1:
            st.markdown('<div class="formula-box">', unsafe_allow_html=True)
            st.latex(r"\phi_{\mathrm{Fixed}}=\sqrt{\frac{E_{\mathrm{Fixed}}}{\rho_{\mathrm{Fixed}}}}")
            st.markdown("**Units:** [m/s]")
            st.markdown(f"**Value:** {v['Phi_Fixed']:.6f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with eq2:
            st.markdown('<div class="formula-box">', unsafe_allow_html=True)
            st.latex(r"\phi_{\mathrm{Free}}=\sqrt{\frac{E_{\mathrm{Free}}}{\rho_{\mathrm{Free}}}}")
            st.markdown("**Units:** [m/s]")
            st.markdown(f"**Value:** {v['Phi_Other']:.6f}")
            st.markdown("</div>", unsafe_allow_html=True)

        chi1, chi2 = st.columns(2)
        with chi1:
            st.markdown('<div class="formula-box">', unsafe_allow_html=True)
            st.latex(r"\chi=\frac{E_{\mathrm{Fixed}}/E_{\mathrm{Free}}}{\rho_{\mathrm{Fixed}}/\rho_{\mathrm{Free}}}")
            st.markdown("**Units:** [-]")
            st.markdown(f"**Value of χ:** {v['chi']:.6f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with chi2:
            st.markdown('<div class="formula-box">', unsafe_allow_html=True)
            st.latex(r"\sqrt{\chi}=\sqrt{\frac{E_{\mathrm{Fixed}}/E_{\mathrm{Free}}}{\rho_{\mathrm{Fixed}}/\rho_{\mathrm{Free}}}}")
            st.markdown("**Units:** [-]")
            st.markdown(f"**Value of √χ:** {v['sqrt_chi']:.6f}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Training / Validity Regions")
        st.markdown(
            '<div class="section-note">These are the exact lower and upper boundaries from summary_axial_all_cases.csv.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="limit-box">
                <b>Contour-data limits</b><br>
                ϕ (Fixed) [m/s]: {x_min:.6f} to {x_max:.6f}<br>
                ϕ (Free) [m/s]: {y_min:.6f} to {y_max:.6f}
            </div>
            """,
            unsafe_allow_html=True,
        )

        pw_table = build_exact_range_table(pw_df, EXACT_BOUNDS_PW)
        gb_table = build_exact_range_table(gb_df, EXACT_BOUNDS_GB)
        gam_table = build_exact_range_table(gam_df, EXACT_BOUNDS_GAM)

        st.write("**2-piece ElasticNet exact boundaries**")
        st.dataframe(pw_table, use_container_width=True, hide_index=True)

        st.write("**GradientBoosting exact boundaries**")
        st.dataframe(gb_table, use_container_width=True, hide_index=True)

        st.write("**GAM exact boundaries**")
        st.dataframe(gam_table, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error: {e}")
