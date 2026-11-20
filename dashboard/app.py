
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import gdown

from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CLOUD CONFIGURATION AND GOOGLE DRIVE ASSET LOADER
# ============================================================

ROOT = Path("/tmp/AI-Energy-Intelligence")
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

MODEL_FOLDER_ID = "1LQUFd9PTrO1wns0hA6HTfCZbYpMvoGCj"
DATA_FOLDER_ID = "1F-vAQnZn0xkvfMQbIkPG2iGSH5pFN_g5"

REQUIRED_MODELS = [
    "demand_forecasting_model.pkl",
    "load_shedding_model.pkl",
    "grid_isolation_forest.pkl",
    "grid_lof_model.pkl",
    "grid_scaler.pkl",
    "load_shedding_imputer.pkl",
    "solar_isolation_forest.pkl",
    "solar_lof_model.pkl",
    "solar_scaler.pkl",
    "solar_imputer.pkl"
]

REQUIRED_DATASETS = [
    "demand_data.pkl",
    "load_shedding_data.pkl",
    "grid_anomaly_data.pkl",
    "solar_anomaly_data.pkl"
]


@st.cache_resource(show_spinner="Preparing Google Drive models and datasets...")
def prepare_assets():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    folders = [
        (
            MODEL_FOLDER_ID,
            MODEL_DIR,
            REQUIRED_MODELS,
            "Models"
        ),
        (
            DATA_FOLDER_ID,
            DATA_DIR,
            REQUIRED_DATASETS,
            "Datasets"
        )
    ]

    for folder_id, target_dir, required_files, label in folders:

        missing = [
            filename
            for filename in required_files
            if not (target_dir / filename).exists()
        ]

        if missing:

            st.write(f"Downloading {label} from Google Drive...")

            url = f"https://drive.google.com/drive/folders/{folder_id}"

            gdown.download_folder(
                url=url,
                output=str(target_dir),
                quiet=False,
                use_cookies=False,
                remaining_ok=True
            )

        missing_after_download = [
            filename
            for filename in required_files
            if not (target_dir / filename).exists()
        ]

        if missing_after_download:

            raise FileNotFoundError(
                f"{label} download incomplete. Missing files: "
                + ", ".join(missing_after_download)
            )

    return True


try:
    prepare_assets()

except Exception as e:

    st.error(f"Google Drive asset loading failed: {e}")

    st.info(
        "Check that both Google Drive folders are accessible "
        "to anyone with the link."
    )

    st.stop()


# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: #0b1120;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263449;
}

.hero {
    background: linear-gradient(120deg, #142c4e, #123c43);
    border: 1px solid #28536b;
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 24px;
}

.hero h1 {
    color: white;
    font-size: 30px;
    font-weight: 800;
}

.hero p {
    color: #b8c9dc;
    font-size: 14px;
}

.metric-card, .module-card {
    background: #151f31;
    border: 1px solid #29384e;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}

.metric-label {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
}

.metric-value {
    color: #f8fafc;
    font-size: 25px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-sub {
    color: #94a3b8;
    font-size: 11px;
    margin-top: 4px;
}

.module-card h3 {
    color: #f8fafc;
    font-size: 16px;
}

.module-card p {
    color: #9caec3;
    font-size: 12px;
    line-height: 1.6;
}

.result-box {
    background: #142a2b;
    border: 1px solid #28605a;
    border-radius: 13px;
    padding: 18px;
    margin-top: 15px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #0d9488, #0891b2);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    width: 100%;
}

div[data-testid="stMetric"] {
    background: #151f31;
    border: 1px solid #29384e;
    padding: 14px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

h1, h2, h3, h4 {
    color: #f8fafc;
}

hr {
    border-color: #29384e;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE LOADING
# ============================================================

def load_dataframe(path):

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    errors = []

    for loader in (pd.read_pickle, joblib.load):

        try:
            obj = loader(path)

            if isinstance(obj, pd.DataFrame):
                return obj

            raise TypeError(
                f"{path.name} is not a DataFrame: {type(obj)}"
            )

        except Exception as e:
            errors.append(str(e))

    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, pd.DataFrame):
            return obj

    except Exception as e:
        errors.append(str(e))

    raise RuntimeError(
        f"Cannot load {path.name}: {' | '.join(errors)}"
    )


def load_model(path):

    if not path.exists():
        return None

    for loader in (joblib.load, pd.read_pickle):

        try:
            return loader(path)
        except Exception:
            pass

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Model loading failed: {path.name}: {e}")
        return None


@st.cache_data(show_spinner="Loading datasets...")
def load_datasets():

    return {
        "demand": load_dataframe(DATA_DIR / "demand_data.pkl"),
        "load": load_dataframe(DATA_DIR / "load_shedding_data.pkl"),
        "grid": load_dataframe(DATA_DIR / "grid_anomaly_data.pkl"),
        "solar": load_dataframe(DATA_DIR / "solar_anomaly_data.pkl")
    }


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():

    files = {
        "demand": "demand_forecasting_model.pkl",
        "load": "load_shedding_model.pkl",
        "grid": "grid_isolation_forest.pkl",
        "grid_lof": "grid_lof_model.pkl",
        "grid_scaler": "grid_scaler.pkl",
        "load_imputer": "load_shedding_imputer.pkl",
        "solar": "solar_isolation_forest.pkl",
        "solar_lof": "solar_lof_model.pkl",
        "solar_scaler": "solar_scaler.pkl",
        "solar_imputer": "solar_imputer.pkl"
    }

    return {
        key: load_model(MODEL_DIR / filename)
        for key, filename in files.items()
    }


try:
    DATA = load_datasets()
    MODELS = load_models()

    demand_df = DATA["demand"]
    load_df = DATA["load"]
    grid_df = DATA["grid"]
    solar_df = DATA["solar"]

    demand_model = MODELS["demand"]
    load_model_obj = MODELS["load"]
    grid_model = MODELS["grid"]
    grid_lof = MODELS["grid_lof"]
    grid_scaler = MODELS["grid_scaler"]
    load_imputer = MODELS["load_imputer"]
    solar_model = MODELS["solar"]
    solar_lof = MODELS["solar_lof"]
    solar_scaler = MODELS["solar_scaler"]
    solar_imputer = MODELS["solar_imputer"]

except Exception as e:
    st.error(f"Application loading error: {e}")
    st.stop()


# ============================================================
# COLUMN UTILITIES
# ============================================================

def normalize_name(value):

    return (
        str(value)
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def find_column(df, candidates):

    lookup = {
        normalize_name(c): c
        for c in df.columns
    }

    for candidate in candidates:

        key = normalize_name(candidate)

        if key in lookup:
            return lookup[key]

    return None


def get_datetime_column(df):

    col = find_column(
        df,
        [
            "timestamp",
            "datetime",
            "TimeStamp",
            "date_time",
            "date"
        ]
    )

    if col is not None:
        return col

    for col in df.columns:

        if "time" in str(col).lower():
            parsed = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            if parsed.notna().sum() > len(df) * 0.5:
                return col

    return None


def get_series(df, candidates):

    col = find_column(df, candidates)

    if col is None:
        return None

    return pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    result = df.copy()

    date_col = get_datetime_column(result)

    if date_col is not None:

        result["_dashboard_time"] = pd.to_datetime(
            result[date_col],
            errors="coerce"
        )

        result = result.sort_values(
            "_dashboard_time"
        ).reset_index(drop=True)

        dt = result["_dashboard_time"]

        result["hour"] = dt.dt.hour
        result["day"] = dt.dt.day
        result["dayofweek"] = dt.dt.dayofweek
        result["month"] = dt.dt.month

        result["is_weekend"] = (
            result["dayofweek"] >= 5
        ).astype(int)

        result["hour_sin"] = np.sin(
            2 * np.pi * result["hour"] / 24
        )

        result["hour_cos"] = np.cos(
            2 * np.pi * result["hour"] / 24
        )

        result["month_sin"] = np.sin(
            2 * np.pi * result["month"] / 12
        )

        result["month_cos"] = np.cos(
            2 * np.pi * result["month"] / 12
        )

    # --------------------------------------------------------
    # Source series
    # --------------------------------------------------------

    aliases = {
        "demand": [
            "demand_mw", "demand", "total_demand",
            "Demand_MW", "load_mw", "system_demand"
        ],
        "generation": [
            "generation_mw", "generation",
            "total_generation", "Generation_MW"
        ],
        "gas": [
            "gas", "gas_mw", "gas_generation",
            "gas_power"
        ],
        "liquid_fuel": [
            "liquid_fuel", "liquid_fuel_mw",
            "liquid_fuel_generation"
        ],
        "coal": [
            "coal", "coal_mw", "coal_generation"
        ],
        "hydro": [
            "hydro", "hydro_mw", "hydro_generation"
        ],
        "india_bheramara_hvdc": [
            "india_bheramara_hvdc",
            "india_bheramara",
            "bheramara_hvdc",
            "india_import_bheramara"
        ],
        "india_tripura": [
            "india_tripura",
            "tripura",
            "india_import_tripura"
        ],
        "load_shedding": [
            "load_shedding",
            "load_shedding_mw",
            "load_shedding_target",
            "is_load_shedding"
        ],
        "solar": [
            "solar",
            "solar_mw",
            "solar_generation",
            "solar_generation_mw"
        ]
    }

    source = {}

    for key, candidates in aliases.items():
        source[key] = get_series(result, candidates)

    # --------------------------------------------------------
    # Demand forecasting lags
    # --------------------------------------------------------

    if source["demand"] is not None:

        demand = source["demand"]

        for lag in [1, 2, 3, 6, 12, 24, 168]:

            name = f"demand_lag_{lag}h"

            if name not in result.columns:
                result[name] = demand.shift(lag)

            missing_name = f"{name}_missing"

            if missing_name not in result.columns:
                result[missing_name] = (
                    result[name].isna().astype(int)
                )

        for window in [3, 24]:

            mean_name = f"demand_rolling_mean_{window}"

            std_name = f"demand_rolling_std_{window}"

            if mean_name not in result.columns:
                result[mean_name] = (
                    demand.shift(1)
                    .rolling(window, min_periods=1)
                    .mean()
                )

            if std_name not in result.columns:
                result[std_name] = (
                    demand.shift(1)
                    .rolling(window, min_periods=2)
                    .std()
                )

            for name in [mean_name, std_name]:

                missing_name = f"{name}_missing"

                if missing_name not in result.columns:
                    result[missing_name] = (
                        result[name].isna().astype(int)
                    )

        if "demand_pct_change" not in result.columns:
            result["demand_pct_change"] = demand.pct_change()

        if "demand_change" not in result.columns:
            result["demand_change"] = demand.diff()

        for name in ["demand_pct_change", "demand_change"]:

            missing_name = f"{name}_missing"

            if missing_name not in result.columns:
                result[missing_name] = (
                    result[name].isna().astype(int)
                )

    # --------------------------------------------------------
    # Load-shedding lag features
    # --------------------------------------------------------

    load_feature_sources = {
        "demand_mw_lag_1h": "demand",
        "generation_mw_lag_1h": "generation",
        "gas_lag_1h": "gas",
        "liquid_fuel_lag_1h": "liquid_fuel",
        "coal_lag_1h": "coal",
        "hydro_lag_1h": "hydro",
        "india_bheramara_hvdc_lag_1h": "india_bheramara_hvdc",
        "india_tripura_lag_1h": "india_tripura",
        "load_shedding_lag_1h": "load_shedding"
    }

    for feature_name, source_name in load_feature_sources.items():

        series = source.get(source_name)

        if feature_name not in result.columns and series is not None:
            result[feature_name] = series.shift(1)

    # --------------------------------------------------------
    # Grid anomaly features
    # --------------------------------------------------------

    if source["generation"] is not None and source["demand"] is not None:

        generation = source["generation"]
        demand = source["demand"]

        if "generation_demand_difference" not in result.columns:
            result["generation_demand_difference"] = (
                generation - demand
            )

        if "generation_demand_ratio" not in result.columns:
            result["generation_demand_ratio"] = (
                generation / demand.replace(0, np.nan)
            )

    if source["solar"] is not None:

        if "solar_missing" not in result.columns:
            result["solar_missing"] = (
                source["solar"].isna().astype(int)
            )

    # Calendar features may be missing in the dataset.
    if "_dashboard_time" in result.columns:

        dt = result["_dashboard_time"]

        calendar = {
            "hour": dt.dt.hour,
            "day": dt.dt.day,
            "dayofweek": dt.dt.dayofweek,
            "month": dt.dt.month,
            "is_weekend": (dt.dt.dayofweek >= 5).astype(int),
            "hour_sin": np.sin(2 * np.pi * dt.dt.hour / 24),
            "hour_cos": np.cos(2 * np.pi * dt.dt.hour / 24),
            "month_sin": np.sin(2 * np.pi * dt.dt.month / 12),
            "month_cos": np.cos(2 * np.pi * dt.dt.month / 12)
        }

        for name, values in calendar.items():

            if name not in result.columns:
                result[name] = values

    return result


# Engineer all available datasets once.
demand_features_df = engineer_features(demand_df)
load_features_df = engineer_features(load_df)
grid_features_df = engineer_features(grid_df)
solar_features_df = engineer_features(solar_df)


# ============================================================
# EXACT MODEL FEATURE SCHEMAS
# ============================================================

def get_model_features(model, transformer=None):

    if model is not None:

        names = getattr(
            model,
            "feature_names_in_",
            None
        )

        if names is not None:
            return list(names)

    if transformer is not None:

        names = getattr(
            transformer,
            "feature_names_in_",
            None
        )

        if names is not None:
            return list(names)

    return None


def get_defaults(df, features):

    if df.empty:
        return {}

    row = df.iloc[-1]

    defaults = {}

    for feature in features:

        if feature in df.columns:

            value = row[feature]

            if pd.isna(value):

                values = pd.to_numeric(
                    df[feature],
                    errors="coerce"
                )

                value = values.median()

            if pd.isna(value):
                value = 0.0

            defaults[feature] = float(value)

        else:

            if feature.endswith("_missing"):
                defaults[feature] = 0.0

            elif feature in [
                "hour", "day", "dayofweek", "month",
                "is_weekend", "hour_sin", "hour_cos",
                "month_sin", "month_cos"
            ]:
                defaults[feature] = 0.0

            else:
                defaults[feature] = None

    return defaults


def check_schema(model, transformer, df, module_name):

    features = get_model_features(
        model,
        transformer
    )

    if features is None:

        st.error(
            f"{module_name}: the saved model does not expose "
            "its exact training feature names. Prediction is "
            "disabled rather than using guessed columns."
        )

        return None

    model_count = getattr(
        model,
        "n_features_in_",
        len(features)
    )

    if len(features) != model_count:

        st.error(
            f"{module_name}: model expects {model_count} features "
            f"but its stored feature list has {len(features)}."
        )

        return None

    transformer_count = getattr(
        transformer,
        "n_features_in_",
        len(features)
    ) if transformer is not None else len(features)

    if transformer_count != len(features):

        st.error(
            f"{module_name}: preprocessing expects "
            f"{transformer_count} features, but the model schema "
            f"contains {len(features)}."
        )

        return None

    engineered = engineer_features(df)

    missing = [
        feature
        for feature in features
        if feature not in engineered.columns
    ]

    if missing:

        st.error(
            f"{module_name}: these features could not be generated "
            "from the saved dataset: "
            + ", ".join(missing)
        )

        st.caption(
            "The original training feature list and source columns "
            "must be available to reconstruct these values."
        )

        return None

    return features


# ============================================================
# INPUT FORM
# ============================================================

def input_form(df, features, form_key, columns=2):

    defaults = get_defaults(df, features)

    unavailable = [
        f for f in features
        if defaults.get(f) is None
    ]

    if unavailable:

        st.error(
            "Cannot initialize these model inputs from the dataset: "
            + ", ".join(unavailable)
        )

        return None, False

    values = {}

    with st.form(form_key):

        st.markdown("### Model Input Parameters")

        st.caption(
            "The initial values are taken from the latest available "
            "engineered observation. You can modify them before analysis."
        )

        cols = st.columns(columns)

        for i, feature in enumerate(features):

            default = defaults[feature]

            with cols[i % columns]:

                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=float(default),
                    format="%.5f",
                    key=f"{form_key}_{feature}"
                )

        submitted = st.form_submit_button(
            "Run Analysis"
        )

    X = pd.DataFrame(
        [values],
        columns=features
    )

    return X, submitted


# ============================================================
# PREPROCESSING
# ============================================================

def transform_input(X, transformer):

    result = X.copy()

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if transformer is not None:

        transformed = transformer.transform(result)

        result = pd.DataFrame(
            transformed,
            columns=X.columns,
            index=X.index
        )

    return result


def get_positive_probability(model, X):

    probabilities = model.predict_proba(X)[0]

    classes = list(model.classes_)

    if 1 in classes:
        return float(probabilities[classes.index(1)])

    if True in classes:
        return float(probabilities[classes.index(True)])

    raise ValueError(
        "Positive class 1 was not found in model.classes_."
    )


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def metric_card(label, value, subtitle=""):

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def plot_bar(labels, values, title):

    fig, ax = plt.subplots(figsize=(8, 3.5))

    fig.patch.set_facecolor("#151f31")
    ax.set_facecolor("#151f31")

    bars = ax.bar(labels, values)

    ax.set_title(title, color="white")
    ax.tick_params(colors="#cbd5e1")
    ax.grid(axis="y", alpha=0.15)

    for spine in ax.spines.values():
        spine.set_color("#34445c")

    for bar in bars:

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():,.0f}",
            ha="center",
            va="bottom",
            color="white"
        )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_line(df, x, y, title, ylabel):

    fig, ax = plt.subplots(figsize=(10, 3.5))

    fig.patch.set_facecolor("#151f31")
    ax.set_facecolor("#151f31")

    ax.plot(df[x], df[y], linewidth=1.5)

    ax.set_title(title, color="white")
    ax.set_ylabel(ylabel, color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    ax.grid(alpha=0.15)

    for spine in ax.spines.values():
        spine.set_color("#34445c")

    fig.autofmt_xdate()
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## ⚡ AI ENERGY")
st.sidebar.caption("Bangladesh Power System Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Demand Forecasting",
        "Load-Shedding Prediction",
        "Grid Anomaly Detection",
        "Solar Anomaly Detection",
        "Model Performance & Results"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("PGCB Historical Electricity Dataset")
st.sidebar.caption("2015–2026")


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>AI Energy Intelligence</h1>
    <p>
        An integrated analytics platform for Bangladesh's electricity
        demand, load-shedding events, grid behaviour and solar generation.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("System Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Demand Records",
            f"{len(demand_df):,}",
            "Forecasting dataset"
        )

    with c2:
        metric_card(
            "Load-Shedding F1",
            "89.55%",
            "HistGradientBoosting"
        )

    with c3:
        metric_card(
            "Demand R²",
            "0.9651",
            "ExtraTreesRegressor"
        )

    with c4:
        metric_card(
            "Solar Anomaly Rate",
            "1.84%",
            "Adaptive baseline"
        )

    st.subheader("Four Analytics Modules")

    cols = st.columns(4)

    modules = [
        (
            "01 | Demand Forecasting",
            "Forecasts electricity demand using historical and temporal features."
        ),
        (
            "02 | Load-Shedding Prediction",
            "Estimates the probability of a load-shedding event."
        ),
        (
            "03 | Grid Anomaly Detection",
            "Identifies unusual aggregate grid-level electricity behaviour."
        ),
        (
            "04 | Solar Anomaly Detection",
            "Identifies unusually high or low solar generation."
        )
    ]

    for col, item in zip(cols, modules):

        with col:

            st.markdown(
                f"""
                <div class="module-card">
                    <h3>{item[0]}</h3>
                    <p>{item[1]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.subheader("System Analytics")

    left, right = st.columns(2)

    with left:

        st.markdown("#### Historical Electricity Demand")

        demand_col = find_column(
            demand_df,
            [
                "demand_mw",
                "Demand_MW",
                "demand",
                "total_demand"
            ]
        )

        date_col = get_datetime_column(demand_df)

        if demand_col and date_col:

            temp = demand_df[[date_col, demand_col]].copy()

            temp[date_col] = pd.to_datetime(
                temp[date_col],
                errors="coerce"
            )

            temp[demand_col] = pd.to_numeric(
                temp[demand_col],
                errors="coerce"
            )

            temp = temp.dropna().tail(500)

            if not temp.empty:

                plot_line(
                    temp,
                    date_col,
                    demand_col,
                    "Recent Historical Electricity Demand",
                    "Demand (MW)"
                )

            else:
                st.info("Demand trend unavailable.")

        else:
            st.info("Demand or timestamp column not found.")

    with right:

        st.markdown("#### Load-Shedding Distribution")

        target_col = find_column(
            load_df,
            [
                "load_shedding",
                "load_shedding_target",
                "is_load_shedding",
                "target"
            ]
        )

        if target_col:

            counts = load_df[target_col].value_counts().sort_index()

            plot_bar(
                counts.index.astype(str),
                counts.values,
                "Observed Load-Shedding Labels"
            )

        else:
            st.info("Load-shedding target column not found.")

    left, right = st.columns(2)

    with left:

        st.markdown("#### Grid Anomaly Detection")

        st.metric(
            "Isolation Forest Flagged Observations",
            "3,363"
        )

        st.metric(
            "LOF Flagged Observations",
            "4,306"
        )

        st.caption(
            "Historical unsupervised detection results."
        )

    with right:

        st.markdown("#### Solar Anomaly Distribution")

        plot_bar(
            [
                "Normal",
                "Low",
                "High",
                "Unscored"
            ],
            [
                39130,
                484,
                251,
                195
            ],
            "Adaptive Baseline Solar Status"
        )

    st.caption(
        "Historical visualizations summarize stored experimental results. "
        "Interactive predictions are available in the individual modules."
    )


# ============================================================
# DEMAND FORECASTING
# ============================================================

elif page == "Demand Forecasting":

    st.header("Electricity Demand Forecasting")

    st.write(
        "Modify the trained model's input features to estimate electricity demand."
    )

    if demand_model is None:

        st.error("Demand forecasting model is unavailable.")

    else:

        features = check_schema(
            demand_model,
            None,
            demand_features_df,
            "Demand forecasting"
        )

        if features is not None:

            X, submitted = input_form(
                demand_features_df,
                features,
                "demand_form"
            )

            if submitted and X is not None:

                try:

                    prediction = float(
                        demand_model.predict(X)[0]
                    )

                    st.markdown(
                        '<div class="result-box"><h3>Forecast Result</h3></div>',
                        unsafe_allow_html=True
                    )

                    a, b = st.columns(2)

                    a.metric(
                        "Predicted Electricity Demand",
                        f"{prediction:,.2f} MW"
                    )

                    b.metric(
                        "Prediction Model",
                        "ExtraTreesRegressor"
                    )

                    st.success(
                        f"Predicted demand: {prediction:,.2f} MW"
                    )

                except Exception as e:

                    st.error(
                        f"Demand prediction failed: {e}"
                    )

    st.subheader("Model Evaluation")

    a, b, c = st.columns(3)

    a.metric("MAE", "246.18 MW")
    b.metric("RMSE", "428.93 MW")
    c.metric("R² Score", "0.9651")


# ============================================================
# LOAD-SHEDDING
# ============================================================

elif page == "Load-Shedding Prediction":

    st.header("Load-Shedding Probability Prediction")

    st.write(
        "Modify the model input features to estimate load-shedding probability."
    )

    if load_model_obj is None:

        st.error("Load-shedding model is unavailable.")

    else:

        # FIX: Validate only against the model's 17-feature schema.
        # The saved imputer expects 44 features and is not used here.
        features = check_schema(
            load_model_obj,
            None,
            load_features_df,
            "Load-shedding prediction"
        )

        if features is not None:

            X, submitted = input_form(
                load_features_df,
                features,
                "load_form"
            )

            if submitted and X is not None:

                try:

                    # FIX: Skip the incompatible 44-feature imputer.
                    # Clean the exact 17 model features directly.
                    X_processed = (
                        X.replace([np.inf, -np.inf], np.nan)
                         .fillna(0)
                    )

                    probability = get_positive_probability(
                        load_model_obj,
                        X_processed
                    )

                    threshold = 0.46

                    prediction = int(
                        probability >= threshold
                    )

                    st.markdown(
                        '<div class="result-box"><h3>Prediction Result</h3></div>',
                        unsafe_allow_html=True
                    )

                    a, b = st.columns(2)

                    a.metric(
                        "Load-Shedding Probability",
                        f"{probability * 100:.2f}%"
                    )

                    b.metric(
                        "Prediction",
                        "Event Expected"
                        if prediction
                        else "No Event Expected"
                    )

                    st.progress(
                        float(np.clip(probability, 0, 1))
                    )

                    if prediction:

                        st.error(
                            "The model predicts a potential load-shedding event."
                        )

                    else:

                        st.success(
                            "The model does not predict a load-shedding event."
                        )

                    st.caption(
                        "Decision threshold: 0.46. This is a model-estimated "
                        "probability, not a guarantee of an actual event."
                    )

                except Exception as e:

                    st.error(
                        f"Load-shedding prediction failed: {e}"
                    )

    st.subheader("Model Evaluation")

    a, b, c, d = st.columns(4)

    a.metric("Accuracy", "94.05%")
    b.metric("Precision", "88.71%")
    c.metric("Recall", "90.41%")
    d.metric("F1-Score", "89.55%")

    a, b, c = st.columns(3)

    a.metric("ROC-AUC", "0.9720")
    b.metric("PR-AUC", "0.9457")
    c.metric("Brier Score", "0.0493")


# ============================================================
# GRID ANOMALY
# ============================================================

elif page == "Grid Anomaly Detection":

    st.header("Grid-Level Anomaly Detection")

    st.write(
        "Analyze unusual aggregate grid-level electricity behaviour."
    )

    if grid_model is None:

        st.error("Grid Isolation Forest model is unavailable.")

    else:

        features = check_schema(
            grid_model,
            grid_scaler,
            grid_features_df,
            "Grid anomaly detection"
        )

        if features is not None:

            X, submitted = input_form(
                grid_features_df,
                features,
                "grid_form"
            )

            if submitted and X is not None:

                try:

                    X_processed = transform_input(
                        X,
                        grid_scaler
                    )

                    prediction = int(
                        grid_model.predict(X_processed)[0]
                    )

                    score = float(
                        grid_model.decision_function(X_processed)[0]
                    )

                    st.markdown(
                        '<div class="result-box"><h3>Grid Anomaly Assessment</h3></div>',
                        unsafe_allow_html=True
                    )

                    a, b = st.columns(2)

                    a.metric(
                        "Detection Result",
                        "Anomaly Detected"
                        if prediction == -1
                        else "Normal Behaviour"
                    )

                    b.metric(
                        "Decision Score",
                        f"{score:.5f}"
                    )

                    if prediction == -1:

                        st.error(
                            "Unusual grid-level behaviour detected. "
                            "This observation should be investigated."
                        )

                    else:

                        st.success(
                            "The observation is classified as normal "
                            "grid-level behaviour."
                        )

                    st.caption(
                        "Isolation Forest assigns -1 to anomalies and 1 "
                        "to normal observations. Lower decision scores "
                        "indicate more anomalous observations."
                    )

                    if grid_lof is not None:

                        if getattr(grid_lof, "novelty", False):

                            try:

                                lof_prediction = int(
                                    grid_lof.predict(X_processed)[0]
                                )

                                st.markdown("#### LOF Diagnostic")

                                st.write(
                                    "Anomaly detected"
                                    if lof_prediction == -1
                                    else "Normal according to LOF"
                                )

                            except Exception as e:

                                st.caption(
                                    f"LOF diagnostic unavailable: {e}"
                                )

                        else:

                            st.caption(
                                "The saved LOF model was fitted for "
                                "training-set analysis and does not "
                                "support new-observation prediction."
                            )

                except Exception as e:

                    st.error(
                        f"Grid anomaly analysis failed: {e}"
                    )

    st.subheader("Historical Detection Summary")

    a, b, c = st.columns(3)

    a.metric("Test Observations", "18,983")
    b.metric("Isolation Forest Flags", "3,363")
    c.metric("IF Flag Rate", "17.72%")

    st.caption(
        "This is an unsupervised task without verified ground-truth "
        "labels. Accuracy and F1-score are not reported."
    )


# ============================================================
# SOLAR ANOMALY
# ============================================================

elif page == "Solar Anomaly Detection":

    st.header("Solar Generation Anomaly Detection")

    st.write(
        "Evaluate solar generation against an adaptive historical baseline."
    )

    st.info(
        "This interactive assessment applies the project's robust "
        "median-MAD baseline rule. It is not presented as a live "
        "Isolation Forest or LOF prediction."
    )

    with st.form("solar_form"):

        c1, c2 = st.columns(2)

        with c1:

            generation = st.number_input(
                "Current Solar Generation (MW)",
                min_value=0.0,
                value=100.0,
                step=5.0
            )

            adaptive_median = st.number_input(
                "Adaptive Historical Median (MW)",
                min_value=0.0,
                value=100.0,
                step=5.0
            )

        with c2:

            adaptive_scale = st.number_input(
                "Adaptive Robust Scale (MW)",
                min_value=0.01,
                value=10.0,
                step=1.0
            )

            threshold = st.number_input(
                "Robust Z-Score Threshold",
                min_value=0.1,
                value=3.5,
                step=0.1
            )

        submitted = st.form_submit_button(
            "Analyze Solar Generation"
        )

    if submitted:

        robust_z = (
            generation - adaptive_median
        ) / max(adaptive_scale, 0.01)

        if robust_z > threshold:

            status = "High Generation Anomaly"

        elif robust_z < -threshold:

            status = "Low Generation Anomaly"

        else:

            status = "Normal"

        a, b, c = st.columns(3)

        a.metric(
            "Robust Z-Score",
            f"{robust_z:.3f}"
        )

        b.metric(
            "Adaptive Median",
            f"{adaptive_median:,.2f} MW"
        )

        c.metric(
            "Threshold",
            f"±{threshold:.2f}"
        )

        if status == "Normal":

            st.success(
                "Solar generation is within the expected range."
            )

        elif status == "High Generation Anomaly":

            st.error(
                "Solar generation is unusually high compared with "
                "the adaptive baseline."
            )

        else:

            st.warning(
                "Solar generation is unusually low compared with "
                "the adaptive baseline."
            )

        st.markdown(f"**Final Status:** {status}")

    st.subheader("Historical Solar Results")

    a, b, c = st.columns(3)

    a.metric("Scored Observations", "39,865")
    b.metric("Anomaly Candidates", "735")
    c.metric("Anomaly Rate", "1.84%")

    st.caption(
        "The historical adaptive baseline uses a rolling median "
        "and MAD-based robust scale. Isolation Forest and LOF "
        "are supporting diagnostic methods."
    )


# ============================================================
# MODEL PERFORMANCE & RESULTS
# ============================================================

elif page == "Model Performance & Results":

    st.header("Model Performance & Results")

    st.write(
        "Algorithms, evaluation methods and experimental results "
        "from all four analytics modules."
    )

    st.subheader("Models Used")

    results = pd.DataFrame([
        {
            "Module": "Demand Forecasting",
            "Primary Model": "ExtraTreesRegressor",
            "Additional Methods": "Lag features, rolling statistics",
            "Evaluation": "MAE, RMSE, R²"
        },
        {
            "Module": "Load-Shedding Prediction",
            "Primary Model": "HistGradientBoostingClassifier",
            "Additional Methods": "Probability threshold tuning",
            "Evaluation": "Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier"
        },
        {
            "Module": "Grid Anomaly Detection",
            "Primary Model": "Isolation Forest",
            "Additional Methods": "LOF, statistical baseline",
            "Evaluation": "Flag rate and method comparison"
        },
        {
            "Module": "Solar Anomaly Detection",
            "Primary Model": "Adaptive Median-MAD Baseline",
            "Additional Methods": "Isolation Forest, LOF",
            "Evaluation": "Anomaly count and anomaly rate"
        }
    ])

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.subheader("Module 1: Electricity Demand Forecasting")

    a, b, c = st.columns(3)

    a.metric("MAE", "246.18 MW")
    b.metric("RMSE", "428.93 MW")
    c.metric("R² Score", "0.9651")

    st.caption(
        "ExtraTreesRegressor. Chronological evaluation with "
        "18,098 test observations."
    )

    st.markdown("---")
    st.subheader("Module 2: Load-Shedding Prediction")

    a, b, c, d = st.columns(4)

    a.metric("Accuracy", "94.05%")
    b.metric("Precision", "88.71%")
    c.metric("Recall", "90.41%")
    d.metric("F1-Score", "89.55%")

    a, b, c = st.columns(3)

    a.metric("ROC-AUC", "0.9720")
    b.metric("PR-AUC", "0.9457")
    c.metric("Brier Score", "0.0493")

    st.caption(
        "HistGradientBoostingClassifier with selected probability threshold 0.46."
    )

    st.markdown("---")
    st.subheader("Module 3: Grid-Level Anomaly Detection")

    a, b, c = st.columns(3)

    a.metric("Test Observations", "18,983")
    b.metric("Isolation Forest Flags", "3,363")
    c.metric("IF Flag Rate", "17.72%")

    grid_results = pd.DataFrame([
        {
            "Method": "Isolation Forest",
            "Flagged Observations": 3363,
            "Flag Rate": "17.72%"
        },
        {
            "Method": "Local Outlier Factor",
            "Flagged Observations": 4306,
            "Flag Rate": "22.68%"
        },
        {
            "Method": "Statistical Baseline",
            "Flagged Observations": 475,
            "Flag Rate": "2.50%"
        }
    ])

    st.dataframe(
        grid_results,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Unsupervised detection without verified ground-truth labels. "
        "Flagged observations are candidates for review, not confirmed faults."
    )

    st.markdown("---")
    st.subheader("Module 4: Solar Generation Anomaly Detection")

    a, b, c = st.columns(3)

    a.metric("Scored Observations", "39,865")
    b.metric("Anomaly Candidates", "735")
    c.metric("Anomaly Rate", "1.84%")

    solar_results = pd.DataFrame([
        {"Status": "Normal", "Observations": 39130},
        {"Status": "Low Generation Anomaly", "Observations": 484},
        {"Status": "High Generation Anomaly", "Observations": 251},
        {"Status": "Unscored", "Observations": 195}
    ])

    st.dataframe(
        solar_results,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Adaptive median-MAD baseline is the primary solar anomaly method. "
        "The anomaly rate is calculated over scored observations."
    )

    st.markdown("---")

    st.info(
        "Demand forecasting and load-shedding prediction use supervised "
        "test-set metrics. Grid and solar anomaly detection are unsupervised "
        "or baseline-based tasks. Verified ground-truth anomaly labels are "
        "unavailable, so accuracy, precision, recall and F1-score are not claimed."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#718096;font-size:11px;">
        AI Energy Intelligence | Bangladesh Power System Analytics<br>
        Research Dashboard | PGCB Historical Electricity Dataset
    </div>
    """,
    unsafe_allow_html=True
)
