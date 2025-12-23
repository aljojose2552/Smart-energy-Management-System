import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from io import BytesIO
import plotly.express as px

# -----------------------------
# App config
# -----------------------------
st.set_page_config(layout="wide", page_title="Smart Energy Management: Clustering + Simulation")
st.title("🏡 Smart Energy Management: Clustering + Demand Response Simulation")

DATE_COL = "date"
HOUSEHOLD_COL = "household"
CONS_COL = "Consumption(Wh)"
GRID_COL = "From grid(Wh)"

# Peak window for feature engineering (match your training logic)
EVENING_START = 17
EVENING_END = 23

# Model artifacts
MODEL_DIR = Path("household_clustering_model")
KMEANS_PATH = MODEL_DIR / "kmeans_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.pkl"

# Optional: change these after you interpret clusters
CLUSTER_LABELS = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
}

# -----------------------------
# Helpers
# -----------------------------
def strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

def require_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]

def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.dropna(subset=[col])

def ensure_numeric_nonneg(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df[col] = df[col].clip(lower=0)
    return df

def load_artifacts():
    if not (KMEANS_PATH.exists() and SCALER_PATH.exists() and FEATURES_PATH.exists()):
        return None, None, None
    kmeans = joblib.load(KMEANS_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return kmeans, scaler, feature_cols

def build_features_for_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Must match your notebook training feature engineering.
    Expects at least: date, Consumption(Wh). Optional: household.
    """
    df = strip_cols(df)
    if HOUSEHOLD_COL not in df.columns:
        df[HOUSEHOLD_COL] = "NEW_HOUSEHOLD"

    df = ensure_datetime(df, DATE_COL)
    df = ensure_numeric_nonneg(df, CONS_COL)

    df["hour"] = df[DATE_COL].dt.hour
    df["weekday"] = df[DATE_COL].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    feats = df.groupby(HOUSEHOLD_COL)[CONS_COL].agg(
        total_Wh="sum",
        mean_Wh="mean",
        peak_Wh="max",
        std_Wh="std"
    ).reset_index()

    evening = df[(df["hour"] >= EVENING_START) & (df["hour"] <= EVENING_END)] \
        .groupby(HOUSEHOLD_COL)[CONS_COL].sum().reset_index(name="evening_Wh")
    feats = feats.merge(evening, on=HOUSEHOLD_COL, how="left")
    feats["evening_Wh"] = feats["evening_Wh"].fillna(0)
    feats["evening_ratio"] = feats["evening_Wh"] / (feats["total_Wh"] + 1e-9)

    weekend = df[df["is_weekend"] == 1] \
        .groupby(HOUSEHOLD_COL)[CONS_COL].sum().reset_index(name="weekend_Wh")
    feats = feats.merge(weekend, on=HOUSEHOLD_COL, how="left")
    feats["weekend_Wh"] = feats["weekend_Wh"].fillna(0)
    feats["weekend_ratio"] = feats["weekend_Wh"] / (feats["total_Wh"] + 1e-9)

    feats["std_Wh"] = feats["std_Wh"].fillna(0)
    return feats

@st.cache_data(show_spinner=False)
def run_simulation(file_bytes: bytes, efficiency_measures: bool, demand_response: bool):
    """
    Demand-response simulation using uploaded CSV bytes.
    Required cols: date, household, Consumption(Wh), From grid(Wh)
    Returns: df_scaled, grid_summary_scaled, err
    """
    if not file_bytes:
        return None, None, "Uploaded file is empty (0 bytes)."

    df = pd.read_csv(BytesIO(file_bytes))
    df = strip_cols(df)

    # Validate required columns
    required = [DATE_COL, HOUSEHOLD_COL, CONS_COL, GRID_COL]
    missing = require_columns(df, required)
    if missing:
        return None, None, f"Missing columns: {missing}"

    df = ensure_datetime(df, DATE_COL)
    df = ensure_numeric_nonneg(df, CONS_COL)
    df = ensure_numeric_nonneg(df, GRID_COL)

    df["hour"] = df[DATE_COL].dt.hour
    df_scaled = df.copy()

    # Efficiency (simple assumption): reduce grid import by 10%
    if efficiency_measures:
        df_scaled[GRID_COL] = df_scaled[GRID_COL] * 0.9

    # Demand response: classify peak hours based on 75th percentile of average hourly consumption
    if demand_response:
        hourly_avg = df_scaled.groupby("hour")[CONS_COL].mean()
        threshold = hourly_avg.quantile(0.75)
        peak_hours = hourly_avg[hourly_avg >= threshold].index.tolist()

        df_scaled["demand_period"] = np.where(df_scaled["hour"].isin(peak_hours), "Peak", "Off-peak")

        # Vectorized shifting: peak rows reduced to 80% / 70%, off-peak unchanged
        is_peak = (df_scaled["demand_period"] == "Peak")
        df_scaled["Adjusted_20%"] = df_scaled[GRID_COL] * np.where(is_peak, 0.8, 1.0)
        df_scaled["Adjusted_30%"] = df_scaled[GRID_COL] * np.where(is_peak, 0.7, 1.0)
    else:
        df_scaled["demand_period"] = "N/A"
        df_scaled["Adjusted_20%"] = df_scaled[GRID_COL]
        df_scaled["Adjusted_30%"] = df_scaled[GRID_COL]

    # Aggregate per household
    grid_summary_scaled = df_scaled.groupby(HOUSEHOLD_COL).agg(
        **{
            "Original Grid Use(Wh)": (GRID_COL, "sum"),
            "Grid Use after 20% Shift": ("Adjusted_20%", "sum"),
            "Grid Use after 30% Shift": ("Adjusted_30%", "sum"),
        }
    ).reset_index()

    grid_summary_scaled["Savings_20% (Wh)"] = grid_summary_scaled["Original Grid Use(Wh)"] - grid_summary_scaled["Grid Use after 20% Shift"]
    grid_summary_scaled["Savings_30% (Wh)"] = grid_summary_scaled["Original Grid Use(Wh)"] - grid_summary_scaled["Grid Use after 30% Shift"]
    grid_summary_scaled["Savings_20% (%)"] = (grid_summary_scaled["Savings_20% (Wh)"] / (grid_summary_scaled["Original Grid Use(Wh)"] + 1e-9)) * 100
    grid_summary_scaled["Savings_30% (%)"] = (grid_summary_scaled["Savings_30% (Wh)"] / (grid_summary_scaled["Original Grid Use(Wh)"] + 1e-9)) * 100

    return df_scaled, grid_summary_scaled.round(2), None


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["🧠 Cluster Assignment", "⚡ Demand Response Simulation"])


# =============================
# TAB 1: Cluster assignment
# =============================
with tab1:
    st.subheader("Upload a new household CSV → get cluster pattern")
    st.caption("Required columns: `date`, `Consumption(Wh)` (optional: `household`).")

    kmeans, scaler, feature_cols = load_artifacts()

    colA, colB = st.columns([2, 1])

    with colB:
        st.markdown("### Model status")
        st.write(f"Looking for artifacts in: `{MODEL_DIR.resolve()}`")
        st.write(f"- `{KMEANS_PATH.name}`: {'✅' if KMEANS_PATH.exists() else '❌'}")
        st.write(f"- `{SCALER_PATH.name}`: {'✅' if SCALER_PATH.exists() else '❌'}")
        st.write(f"- `{FEATURES_PATH.name}`: {'✅' if FEATURES_PATH.exists() else '❌'}")

        if kmeans is None:
            st.warning("Artifacts not found. Train & save the K-Means model in your notebook first.")
            st.info("Your notebook must save: kmeans_model.pkl, scaler.pkl, feature_columns.pkl into household_clustering_model/")

    with colA:
        uploaded_cluster = st.file_uploader("Upload new household CSV (for clustering)", type="csv", key="cluster_upload")
        if uploaded_cluster is None:
            st.info("Upload a CSV to classify a household.")
        else:
            file_bytes = uploaded_cluster.getvalue()
            if not file_bytes:
                st.error("Uploaded file is empty (0 bytes).")
            else:
                df_new = pd.read_csv(BytesIO(file_bytes))
                df_new = strip_cols(df_new)

                st.markdown("#### Diagnostic: detected columns")
                st.write(df_new.columns.tolist())

                missing = require_columns(df_new, [DATE_COL, CONS_COL])
                if missing:
                    st.error(f"Missing required columns for clustering: {missing}")
                elif kmeans is None:
                    st.error("Model artifacts are missing — cannot assign clusters yet.")
                else:
                    feats = build_features_for_clustering(df_new)
                    X = feats.reindex(columns=[HOUSEHOLD_COL] + list(feature_cols), fill_value=0)
                    X_mat = X[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0)

                    X_scaled = scaler.transform(X_mat)
                    feats["cluster"] = kmeans.predict(X_scaled)
                    feats["cluster_label"] = feats["cluster"].map(CLUSTER_LABELS).fillna(feats["cluster"].astype(str))

                    st.markdown("### Cluster assignment result")
                    st.dataframe(feats, use_container_width=True)

                    # Hourly profile plot
                    try:
                        df_plot = ensure_datetime(df_new, DATE_COL)
                        df_plot = ensure_numeric_nonneg(df_plot, CONS_COL)
                        df_plot["hour"] = pd.to_datetime(df_plot[DATE_COL]).dt.hour
                        hourly = df_plot.groupby("hour")[CONS_COL].mean().reset_index()

                        fig = px.line(hourly, x="hour", y=CONS_COL, markers=True,
                                      title="Uploaded household: average hourly consumption")
                        fig.update_xaxes(dtick=1)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.warning("Could not plot hourly profile (check your `date` and `Consumption(Wh)` values).")


# =============================
# TAB 2: Simulation
# =============================
with tab2:
    st.subheader("Upload household grid-import data → simulate demand response savings")
    st.caption("Required columns: `date`, `household`, `Consumption(Wh)`, `From grid(Wh)`.")

    with st.sidebar:
        st.header("Scenario controls (Simulation tab)")
        efficiency_measures = st.checkbox("Apply Energy Efficiency Measures (10% grid reduction)", value=False, key="eff")
        demand_response = st.checkbox("Enable Demand Response (peak shifting)", value=False, key="dr")
        st.divider()
        st.caption("These controls affect only the Simulation tab.")

    uploaded_sim = st.file_uploader("Upload CSV (for simulation)", type="csv", key="sim_upload")

    if uploaded_sim is None:
        st.info("Upload a simulation CSV to run demand-response analysis.")
    else:
        # Read bytes ONCE
        sim_bytes = uploaded_sim.getvalue()
        if not sim_bytes:
            st.error("Uploaded file is empty (0 bytes). Please re-export and upload again.")
            st.stop()

        # Preview + diagnostics
        df_preview = pd.read_csv(BytesIO(sim_bytes))
        df_preview = strip_cols(df_preview)

        st.markdown("#### Diagnostic: detected columns")
        st.write(df_preview.columns.tolist())

        st.markdown("#### Preview")
        st.dataframe(df_preview.head(50), use_container_width=True)

        st.write("Uploaded file size (bytes):", len(sim_bytes))

        run_btn = st.button("Run Simulation", type="primary")
        if run_btn:
            with st.spinner("Running simulation..."):
                df_scaled, grid_summary_scaled, err = run_simulation(sim_bytes, efficiency_measures, demand_response)

            if err:
                st.error(err)
            else:
                st.success("Simulation completed.")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("### Grid usage by household")
                    fig1 = px.bar(
                        grid_summary_scaled,
                        x=HOUSEHOLD_COL,
                        y=["Original Grid Use(Wh)", "Grid Use after 20% Shift", "Grid Use after 30% Shift"],
                        barmode="group",
                        title="Grid Use (Original vs Shifted)",
                        labels={"value": "Grid Use (Wh)", HOUSEHOLD_COL: "Household"}
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                    st.markdown("### Savings percentage by household")
                    fig2 = px.line(
                        grid_summary_scaled,
                        x=HOUSEHOLD_COL,
                        y=["Savings_20% (%)", "Savings_30% (%)"],
                        markers=True,
                        title="Savings (%) Under Demand Response",
                        labels={"value": "Savings (%)", HOUSEHOLD_COL: "Household"}
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                with col2:
                    st.markdown("### Key metrics")
                    overall_original = grid_summary_scaled["Original Grid Use(Wh)"].sum()
                    overall_s20 = grid_summary_scaled["Savings_20% (Wh)"].sum()
                    overall_s30 = grid_summary_scaled["Savings_30% (Wh)"].sum()

                    st.metric("Total Original Grid Use", f"{overall_original:.2f} Wh")
                    st.metric("Total Savings (20%)", f"{overall_s20:.2f} Wh")
                    st.metric("Total Savings (30%)", f"{overall_s30:.2f} Wh")

                    st.markdown("### Results table")
                    st.dataframe(grid_summary_scaled, use_container_width=True)

                    st.download_button(
                        label="Download results as CSV",
                        data=grid_summary_scaled.to_csv(index=False).encode("utf-8"),
                        file_name="energy_analysis_results.csv",
                        mime="text/csv",
                    )

                if demand_response:
                    st.markdown("### Demand response detail (peak/off-peak)")
                    st.caption("Peak hours are defined as the top 25% hours by average consumption (75th percentile threshold).")
                    df_scaled_show = df_scaled[[DATE_COL, HOUSEHOLD_COL, CONS_COL, GRID_COL, "hour", "demand_period"]].head(200)
                    st.dataframe(df_scaled_show, use_container_width=True)
