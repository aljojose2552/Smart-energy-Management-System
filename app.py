import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from io import BytesIO
import plotly.express as px

# App config

st.set_page_config(layout="wide", page_title="Smart Energy: Clustering + Simulation + Solar Patterns")
st.title(" Smart Energy Management: Clustering + Demand Response + Solar Patterns")

DATE_COL = "date"
HOUSEHOLD_COL = "household"
CONS_COL = "Consumption(Wh)"
PROD_COL = "Production(Wh)"
GRID_COL = "From grid(Wh)"

EVENING_START = 17
EVENING_END = 23

MODEL_DIR = Path("household_clustering_model")
KMEANS_PATH = MODEL_DIR / "kmeans_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.pkl"



CLUSTER_LABELS = {
    0: "Solar-Efficient Low-Demand Households",
    1: "Evening-Peak Grid-Dependent Households",
    2: "High-Consumption Solar-Prosumers"
}


# Helpers

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
    Solar-aware feature engineering.
    IMPORTANT: must match your notebook training.
    Required cols for FULL solar model: date, Consumption(Wh), Production(Wh), From grid(Wh)
    household optional (will be set to NEW_HOUSEHOLD if missing).
    """
    df = strip_cols(df)

    if HOUSEHOLD_COL not in df.columns:
        df[HOUSEHOLD_COL] = "NEW_HOUSEHOLD"

    # required baseline
    missing_base = require_columns(df, [DATE_COL, CONS_COL])
    if missing_base:
        raise ValueError(f"Missing required columns: {missing_base}")

    df = ensure_datetime(df, DATE_COL)
    df = ensure_numeric_nonneg(df, CONS_COL)

    # time features
    df["hour"] = df[DATE_COL].dt.hour
    df["weekday"] = df[DATE_COL].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # --- Consumption behavioural features
    feats = df.groupby(HOUSEHOLD_COL)[CONS_COL].agg(
        total_Wh="sum",
        mean_Wh="mean",
        peak_Wh="max",
        std_Wh="std"
    ).reset_index()
    feats["std_Wh"] = feats["std_Wh"].fillna(0)

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

    # --- Solar production pattern + utilisation (only if columns exist)
    if PROD_COL in df.columns:
        df = ensure_numeric_nonneg(df, PROD_COL)

        prod = df.groupby(HOUSEHOLD_COL)[PROD_COL].agg(
            prod_total_Wh="sum",
            prod_mean_Wh="mean",
            prod_peak_Wh="max",
            prod_std_Wh="std",
        ).reset_index()
        prod["prod_std_Wh"] = prod["prod_std_Wh"].fillna(0)

        midday = df[(df["hour"] >= 11) & (df["hour"] <= 15)] \
            .groupby(HOUSEHOLD_COL)[PROD_COL].sum().reset_index(name="midday_prod_Wh")

        prod = prod.merge(midday, on=HOUSEHOLD_COL, how="left")
        prod["midday_prod_Wh"] = prod["midday_prod_Wh"].fillna(0)
        prod["midday_prod_ratio"] = prod["midday_prod_Wh"] / (prod["prod_total_Wh"] + 1e-9)

        feats = feats.merge(prod, on=HOUSEHOLD_COL, how="left")

    if (PROD_COL in df.columns) and (GRID_COL in df.columns):
        df = ensure_numeric_nonneg(df, GRID_COL)

        df["solar_used_Wh"] = np.minimum(df[CONS_COL], df[PROD_COL])
        solar = df.groupby(HOUSEHOLD_COL).agg(
            total_consumption=(CONS_COL, "sum"),
            total_production=(PROD_COL, "sum"),
            grid_import=(GRID_COL, "sum"),
            solar_used=("solar_used_Wh", "sum")
        ).reset_index()

        solar["solar_self_consumption_ratio"] = solar["solar_used"] / (solar["total_production"] + 1e-9)
        solar["grid_dependency_ratio"] = solar["grid_import"] / (solar["total_consumption"] + 1e-9)

        feats = feats.merge(
            solar[[HOUSEHOLD_COL, "solar_self_consumption_ratio", "grid_dependency_ratio"]],
            on=HOUSEHOLD_COL,
            how="left"
        )

    # Fill any missing engineered cols with 0 (safe)
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    return feats

def attach_cluster_to_timeseries(df_raw: pd.DataFrame, feats_with_cluster: pd.DataFrame) -> pd.DataFrame:
    """Join predicted cluster back to row-level data for per-cluster pattern plots."""
    df = strip_cols(df_raw)
    if HOUSEHOLD_COL not in df.columns:
        df[HOUSEHOLD_COL] = "NEW_HOUSEHOLD"
    df = ensure_datetime(df, DATE_COL)

    keep = feats_with_cluster[[HOUSEHOLD_COL, "cluster", "cluster_label"]].copy()
    out = df.merge(keep, on=HOUSEHOLD_COL, how="left")
    out["hour"] = out[DATE_COL].dt.hour
    out["month"] = out[DATE_COL].dt.to_period("M").dt.to_timestamp()
    return out

@st.cache_data(show_spinner=False)
def run_simulation(file_bytes: bytes, efficiency_measures: bool, demand_response: bool):
    """Demand-response simulation using uploaded CSV bytes."""
    if not file_bytes:
        return None, None, "Uploaded file is empty (0 bytes)."

    df = pd.read_csv(BytesIO(file_bytes))
    df = strip_cols(df)

    required = [DATE_COL, HOUSEHOLD_COL, CONS_COL, GRID_COL]
    missing = require_columns(df, required)
    if missing:
        return None, None, f"Missing columns: {missing}"

    df = ensure_datetime(df, DATE_COL)
    df = ensure_numeric_nonneg(df, CONS_COL)
    df = ensure_numeric_nonneg(df, GRID_COL)
    df["hour"] = df[DATE_COL].dt.hour

    df_scaled = df.copy()

    # Efficiency: reduce grid import by 10%
    if efficiency_measures:
        df_scaled[GRID_COL] = df_scaled[GRID_COL] * 0.9

    if demand_response:
        hourly_avg = df_scaled.groupby("hour")[CONS_COL].mean()
        threshold = hourly_avg.quantile(0.75)
        peak_hours = hourly_avg[hourly_avg >= threshold].index.tolist()

        df_scaled["demand_period"] = np.where(df_scaled["hour"].isin(peak_hours), "Peak", "Off-peak")

        is_peak = (df_scaled["demand_period"] == "Peak")
        df_scaled["Adjusted_20%"] = df_scaled[GRID_COL] * np.where(is_peak, 0.8, 1.0)
        df_scaled["Adjusted_30%"] = df_scaled[GRID_COL] * np.where(is_peak, 0.7, 1.0)
    else:
        df_scaled["demand_period"] = "N/A"
        df_scaled["Adjusted_20%"] = df_scaled[GRID_COL]
        df_scaled["Adjusted_30%"] = df_scaled[GRID_COL]

    grid_summary = df_scaled.groupby(HOUSEHOLD_COL).agg(
        **{
            "Original Grid Use(Wh)": (GRID_COL, "sum"),
            "Grid Use after 20% Shift": ("Adjusted_20%", "sum"),
            "Grid Use after 30% Shift": ("Adjusted_30%", "sum"),
        }
    ).reset_index()

    grid_summary["Savings_20% (Wh)"] = grid_summary["Original Grid Use(Wh)"] - grid_summary["Grid Use after 20% Shift"]
    grid_summary["Savings_30% (Wh)"] = grid_summary["Original Grid Use(Wh)"] - grid_summary["Grid Use after 30% Shift"]
    grid_summary["Savings_20% (%)"] = (grid_summary["Savings_20% (Wh)"] / (grid_summary["Original Grid Use(Wh)"] + 1e-9)) * 100
    grid_summary["Savings_30% (%)"] = (grid_summary["Savings_30% (Wh)"] / (grid_summary["Original Grid Use(Wh)"] + 1e-9)) * 100

    return df_scaled, grid_summary.round(2), None



# Session state for solar pattern tab

if "cluster_raw_ts" not in st.session_state:
    st.session_state.cluster_raw_ts = None
if "cluster_feats" not in st.session_state:
    st.session_state.cluster_feats = None



# Tabs

tab1, tab2, tab3 = st.tabs([" Cluster Assignment", " Demand Response Simulation", " Solar Patterns (by Cluster)"])


# TAB 1: Cluster assignment

with tab1:
    st.subheader("Upload household CSV(s) → assign cluster(s)")
    st.caption(
        "Recommended (solar-aware model): `date`, `household`, `Consumption(Wh)`, `Production(Wh)`, `From grid(Wh)`.\n"
        "If `household` is missing, it will be treated as one household."
    )

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

    with colA:
        uploaded_cluster = st.file_uploader("Upload CSV for clustering", type="csv", key="cluster_upload")
        if uploaded_cluster is None:
            st.info("Upload a CSV to classify households.")
        else:
            file_bytes = uploaded_cluster.getvalue()
            if not file_bytes:
                st.error("Uploaded file is empty (0 bytes).")
            elif kmeans is None:
                st.error("Model artifacts are missing — cannot assign clusters yet.")
            else:
                df_new = pd.read_csv(BytesIO(file_bytes))
                df_new = strip_cols(df_new)

                st.markdown("#### Diagnostic: detected columns")
                st.write(df_new.columns.tolist())

                
                try:
                    feats = build_features_for_clustering(df_new)
                except Exception as e:
                    st.error(f"Feature engineering failed: {e}")
                    st.stop()

                
                for c in feature_cols:
                    if c not in feats.columns:
                        feats[c] = 0

                X_new = feats[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0)
                X_new_scaled = scaler.transform(X_new)
                preds = kmeans.predict(X_new_scaled)

                feats["cluster"] = preds
                feats["cluster_label"] = feats["cluster"].map(CLUSTER_LABELS).fillna(feats["cluster"].astype(str))

                st.markdown("### Cluster assignment result")
                st.dataframe(feats, use_container_width=True)

                
                try:
                    st.session_state.cluster_feats = feats.copy()
                    ts = attach_cluster_to_timeseries(df_new, feats)
                    
                    if CONS_COL in ts.columns:
                        ts = ensure_numeric_nonneg(ts, CONS_COL)
                    if PROD_COL in ts.columns:
                        ts = ensure_numeric_nonneg(ts, PROD_COL)
                    if GRID_COL in ts.columns:
                        ts = ensure_numeric_nonneg(ts, GRID_COL)
                    st.session_state.cluster_raw_ts = ts
                    st.success("Saved clustered time-series into session (Solar Patterns tab is now available).")
                except Exception as e:
                    st.warning(f"Could not prepare Solar Patterns tab data: {e}")

                
                try:
                    ts_plot = st.session_state.cluster_raw_ts
                    hourly_cons = ts_plot.groupby("hour")[CONS_COL].mean().reset_index()
                    fig = px.line(hourly_cons, x="hour", y=CONS_COL, markers=True,
                                  title="Uploaded data: average hourly consumption (overall)")
                    fig.update_xaxes(dtick=1)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass



# TAB 2: Simulation

with tab2:
    st.subheader("Upload grid-import data → simulate demand response savings")
    st.caption("Required: `date`, `household`, `Consumption(Wh)`, `From grid(Wh)`.")

    with st.sidebar:
        st.header("Scenario controls (Simulation tab)")
        efficiency_measures = st.checkbox("Apply Energy Efficiency Measures (10% grid reduction)", value=False, key="eff")
        demand_response = st.checkbox("Enable Demand Response (peak shifting)", value=False, key="dr")
        st.divider()

    uploaded_sim = st.file_uploader("Upload CSV (for simulation)", type="csv", key="sim_upload")
    if uploaded_sim is None:
        st.info("Upload a simulation CSV to run demand-response analysis.")
    else:
        sim_bytes = uploaded_sim.getvalue()
        if not sim_bytes:
            st.error("Uploaded file is empty (0 bytes). Please re-export and upload again.")
            st.stop()

        df_preview = pd.read_csv(BytesIO(sim_bytes))
        df_preview = strip_cols(df_preview)

        st.markdown("#### Diagnostic: detected columns")
        st.write(df_preview.columns.tolist())
        st.dataframe(df_preview.head(50), use_container_width=True)

        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                df_scaled, grid_summary_scaled, err = run_simulation(sim_bytes, efficiency_measures, demand_response)

            if err:
                st.error(err)
            else:
                st.success("Simulation completed.")

                col1, col2 = st.columns([2, 1])

                with col1:
                    fig1 = px.bar(
                        grid_summary_scaled,
                        x=HOUSEHOLD_COL,
                        y=["Original Grid Use(Wh)", "Grid Use after 20% Shift", "Grid Use after 30% Shift"],
                        barmode="group",
                        title="Grid Use (Original vs Shifted)",
                        labels={"value": "Grid Use (Wh)", HOUSEHOLD_COL: "Household"}
                    )
                    st.plotly_chart(fig1, use_container_width=True)

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
                    overall_original = grid_summary_scaled["Original Grid Use(Wh)"].sum()
                    overall_s20 = grid_summary_scaled["Savings_20% (Wh)"].sum()
                    overall_s30 = grid_summary_scaled["Savings_30% (Wh)"].sum()

                    st.metric("Total Original Grid Use", f"{overall_original:.2f} Wh")
                    st.metric("Total Savings (20%)", f"{overall_s20:.2f} Wh")
                    st.metric("Total Savings (30%)", f"{overall_s30:.2f} Wh")

                    st.dataframe(grid_summary_scaled, use_container_width=True)

                    st.download_button(
                        label="Download results as CSV",
                        data=grid_summary_scaled.to_csv(index=False).encode("utf-8"),
                        file_name="energy_analysis_results.csv",
                        mime="text/csv",
                    )

                if demand_response:
                    st.markdown("### Demand response detail (preview)")
                    st.dataframe(df_scaled[[DATE_COL, HOUSEHOLD_COL, CONS_COL, GRID_COL, "hour", "demand_period"]].head(200),
                                 use_container_width=True)



# TAB 3: Solar patterns per cluster

with tab3:
    st.subheader("Solar patterns by predicted cluster")
    st.caption("First run Cluster Assignment tab (upload a CSV) to populate this dashboard.")

    ts = st.session_state.cluster_raw_ts
    feats = st.session_state.cluster_feats

    if ts is None or feats is None:
        st.info("No clustered data in session yet. Go to **Cluster Assignment** tab, upload a CSV, and classify it.")
    else:
        # Controls
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            metric = st.selectbox("Metric", [PROD_COL, CONS_COL, GRID_COL], index=0)
        with c2:
            agg = st.selectbox("Aggregation", ["mean", "sum"], index=0)
        with c3:
            view = st.selectbox("View", ["Hourly (diurnal)", "Monthly (seasonality)"], index=0)

        
        if metric not in ts.columns:
            st.error(f"Metric `{metric}` not found in uploaded data. Available columns: {ts.columns.tolist()}")
            st.stop()

        
        st.markdown("### Cluster feature profiles (uploaded batch)")
        show_profile = feats.groupby(["cluster", "cluster_label"]).mean(numeric_only=True).round(3).reset_index()
        st.dataframe(show_profile, use_container_width=True)

        # Pattern plots
        if view.startswith("Hourly"):
            group_cols = ["cluster_label", "hour"]
        else:
            group_cols = ["cluster_label", "month"]

        if agg == "mean":
            pat = ts.groupby(group_cols)[metric].mean().reset_index()
        else:
            pat = ts.groupby(group_cols)[metric].sum().reset_index()

        title = f"{agg.upper()} {metric} by Cluster ({'Hourly' if 'hour' in group_cols else 'Monthly'})"

        if "hour" in group_cols:
            fig = px.line(pat, x="hour", y=metric, color="cluster_label", markers=True, title=title)
            fig.update_xaxes(dtick=1)
        else:
            fig = px.line(pat, x="month", y=metric, color="cluster_label", markers=True, title=title)

        st.plotly_chart(fig, use_container_width=True)

        
        if (PROD_COL in ts.columns) and (CONS_COL in ts.columns):
            st.markdown("### Solar self-consumption pattern (per cluster)")
            tmp = ts.copy()
            tmp[PROD_COL] = pd.to_numeric(tmp[PROD_COL], errors="coerce").fillna(0).clip(lower=0)
            tmp[CONS_COL] = pd.to_numeric(tmp[CONS_COL], errors="coerce").fillna(0).clip(lower=0)
            tmp["solar_used_Wh"] = np.minimum(tmp[CONS_COL], tmp[PROD_COL])
            tmp["self_consumption_inst"] = tmp["solar_used_Wh"] / (tmp[PROD_COL] + 1e-9)

            sc = tmp.groupby(["cluster_label", "hour"])["self_consumption_inst"].mean().reset_index()
            fig_sc = px.line(sc, x="hour", y="self_consumption_inst", color="cluster_label", markers=True,
                             title="Average self-consumption ratio by hour (per cluster)")
            fig_sc.update_xaxes(dtick=1)
            st.plotly_chart(fig_sc, use_container_width=True)

        # Download clustered time-series
        st.download_button(
            "Download clustered time-series (rows tagged with cluster)",
            data=ts.to_csv(index=False).encode("utf-8"),
            file_name="clustered_timeseries.csv",
            mime="text/csv",
        )
