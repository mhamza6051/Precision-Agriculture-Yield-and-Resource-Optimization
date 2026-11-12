import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from pathlib import Path
from itertools import product

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HQCA Precision Agriculture Optimizer",
    page_icon="🌱",
    layout="wide"
)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

FINAL_DATASET_PATH = DATA_DIR / "final_model_dataset.csv"
FINAL_RECOMMENDATION_PATH = DATA_DIR / "final_recommendation_summary.csv"
CLASSICAL_RESULTS_PATH = DATA_DIR / "classical_optimization_results.csv"
QUBO_SUMMARY_PATH = DATA_DIR / "qubo_summary.csv"
EXACT_SUMMARY_PATH = DATA_DIR / "exact_solver_summary.csv"

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #1b5e20;
    margin-bottom: 0.15rem;
}
.sub-title {
    font-size: 1.05rem;
    color: #4e5d6c;
    margin-bottom: 1.4rem;
}
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #17324d;
    margin-top: 0.8rem;
    margin-bottom: 0.8rem;
}
.info-box {
    background-color: #f8fbf8;
    border: 1px solid #dfeee0;
    border-radius: 14px;
    padding: 0.9rem 1rem;
}
.small-note {
    font-size: 0.9rem;
    color: #5f6b76;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD ARTIFACTS
# =========================================================
@st.cache_resource
def load_artifacts():
    # Yield model
    yield_xgb_path = MODELS_DIR / "yield_xgboost.pkl"
    yield_rf_path = MODELS_DIR / "yield_random_forest.pkl"

    if yield_xgb_path.exists():
        yield_model = joblib.load(yield_xgb_path)
        yield_model_name = "XGBoost"
    else:
        yield_model = joblib.load(yield_rf_path)
        yield_model_name = "Random Forest"

    water_model = joblib.load(MODELS_DIR / "water_model.pkl")
    fert_model = joblib.load(MODELS_DIR / "fertilizer_model.pkl")
    energy_model = joblib.load(MODELS_DIR / "energy_model_v2.pkl")

    yield_scaler = joblib.load(MODELS_DIR / "yield_scaler.pkl")
    resource_scaler = joblib.load(MODELS_DIR / "resource_scaler.pkl")
    energy_scaler = joblib.load(MODELS_DIR / "energy_scaler.pkl")

    with open(MODELS_DIR / "yield_feature_cols.json", "r") as f:
        yield_cols = json.load(f)

    with open(MODELS_DIR / "resource_feature_cols.json", "r") as f:
        resource_cols = json.load(f)

    with open(MODELS_DIR / "energy_feature_cols.json", "r") as f:
        energy_cols = json.load(f)

    return {
        "yield_model": yield_model,
        "yield_model_name": yield_model_name,
        "water_model": water_model,
        "fert_model": fert_model,
        "energy_model": energy_model,
        "yield_scaler": yield_scaler,
        "resource_scaler": resource_scaler,
        "energy_scaler": energy_scaler,
        "yield_cols": yield_cols,
        "resource_cols": resource_cols,
        "energy_cols": energy_cols,
    }


@st.cache_data
def load_saved_results():
    data = {}

    if FINAL_DATASET_PATH.exists():
        data["final_dataset"] = pd.read_csv(FINAL_DATASET_PATH)
    else:
        data["final_dataset"] = pd.DataFrame()

    if FINAL_RECOMMENDATION_PATH.exists():
        data["final_recommendation"] = pd.read_csv(FINAL_RECOMMENDATION_PATH)
    else:
        data["final_recommendation"] = pd.DataFrame()

    if CLASSICAL_RESULTS_PATH.exists():
        data["classical_results"] = pd.read_csv(CLASSICAL_RESULTS_PATH)
    else:
        data["classical_results"] = pd.DataFrame()

    if QUBO_SUMMARY_PATH.exists():
        data["qubo_summary"] = pd.read_csv(QUBO_SUMMARY_PATH)
    else:
        data["qubo_summary"] = pd.DataFrame()

    if EXACT_SUMMARY_PATH.exists():
        data["exact_summary"] = pd.read_csv(EXACT_SUMMARY_PATH)
    else:
        data["exact_summary"] = pd.DataFrame()

    return data


artifacts = load_artifacts()
saved = load_saved_results()

# =========================================================
# HELPERS
# =========================================================
def build_features(input_dict: dict) -> pd.DataFrame:
    """
    Build features exactly the same way as training.
    """
    df = pd.DataFrame([input_dict])

    # Match notebook feature engineering exactly
    df["temp_stress"] = df["temperature"] * df["humidity"]
    df["water_demand"] = df["evapotranspiration"] * df["temperature"]
    df["soil_quality"] = (df["organic_carbon"] + df["soil_ph"]) / 2.0

    return df


def align_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = 0
    out = out[cols]
    return out


def predict_all(input_dict: dict) -> dict:
    df = build_features(input_dict)

    # Yield prediction
    X_yield = align_features(df, artifacts["yield_cols"])
    X_yield_scaled = artifacts["yield_scaler"].transform(X_yield)
    pred_yield = float(artifacts["yield_model"].predict(X_yield_scaled)[0])

    # Resource prediction
    X_res = align_features(df, artifacts["resource_cols"])
    X_res_scaled = artifacts["resource_scaler"].transform(X_res)

    pred_water = float(artifacts["water_model"].predict(X_res_scaled)[0])
    pred_fert = float(artifacts["fert_model"].predict(X_res_scaled)[0])

    # Energy prediction with predicted water used
    df_energy = df.copy()
    df_energy["pred_water_used"] = pred_water
    X_energy = align_features(df_energy, artifacts["energy_cols"])
    X_energy_scaled = artifacts["energy_scaler"].transform(X_energy)

    pred_energy = float(artifacts["energy_model"].predict(X_energy_scaled)[0])

    return {
        "yield_suitability": pred_yield,
        "pred_water": pred_water,
        "pred_fertilizer": pred_fert,
        "pred_energy": pred_energy,
        "feature_frame": df
    }


def classify_vigor(ndvi: float) -> str:
    if ndvi >= 0.75:
        return "High crop vigor"
    if ndvi >= 0.55:
        return "Moderate crop vigor"
    return "Low crop vigor"


def classify_moisture(sm: float) -> str:
    if sm < 0.20:
        return "Low soil moisture"
    if sm < 0.35:
        return "Moderate soil moisture"
    return "Good soil moisture"


def classify_soil_health(soil_quality: float) -> str:
    if soil_quality >= 4.8:
        return "Strong soil health"
    if soil_quality >= 4.0:
        return "Moderate soil health"
    return "Weak soil health"


def drought_risk_score(temp: float, rainfall: float, soil_moisture: float) -> float:
    temp_component = min(max((temp - 20) / 20, 0), 1)
    rain_component = min(max((5 - rainfall) / 5, 0), 1)
    moisture_component = min(max((0.35 - soil_moisture) / 0.35, 0), 1)
    return float((temp_component + rain_component + moisture_component) / 3)


def climate_stress_score(temp_stress: float) -> float:
    # soft normalization for display
    score = temp_stress / 3000.0
    return float(min(max(score, 0), 1))


def resource_efficiency_index(yield_outlook: float, water: float, fert: float, energy: float) -> float:
    denom = (water / 400.0) + (fert / 200.0) + (energy / 50.0)
    if denom <= 0:
        return 0.0
    return float(yield_outlook / denom)


def generate_advice(row: dict) -> list:
    advice = []

    if row["drought_risk"] > 0.6:
        advice.append("High drought pressure detected. Prioritize irrigation efficiency and moisture conservation.")
    if row["ndvi"] < 0.45:
        advice.append("Crop vigor is weak. Inspect nutrient balance and crop stress before increasing inputs aggressively.")
    if row["soil_moisture"] < 0.20:
        advice.append("Soil moisture is critically low. Immediate irrigation support is advisable.")
    if row["soil_quality_value"] < 4.0:
        advice.append("Soil health proxy is weak. Organic matter and pH management should be reviewed.")
    if row["pred_energy"] > 30:
        advice.append("Energy demand is relatively high. Consider efficient irrigation scheduling or pump optimization.")
    if len(advice) == 0:
        advice.append("Field conditions are relatively stable. Follow optimized allocation and continue monitoring.")
    return advice


def get_candidate_levels(pred_water: float, pred_fert: float, water_budget: float, fert_budget: float):
    base_water_levels = [220, 250, 280, 310, 340, 370, 400]
    base_fert_levels = [80, 100, 120, 140, 160, 180]

    water_levels = [w for w in base_water_levels if w <= water_budget]
    fert_levels = [f for f in base_fert_levels if f <= fert_budget]

    if not water_levels:
        water_levels = [min(base_water_levels)]
    if not fert_levels:
        fert_levels = [min(base_fert_levels)]

    return water_levels, fert_levels


def evaluate_allocation(base_input: dict, water: float, fertilizer: float,
                        water_budget: float, fert_budget: float, energy_budget: float):
    preds = predict_all(base_input)
    base_df = preds["feature_frame"].copy()

    rec_water = preds["pred_water"]
    rec_fert = preds["pred_fertilizer"]
    base_yield = preds["yield_suitability"]

    # Predicted energy from chosen water
    df_energy = base_df.copy()
    df_energy["pred_water_used"] = water
    X_energy = align_features(df_energy, artifacts["energy_cols"])
    X_energy_scaled = artifacts["energy_scaler"].transform(X_energy)
    pred_energy = float(artifacts["energy_model"].predict(X_energy_scaled)[0])

    # Hard budget infeasibility
    if water > water_budget or fertilizer > fert_budget or pred_energy > energy_budget:
        return {
            "water": water,
            "fertilizer": fertilizer,
            "pred_energy": pred_energy,
            "base_yield": base_yield,
            "adjusted_yield": 0.0,
            "recommended_water": rec_water,
            "recommended_fertilizer": rec_fert,
            "allocation_fit": 0.0,
            "objective_score": -999
        }

    # Smooth allocation-fit around model recommendation
    sigma_w = max(20.0, 0.10 * rec_water)
    sigma_f = max(10.0, 0.10 * rec_fert)

    fit_w = np.exp(-((water - rec_water) ** 2) / (2 * sigma_w ** 2))
    fit_f = np.exp(-((fertilizer - rec_fert) ** 2) / (2 * sigma_f ** 2))
    allocation_fit = float((fit_w + fit_f) / 2.0)

    # Allocation-adjusted yield outlook
    adjusted_yield = float(base_yield * (0.75 + 0.25 * allocation_fit))

    # Objective: higher is better
    score = (
        0.65 * adjusted_yield
        + 0.15 * allocation_fit
        - 0.10 * (water / max(water_budget, 1))
        - 0.05 * (fertilizer / max(fert_budget, 1))
        - 0.05 * (pred_energy / max(energy_budget, 1))
    )

    return {
        "water": water,
        "fertilizer": fertilizer,
        "pred_energy": pred_energy,
        "base_yield": base_yield,
        "adjusted_yield": adjusted_yield,
        "recommended_water": rec_water,
        "recommended_fertilizer": rec_fert,
        "allocation_fit": allocation_fit,
        "objective_score": float(score)
    }


def optimize_allocation(base_input: dict, water_budget: float, fert_budget: float, energy_budget: float):
    initial_preds = predict_all(base_input)
    water_levels, fert_levels = get_candidate_levels(
        initial_preds["pred_water"], initial_preds["pred_fertilizer"], water_budget, fert_budget
    )

    rows = []
    for w in water_levels:
        for f in fert_levels:
            rows.append(
                evaluate_allocation(base_input, w, f, water_budget, fert_budget, energy_budget)
            )

    results_df = pd.DataFrame(rows).sort_values("objective_score", ascending=False).reset_index(drop=True)
    best_solution = results_df.iloc[0].to_dict()

    return results_df, best_solution, water_levels, fert_levels


def qubo_bruteforce_validation(results_df: pd.DataFrame, water_levels: list, fert_levels: list):
    score_lookup = {}
    for _, row in results_df.iterrows():
        score_lookup[(int(row["water"]), int(row["fertilizer"]))] = row["objective_score"]

    num_water = len(water_levels)
    num_fert = len(fert_levels)

    def qubo_objective(binary_vector, penalty=5.0):
        xw = binary_vector[:num_water]
        xf = binary_vector[num_water:num_water + num_fert]

        water_constraint = (np.sum(xw) - 1) ** 2
        fert_constraint = (np.sum(xf) - 1) ** 2

        selected_w = np.where(xw == 1)[0]
        selected_f = np.where(xf == 1)[0]

        if len(selected_w) == 1 and len(selected_f) == 1:
            water = water_levels[selected_w[0]]
            fert = fert_levels[selected_f[0]]
            score = score_lookup.get((water, fert), -999)
        else:
            score = -999

        energy = -score + penalty * (water_constraint + fert_constraint)
        return float(energy)

    best_binary = None
    best_energy = float("inf")

    for bits in product([0, 1], repeat=num_water + num_fert):
        bits = np.array(bits)
        energy = qubo_objective(bits)
        if energy < best_energy:
            best_energy = energy
            best_binary = bits.copy()

    xw_best = best_binary[:num_water]
    xf_best = best_binary[num_water:num_water + num_fert]

    best_w_idx = np.where(xw_best == 1)[0][0]
    best_f_idx = np.where(xf_best == 1)[0][0]

    return {
        "binary_vector": best_binary,
        "qubo_energy": best_energy,
        "water": water_levels[best_w_idx],
        "fertilizer": fert_levels[best_f_idx],
        "water_table": pd.DataFrame({"water_level": water_levels, "selected": xw_best}),
        "fert_table": pd.DataFrame({"fertilizer_level": fert_levels, "selected": xf_best}),
    }


# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">Hybrid Quantum–Classical Precision Agriculture Optimizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Live field input, machine-learning prediction, resource optimization under budget constraints, and QUBO-style binary validation.</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Live Recommender",
        "Dashboard",
        "Dataset Explorer",
        "Optimization Archive",
        "Quantum Validation"
    ]
)

# =========================================================
# LIVE RECOMMENDER PAGE
# =========================================================
if page == "Live Recommender":
    st.markdown('<div class="section-title">New Field Recommendation</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box">Enter field and climate conditions below. The system predicts yield suitability, resource demand, and optimized allocations under limited budgets.</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    with st.sidebar:
        st.markdown("### Field Inputs")

        latitude = st.number_input("Latitude", value=30.50, format="%.4f")
        longitude = st.number_input("Longitude", value=72.20, format="%.4f")
        field_id = st.number_input("Field ID", min_value=1, value=50, step=1)

        temperature = st.slider("Temperature (°C)", 5.0, 50.0, 28.0, 0.1)
        humidity = st.slider("Humidity (%)", 30.0, 90.0, 60.0, 0.1)
        rainfall = st.slider("Rainfall", 0.0, 20.0, 2.5, 0.1)
        solar_radiation = st.slider("Solar Radiation", 10.0, 30.0, 20.0, 0.1)
        wind_speed = st.slider("Wind Speed", 0.5, 8.0, 3.0, 0.1)
        evapotranspiration = st.slider("Evapotranspiration", 2.0, 8.0, 5.0, 0.1)
        biomass = st.slider("Biomass", 1000.0, 5000.0, 2500.0, 1.0)
        water_productivity = st.slider("Water Productivity", 0.5, 2.5, 1.5, 0.01)

        soil_moisture = st.slider("Soil Moisture", 0.10, 0.50, 0.30, 0.01)
        ndvi = st.slider("NDVI", 0.20, 0.90, 0.60, 0.01)
        land_surface_temp = st.slider("Land Surface Temp", 13.0, 50.0, 30.0, 0.1)

        soil_ph = st.slider("Soil pH", 5.5, 8.5, 7.0, 0.01)
        organic_carbon = st.slider("Organic Carbon", 0.5, 3.5, 2.0, 0.01)
        sand_content = st.slider("Sand Content", 20.0, 70.0, 40.0, 0.1)
        clay_content = st.slider("Clay Content", 10.0, 50.0, 30.0, 0.1)

        st.markdown("### Resource Budgets")
        water_budget = st.slider("Max Water Budget", 220, 450, 400, 10)
        fert_budget = st.slider("Max Fertilizer Budget", 80, 200, 180, 10)
        energy_budget = st.slider("Max Energy Budget", 10, 50, 40, 1)

        run_btn = st.button("Run Prediction & Optimization", use_container_width=True)

    if run_btn:
        input_dict = {
            "latitude": latitude,
            "longitude": longitude,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "solar_radiation": solar_radiation,
            "wind_speed": wind_speed,
            "evapotranspiration": evapotranspiration,
            "biomass": biomass,
            "water_productivity": water_productivity,
            "soil_moisture": soil_moisture,
            "ndvi": ndvi,
            "land_surface_temp": land_surface_temp,
            "field_id": field_id,
            "soil_ph": soil_ph,
            "organic_carbon": organic_carbon,
            "sand_content": sand_content,
            "clay_content": clay_content
        }

        preds = predict_all(input_dict)
        results_df, best_solution, water_levels, fert_levels = optimize_allocation(
            input_dict, water_budget, fert_budget, energy_budget
        )
        qubo_live = qubo_bruteforce_validation(results_df, water_levels, fert_levels)

        df_features = build_features(input_dict)
        soil_quality_value = float(df_features["soil_quality"].iloc[0])
        drought_risk = drought_risk_score(temperature, rainfall, soil_moisture)
        climate_stress = climate_stress_score(float(df_features["temp_stress"].iloc[0]))
        vigor_label = classify_vigor(ndvi)
        moisture_label = classify_moisture(soil_moisture)
        soil_label = classify_soil_health(soil_quality_value)

        adjusted_yield = float(best_solution["adjusted_yield"])
        efficiency_index = resource_efficiency_index(
            adjusted_yield,
            float(best_solution["water"]),
            float(best_solution["fertilizer"]),
            float(best_solution["pred_energy"])
        )

        insight_row = {
            "drought_risk": drought_risk,
            "ndvi": ndvi,
            "soil_moisture": soil_moisture,
            "soil_quality_value": soil_quality_value,
            "pred_energy": float(best_solution["pred_energy"])
        }
        advice_notes = generate_advice(insight_row)

        st.markdown('<div class="section-title">Primary Recommendation</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Optimized Water", f"{best_solution['water']:.0f}")
        c2.metric("Optimized Fertilizer", f"{best_solution['fertilizer']:.0f}")
        c3.metric("Predicted Energy", f"{best_solution['pred_energy']:.2f}")
        c4.metric("Objective Score", f"{best_solution['objective_score']:.4f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Yield Suitability", f"{preds['yield_suitability']:.4f}")
        c6.metric("Allocation-Adjusted Yield Outlook", f"{adjusted_yield:.4f}")
        c7.metric("Resource Efficiency Index", f"{efficiency_index:.4f}")
        c8.metric("Allocation Fit", f"{best_solution['allocation_fit']:.4f}")

        st.markdown('<div class="section-title">Agronomic Interpretation</div>', unsafe_allow_html=True)
        c9, c10, c11, c12 = st.columns(4)
        c9.metric("Crop Vigor", vigor_label)
        c10.metric("Soil Moisture Status", moisture_label)
        c11.metric("Soil Health Proxy", soil_label)
        c12.metric("Model Used", artifacts["yield_model_name"])

        c13, c14 = st.columns(2)
        c13.metric("Drought Risk Score", f"{drought_risk:.3f}")
        c14.metric("Climate Stress Score", f"{climate_stress:.3f}")

        st.markdown('<div class="section-title">Model-Based Base Demand</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        b1.metric("Base Water Demand", f"{preds['pred_water']:.2f}")
        b2.metric("Base Fertilizer Demand", f"{preds['pred_fertilizer']:.2f}")
        b3.metric("Base Energy Demand", f"{preds['pred_energy']:.2f}")

        st.markdown('<div class="section-title">Budget Utilization</div>', unsafe_allow_html=True)
        u1, u2, u3 = st.columns(3)
        u1.metric("Water Budget Used", f"{(best_solution['water']/water_budget)*100:.1f}%")
        u2.metric("Fertilizer Budget Used", f"{(best_solution['fertilizer']/fert_budget)*100:.1f}%")
        u3.metric("Energy Budget Used", f"{(best_solution['pred_energy']/energy_budget)*100:.1f}%")

        st.markdown('<div class="section-title">Recommended Action Notes</div>', unsafe_allow_html=True)
        for note in advice_notes:
            st.write(f"- {note}")

        st.markdown('<div class="section-title">Top Allocation Candidates</div>', unsafe_allow_html=True)
        st.dataframe(results_df.head(10), use_container_width=True)

        st.markdown('<div class="section-title">Optimization Heatmap</div>', unsafe_allow_html=True)
        pivot = results_df.pivot(index="fertilizer", columns="water", values="objective_score")
        fig_heat = px.imshow(
            pivot,
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale="YlGnBu",
            labels=dict(x="Water Allocation", y="Fertilizer Allocation", color="Objective Score")
        )
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown('<div class="section-title">QUBO Live Validation</div>', unsafe_allow_html=True)
        v1, v2, v3 = st.columns(3)
        v1.metric("Classical Best", f"({int(best_solution['water'])}, {int(best_solution['fertilizer'])})")
        v2.metric("QUBO Best", f"({qubo_live['water']}, {qubo_live['fertilizer']})")
        v3.metric("QUBO Energy", f"{qubo_live['qubo_energy']:.6f}")

        if int(best_solution["water"]) == qubo_live["water"] and int(best_solution["fertilizer"]) == qubo_live["fertilizer"]:
            st.success("Live classical optimization matches live QUBO-style binary validation.")
        else:
            st.error("Mismatch between classical and QUBO-style optimization.")

        q1, q2 = st.columns(2)
        with q1:
            st.write("Water binary selection")
            st.dataframe(qubo_live["water_table"], use_container_width=True)
        with q2:
            st.write("Fertilizer binary selection")
            st.dataframe(qubo_live["fert_table"], use_container_width=True)

        st.markdown('<div class="section-title">Input Summary</div>', unsafe_allow_html=True)
        input_summary_df = pd.DataFrame([input_dict])
        st.dataframe(input_summary_df, use_container_width=True)

        st.markdown(
            '<div class="small-note">Note: Yield outputs are suitability-style model scores, not measured tons per acre. True physical yield forecasting requires retraining with real yield units and crop-specific targets.</div>',
            unsafe_allow_html=True
        )

# =========================================================
# DASHBOARD PAGE
# =========================================================
elif page == "Dashboard":
    st.markdown('<div class="section-title">Saved Project Dashboard</div>', unsafe_allow_html=True)

    final_dataset = saved["final_dataset"]
    final_recommendation = saved["final_recommendation"]
    classical_results = saved["classical_results"]

    if final_dataset.empty or final_recommendation.empty or classical_results.empty:
        st.warning("Saved processed CSV files are missing.")
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Rows in Final Dataset", f"{len(final_dataset):,}")
        d2.metric("Saved Best Water", f"{final_recommendation.loc[0, 'recommended_water']:.2f}")
        d3.metric("Saved Best Fertilizer", f"{final_recommendation.loc[0, 'recommended_fertilizer']:.2f}")
        d4.metric("Saved Predicted Energy", f"{final_recommendation.loc[0, 'predicted_energy']:.2f}")

        st.dataframe(final_recommendation, use_container_width=True)

        pivot = classical_results.pivot(index="fertilizer", columns="water", values="objective_score")
        fig_heat = px.imshow(
            pivot,
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale="YlGnBu",
            labels=dict(x="Water Allocation", y="Fertilizer Allocation", color="Objective Score")
        )
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

# =========================================================
# DATASET EXPLORER
# =========================================================
elif page == "Dataset Explorer":
    st.markdown('<div class="section-title">Dataset Explorer</div>', unsafe_allow_html=True)

    final_dataset = saved["final_dataset"]

    if final_dataset.empty:
        st.warning("Final dataset file not found.")
    else:
        st.dataframe(final_dataset.head(25), use_container_width=True)

        numeric_cols = final_dataset.select_dtypes(include="number").columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            feature_hist = st.selectbox("Select feature for histogram", numeric_cols, index=0)
        with col2:
            feature_y = st.selectbox(
                "Select Y-axis for scatter",
                numeric_cols,
                index=numeric_cols.index("yield_score") if "yield_score" in numeric_cols else 1
            )

        fig_hist = px.histogram(final_dataset, x=feature_hist, nbins=30, title=f"Distribution of {feature_hist}")
        st.plotly_chart(fig_hist, use_container_width=True)

        feature_x = st.selectbox(
            "Select X-axis for scatter",
            numeric_cols,
            index=numeric_cols.index("soil_moisture") if "soil_moisture" in numeric_cols else 0
        )
        fig_scatter = px.scatter(final_dataset, x=feature_x, y=feature_y, opacity=0.6, title=f"{feature_x} vs {feature_y}")
        st.plotly_chart(fig_scatter, use_container_width=True)

        corr = final_dataset[numeric_cols].corr()
        fig_corr = px.imshow(
            corr,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix"
        )
        fig_corr.update_layout(height=700)
        st.plotly_chart(fig_corr, use_container_width=True)

# =========================================================
# OPTIMIZATION ARCHIVE
# =========================================================
elif page == "Optimization Archive":
    st.markdown('<div class="section-title">Saved Classical Optimization Archive</div>', unsafe_allow_html=True)

    classical_results = saved["classical_results"]
    if classical_results.empty:
        st.warning("Saved classical optimization results file not found.")
    else:
        st.dataframe(classical_results.head(20), use_container_width=True)

        fig_line = go.Figure()
        for fert in sorted(classical_results["fertilizer"].unique()):
            subset = classical_results[classical_results["fertilizer"] == fert]
            fig_line.add_trace(
                go.Scatter(
                    x=subset["water"],
                    y=subset["objective_score"],
                    mode="lines+markers",
                    name=f"Fertilizer {int(fert)}"
                )
            )

        fig_line.update_layout(
            title="Objective Score Across Water Levels",
            xaxis_title="Water Allocation",
            yaxis_title="Objective Score",
            height=520
        )
        st.plotly_chart(fig_line, use_container_width=True)

# =========================================================
# QUANTUM VALIDATION
# =========================================================
elif page == "Quantum Validation":
    st.markdown('<div class="section-title">Saved Quantum Validation</div>', unsafe_allow_html=True)

    exact_summary = saved["exact_summary"]
    qubo_summary = saved["qubo_summary"]

    if exact_summary.empty:
        st.warning("Exact solver summary file not found.")
    else:
        comparison = pd.DataFrame({
            "Method": ["Classical", "QUBO Brute Force", "Exact Qiskit Solver"],
            "Water": [
                int(exact_summary.loc[0, "classical_best_water"]),
                int(exact_summary.loc[0, "qubo_best_water"]),
                int(exact_summary.loc[0, "exact_best_water"])
            ],
            "Fertilizer": [
                int(exact_summary.loc[0, "classical_best_fertilizer"]),
                int(exact_summary.loc[0, "qubo_best_fertilizer"]),
                int(exact_summary.loc[0, "exact_best_fertilizer"])
            ]
        })

        st.dataframe(comparison, use_container_width=True)
        st.metric("Exact Solver fval", f"{exact_summary.loc[0, 'exact_fval']:.6f}")

        match = (
            exact_summary.loc[0, "classical_best_water"] == exact_summary.loc[0, "qubo_best_water"] == exact_summary.loc[0, "exact_best_water"]
            and
            exact_summary.loc[0, "classical_best_fertilizer"] == exact_summary.loc[0, "qubo_best_fertilizer"] == exact_summary.loc[0, "exact_best_fertilizer"]
        )

        if match:
            st.success("All saved optimization methods match. Quantum-ready optimization pipeline is validated.")
        else:
            st.error("Mismatch detected in saved optimization summaries.")

        if not qubo_summary.empty:
            st.dataframe(qubo_summary, use_container_width=True)