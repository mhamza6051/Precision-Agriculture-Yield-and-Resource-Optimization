# Precision Agriculture — Crop Yield & Resource Optimization with ML and QUBO

A precision-agriculture decision system that **predicts crop yield**, **recommends water and fertilizer**, and then **optimizes the allocation three different ways — classical grid search, a QUBO formulation solved by brute force, and Qiskit's exact solver** — with all three methods independently arriving at the same optimum. It ships with a Streamlit app, the *HQCA Precision Agriculture Optimizer*.

> **The headline validation:** classical search, QUBO brute force, and the Qiskit exact solver all selected **water = 280, fertilizer = 180** with the identical objective value (**−0.11588**). The optimization pipeline cross-validates itself.

## Results

**Yield prediction** (test set, 3,000 field records)

| Model | RMSE | R² |
|---|---|---|
| **Linear Regression** | **0.0495** | **0.8751** |
| XGBoost | 0.0503 | 0.8713 |
| Random Forest | 0.0511 | 0.8671 |

**Resource recommendation models**

| Target | RMSE | R² |
|---|---|---|
| Fertilizer | 10.73 | **0.836** |
| Water | 20.91 | 0.676 |
| Energy | 5.41 | 0.110 ⚠️ |

> ⚠️ **Honest finding:** the energy model is weak (R² ≈ 0.11). As documented in the notebook, the energy target was generated with weak dependence on the available features, so the model has little signal to learn. It is reported as-is rather than hidden — the water and fertilizer models carry the recommendation system.

Full numbers: [docs/RESULTS.md](docs/RESULTS.md)

## How it works

1. **Data fusion** — six sources merged into one 3,000-row master dataset: WaPOR remote-sensing data, climate, soil moisture, vegetation indices, soil properties (150 fields), and resource targets
2. **Feature engineering** — derived stress/demand/quality indicators (`temp_stress`, `water_demand`, `soil_quality`, …), scaling, train/test split
3. **Prediction** — yield regressors (LR / RF / XGBoost) plus separate water, fertilizer and energy recommendation models
4. **Optimization** — a yield-utility vs. energy-cost objective over discrete water × fertilizer levels, solved by (a) exhaustive classical search, (b) QUBO binary encoding + brute force, and (c) Qiskit `QuadraticProgram` with an exact solver — all three agree
5. **App** — Streamlit dashboard to explore fields, predictions and the validated recommendation

## Run the app

```bash
pip install -r requirements.txt
# one-time: download the 3 large resource models from the v1.0 Release (see models/README.md)
streamlit run app/app.py
```

## Repository structure

```
app/app.py                     Streamlit app (HQCA Precision Agriculture Optimizer)
notebooks/                     Full pipeline: fusion → models → classical vs QUBO vs Qiskit optimization
data/raw/                      Six source datasets (WaPOR, climate, soil, vegetation, targets)
data/processed/                Master datasets + optimization result tables
models/                        Yield models, scalers, feature lists (+ Release note for the 3 large models)
docs/RESULTS.md                Complete verified metrics
```

## Data & transparency notes

- Feature data comes from WaPOR-style remote sensing and field/soil sources; the resource/yield targets were constructed with realistic agronomic relationships (stated openly in the notebook), which is also why the energy target carries little learnable signal.
- Three resource models (~44 MB each) exceed GitHub's 25 MB web-upload limit and are attached to the **v1.0 Release** instead of the Git tree — see `models/README.md`. All other models, including the 20 MB yield Random Forest, are in `models/`.
- `resource_scaler.pkl` and `yield_scaler.pkl` in the original project are byte-identical copies of the same scaler; both are kept so the app and notebook paths work unchanged.

## Limitations

- Recommendations are validated on one representative field sample; per-field optimization at scale is future work.
- The QUBO is solved by brute force / exact solver on a small discrete grid (7 water × 6 fertilizer levels) — a correctness proof for the formulation, not a quantum-advantage claim.

## Author

**Muhammad Hamza** — Data Scientist · [GitHub @mhamza6051](https://github.com/mhamza6051)
