# Models

## In this folder (in Git)

| File | What it is |
|---|---|
| `yield_linear_regression.pkl` | Best yield model (R² 0.875) |
| `yield_xgboost.pkl` | Yield model (R² 0.871) |
| `yield_random_forest.pkl` | Yield model (R² 0.867, ~20 MB) |
| `yield_scaler.pkl` / `resource_scaler.pkl` / `energy_scaler.pkl` | Feature scalers |
| `*_feature_cols.json` | Feature column lists per model |

## Large resource models — download from the v1.0 Release

These three files are ~44 MB each, above GitHub's 25 MB per-file web limit, so they are attached as **Release assets** instead of living in the Git tree:

- `water_model.pkl`
- `fertilizer_model.pkl`
- `energy_model_v2.pkl`

**To run the app/notebook fully:** download them from the repository's **Releases → v1.0** page and place them in this `models/` folder, then run as described in the main README.
