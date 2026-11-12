# Full Results — Precision Agriculture Yield & Resource Optimization

All figures are taken verbatim from the project notebook outputs and the result CSVs in `data/processed/`.

## Dataset

- 3,000 records × 6 merged sources: WaPOR (3,000 rows), climate (3,000), soil moisture (3,000), vegetation (3,000), resource targets (3,000), soil properties (150 fields)
- Final modelling dataset: `data/processed/final_model_dataset.csv` (3,000 rows)

## Yield prediction (test set)

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 0.0495 | 0.8751 |
| XGBoost | 0.0503 | 0.8713 |
| Random Forest | 0.0511 | 0.8671 |

## Resource recommendation models

| Model | RMSE | R² | Note |
|---|---|---|---|
| Fertilizer | 10.7284 | 0.8357 | Strongest resource model |
| Water | 20.9143 | 0.6760 | |
| Energy (v1) | 5.4065 | 0.1096 | Weak — target has little dependence on features |
| Energy (v2, retrained) | 5.4673 | 0.0895 | Retraining did not fix it; reported honestly |

## Optimization — three solvers, one answer

Objective: maximize yield utility − energy cost over discrete water × fertilizer levels, evaluated on a representative sample (index 1801).

| Method | Best water | Best fertilizer | Objective |
|---|---|---|---|
| Classical exhaustive search | 280 | 180 | 0.11588 |
| QUBO brute force (binary encoding) | 280 | 180 | −0.11588 (energy) |
| Qiskit exact solver (`QuadraticProgram`) | 280 | 180 | fval = −0.11588 |

At the optimum: predicted energy 19.526, yield suitability 0.3334. Notebook conclusion: *"All methods match. Optimization pipeline is validated."*

Source files: `data/processed/classical_optimization_results.csv`, `qubo_summary.csv`, `exact_solver_summary.csv`, `final_recommendation_summary.csv`.
