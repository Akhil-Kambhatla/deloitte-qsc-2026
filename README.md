# Deloitte Quantum Sustainability Challenge 2026

**Team Samanvay** | University of Maryland, College Park

| Member | Program | Contribution |
|--------|---------|-------------|
| Akhil Kambhatla | M.S. Data Science | Quantum circuit design, domain-aware qubit mapping, pipeline architecture, ablation study |
| Hemanth Thulasiraman | M.S. Data Science | Feature engineering, classical baselines, Task 2 insurance modeling |
| Ravi Parvatham | M.S. Machine Learning | Data analysis, Task 2 regression, report writing |

**Submission:** [`Team_Samanvay.pdf`](Team_Samanvay.pdf)
**Platform:** IBM Qiskit 2.3.1, Qiskit Aer statevector simulator

---

## What We Built

A hybrid quantum-classical ML pipeline that predicts wildfire risk across 2,593 California zip codes (Task 1) and forecasts insurance premiums (Task 2), using only the competition-provided datasets.

We trained five 10-qubit Variational Quantum Classifier (VQC) configurations and compared them against XGBoost and Random Forest baselines. Our central contribution is **Domain-Aware Qubit Mapping**: reordering features by scientific domain cluster before qubit assignment so that the circuit's nearest-neighbor entanglement captures within-domain interactions. This zero-cost change improved quantum fire capture from 37% to 63% on 2023 validation data.

We also designed the **WildfireQCircuit**, a problem-specific parameterized quantum circuit whose 14 CZ entangling gates per layer encode causal relationships from wildfire literature, and conducted a controlled ablation study across all five configurations.

---

## Key Results

### Task 1: Wildfire Risk Classification (Train 2018-2020, Test 2021)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| XGBoost (43 features) | 0.889 | 0.412 | 0.665 | **0.509** |
| Random Forest (43 features) | 0.860 | 0.351 | 0.728 | 0.473 |
| Q1: Baseline VQC | 0.793 | 0.203 | 0.478 | 0.285 |
| **Q2: Domain-Aware VQC** | **0.789** | **0.231** | **0.621** | **0.337** |
| Q3: WildfireQCircuit | 0.627 | 0.151 | **0.714** | 0.249 |
| Q4: WildfireQCircuit Tuned | 0.751 | 0.189 | 0.571 | 0.284 |
| Q5: Random Entanglement | 0.675 | 0.163 | 0.670 | 0.262 |

### 2023 Validation (118 known fire zip codes)

| Model | Fires Predicted | Known Fires Captured | Capture Rate |
|-------|----------------|---------------------|-------------|
| XGBoost (top-500) | 195 | 90 | 76% |
| Random Forest (top-500) | 300 | 97 | 82% |
| Q1: Baseline VQC | 479 | 44 | 37% |
| **Q2: Domain-Aware VQC** | **576** | **74** | **63%** |
| Q3: WildfireQCircuit | 715 | 69 | 58% |

### Task 2: Insurance Premium Prediction (Train 2020, Test 2021)

| Model | MAE | R² | MAPE | Median APE |
|-------|-----|-----|------|-----------|
| Naive (last year) | $739,875 | 0.735 | 18.8% | 11.1% |
| Random Forest | $414,644 | 0.862 | 12.9% | 7.1% |
| **XGBoost** | **$378,712** | **0.881** | **12.3%** | **6.0%** |

---

## Repository Structure

```
deloitte-qsc-2026/
├── Team_Samanvay.pdf                           # Competition submission PDF (15 pages)
├── README.md
├── requirements.txt                            # Pinned dependencies (Python 3.11)
├── .gitignore
│
├── data/                                       # All datasets and model outputs
│   ├── wildfire_weather.csv                    # Raw wildfire + weather (125,476 rows)
│   ├── insurance_fire_census_weather_raw.csv   # Raw insurance + census (47,033 rows)
│   ├── Feature_Descsription_FireHistory_Census.csv   # Column descriptions
│   ├── FeatureDescription_fire_insurance.csv         # Column descriptions
│   ├── feature_matrix.csv                      # Engineered features (pre-cleaning)
│   ├── feature_matrix_clean.csv                # Engineered features (43 features)
│   ├── X_train_top10.npy / X_test_top10.npy   # Top-10 features for Q1/Q2 VQC
│   ├── X_train_pca.npy / X_test_pca.npy       # PCA-reduced features (early experiments)
│   ├── X_train_pca10.npy / X_test_pca10.npy   # PCA-10 variant
│   ├── X_train_pca12.npy / X_test_pca12.npy   # PCA-12 variant
│   ├── X_train_top6.npy / X_test_top6.npy     # Top-6 feature variant
│   ├── X_train_wildfire_circuit.npy / X_test_wildfire_circuit.npy  # Q3 circuit features
│   ├── y_train.npy / y_test.npy               # Binary fire labels
│   ├── top10_features.txt / top6_features.txt  # Feature name lists
│   ├── task1_predictions_2023.csv              # Task 1A: 2,593 zip-code predictions
│   ├── task2_predictions_2021.csv              # Task 2: earlier prediction run
│   └── task2_predictions_2021_final.csv        # Task 2: final predictions
│
├── figures/                                    # Publication-quality figures (used in PDF)
│   ├── fig1_pipeline.pdf                       # Pipeline overview
│   ├── fig2_circuit_topology.pdf               # WildfireQCircuit entanglement diagram
│   ├── fig3_model_comparison.pdf               # F1 bar chart (classical vs quantum)
│   ├── fig4_capture_rate.pdf                   # 2023 fire capture validation
│   ├── fig5_task2_scatter.pdf                  # Predicted vs actual premium scatter
│   ├── fig6_feature_importance.pdf             # Top-15 XGBoost feature importances
│   └── references.bib                          # BibTeX references
│
├── notebooks/
│   ├── submission/                             # >>> START HERE <<<
│   │   └── 07_final_submission.ipynb           # Complete pipeline with all results
│   │
│   └── exploration/                            # Development experiments
│       ├── 01_eda.ipynb                        # Exploratory data analysis
│       ├── 02_feature_engineering.ipynb         # 43-feature construction
│       ├── 03_classical_baseline.ipynb          # XGBoost/RF baselines, top-10 selection
│       ├── 04_quantum_model.ipynb               # VQC training (Q1, Q2 configurations)
│       ├── 05_wildfire_qcircuit.ipynb           # WildfireQCircuit design + ablation (Q3-Q5)
│       ├── 05_task2_insurance.ipynb             # Task 2 development (iteration 1)
│       └── 06_task2_insurance.ipynb             # Task 2 development (iteration 2)
│
└── src/
    └── __init__.py                             # Package placeholder
```

---

## Main Deliverable

**[`notebooks/submission/07_final_submission.ipynb`](notebooks/submission/07_final_submission.ipynb)** contains the complete end-to-end pipeline: data loading, feature engineering, classical baselines, all five quantum model configurations, 2023 predictions, Task 2 insurance modeling, and the corrected MAPE evaluation. All cells have saved outputs.

---

## Exploration Notebooks

| Notebook | What it does |
|----------|-------------|
| `01_eda.ipynb` | Dataset exploration: distributions, fire rates, missing values, correlation analysis |
| `02_feature_engineering.ipynb` | Constructs 43 features across 4 domains (fire history, weather, insurance, census) |
| `03_classical_baseline.ipynb` | Trains XGBoost and Random Forest, runs feature importance analysis, selects top-10 features for quantum models |
| `04_quantum_model.ipynb` | Trains Q1 (baseline VQC) and Q2 (domain-aware reordered VQC) using Qiskit |
| `05_wildfire_qcircuit.ipynb` | Designs the WildfireQCircuit (Q3), trains Q4 (tuned) and Q5 (random ablation control) |
| `05_task2_insurance.ipynb` | First iteration of Task 2 insurance premium regression |
| `06_task2_insurance.ipynb` | Refined Task 2 with lag features, rolling averages, and quantum fire risk integration |

---

## Setup

```bash
git clone https://github.com/Akhil-Kambhatla/deloitte-qsc-2026.git
cd deloitte-qsc-2026

python3 -m venv qsc_env
source qsc_env/bin/activate        # Mac/Linux
# qsc_env\Scripts\activate         # Windows

pip install -r requirements.txt

# Register kernel for Jupyter
python3 -m ipykernel install --user --name=qsc_env --display-name "Python (qsc_env)"
```

Then open `notebooks/submission/07_final_submission.ipynb` and select the `Python (qsc_env)` kernel.

To reproduce the full pipeline from scratch, run the exploration notebooks in order: `01` → `02` → `03` → `04` → `05_wildfire_qcircuit` → `05_task2` / `06_task2`.

> **Mac users:** If XGBoost fails to install, run `brew install libomp` first.

---

## Links

- [Deloitte Quantum Sustainability Challenge 2026](https://us.ekipa.de/deloitte-quantum-2026)
- [IBM Qiskit Documentation](https://quantum.cloud.ibm.com/docs/en/guides)
- [Qiskit Machine Learning](https://qiskit-community.github.io/qiskit-machine-learning/)
