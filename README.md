# Deloitte Quantum Sustainability Challenge 2026

**Team Name:** Samanvay
**University:** University of Maryland, College Park

| Member | Role |
|---|---|
| Akhil Kambhatla | Quantum model design, pipeline architecture, final submission |
| Hemanth Thulasiraman | Feature engineering, classical baselines, ablation studies |
| Ravi Parvatham | Task 2 insurance modeling, data analysis, report |

---

## What This Project Does

We built machine learning models to tackle two real-world problems around wildfire risk in California:

**Task 1 — Wildfire Risk Prediction (2023):** Given historical weather and fire data for thousands of California zip codes, predict which zip codes are likely to experience wildfires in 2023. We built both a classical machine learning model and a quantum machine learning model, then compared the two.

**Task 2 — Insurance Premium Prediction:** Using insurance records and fire risk data from previous years, predict what insurance companies will charge homeowners in each zip code.

The novel angle: we used **quantum computing** (IBM Qiskit) to build the wildfire classifier — a Variational Quantum Circuit (VQC) — and compared it head-to-head against traditional machine learning methods like XGBoost.

---

## Key Results

| Task | Model | Metric | Score |
|---|---|---|---|
| Task 1A | Quantum VQC (best) | F1 Score | 0.337 |
| Task 1B | XGBoost (classical) | F1 Score | 0.509 |
| Task 1A | Predictions generated | Zip codes | 2,593 for 2023 |
| Task 1A | Quantum models trained | Count | 5 configurations |
| Task 2 | XGBoost regression | R² | 0.881 |
| Task 2 | XGBoost regression | MAPE | 12.3% |

**Platform:** IBM Qiskit 2.3.1, Qiskit Aer simulator (statevector)

---

## Repository Structure

```
deloitte-qsc-2026/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                                      # Raw and processed datasets
│   ├── wildfire_weather.csv                   # Task 1 raw data (2018–2023)
│   ├── insurance_fire_census_weather_raw.csv  # Task 2 raw data (2018–2021)
│   ├── Feature_Descsription_FireHistory_Census.csv
│   ├── FeatureDescription_fire_insurance.csv
│   ├── feature_matrix.csv                     # Engineered features (pre-cleaning)
│   ├── feature_matrix_clean.csv               # Engineered features (cleaned)
│   ├── X_train_pca.npy / X_test_pca.npy      # PCA-reduced features for quantum model
│   ├── y_train.npy / y_test.npy              # Labels
│   ├── task1_predictions_2023.csv             # Task 1A output
│   └── task2_predictions_2021_final.csv       # Task 2 output
│
├── notebooks/
│   ├── submission/                            # *** MAIN DELIVERABLE ***
│   │   └── 07_final_submission.ipynb
│   │
│   └── exploration/                          # Development experiments
│       ├── 01_eda.ipynb
│       ├── 02_feature_engineering.ipynb
│       ├── 03_classical_baseline.ipynb
│       ├── 04_quantum_model.ipynb
│       ├── 05_wildfire_qcircuit.ipynb
│       ├── 05_task2_insurance.ipynb
│       └── 06_task2_insurance.ipynb
│
├── src/
│   └── __init__.py
├── results/
└── report/
```

---

## The Main Deliverable

**`notebooks/submission/07_final_submission.ipynb`** is the notebook to look at first.

Think of it as a polished, end-to-end report and demo in one file. It walks through the full story: the data, the approach, the quantum circuit we designed, the predictions, and the results — all in one place with explanations written for a general audience. This is what we submitted as our competition entry.

---

## The Exploration Notebooks

The `notebooks/exploration/` folder contains all the intermediate work that led to the final submission — think of these as our lab notebooks or scratch pads.

- **01_eda.ipynb** — We examined both datasets to understand what we were working with: how many zip codes, how often fires occur, which features correlate with fires, etc.
- **02_feature_engineering.ipynb** — We turned the raw data into a set of 43 structured features per zip code per year, including fire history, weather statistics, and insurance metrics.
- **03_classical_baseline.ipynb** — We trained traditional ML models (Random Forest, XGBoost) to set a performance benchmark and generate the compressed features needed for the quantum model.
- **04_quantum_model.ipynb** — We built and trained the Variational Quantum Classifier (VQC) using IBM Qiskit, testing different circuit architectures and optimizers.
- **05_wildfire_qcircuit.ipynb** — Additional quantum circuit experiments with a custom wildfire-specific encoding.
- **05_task2_insurance.ipynb / 06_task2_insurance.ipynb** — Iterative development of the insurance premium prediction model for Task 2.

These notebooks contain experiments that did not all make it into the final submission, including ablation studies (testing what happens when you remove certain features) and comparisons across different model configurations.

---

## Setup Instructions

### 1. Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/Akhil-Kambhatla/deloitte-qsc-2026.git
cd deloitte-qsc-2026
```

You can clone it anywhere you like — your Desktop, Documents, or a dedicated projects folder.

### 2. Create a virtual environment

A virtual environment keeps the project's packages isolated from the rest of your system. Run these commands from inside the `deloitte-qsc-2026` folder:

```bash
python3 -m venv qsc_env
source qsc_env/bin/activate      # Mac/Linux
# qsc_env\Scripts\activate       # Windows (use this instead on Windows)
```

You should see `(qsc_env)` appear at the start of your terminal prompt.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for Mac users:** If XGBoost fails to install, run `brew install libomp` first and retry.

### 4. Open in VS Code (recommended)

Open VS Code, then open the `deloitte-qsc-2026` folder:

```bash
code .
```

Or open VS Code manually, go to **File → Open Folder**, and select the `deloitte-qsc-2026` folder.

### 5. Register the virtual environment as a Jupyter kernel

This step lets VS Code (and Jupyter) find and use the packages you just installed. Make sure your virtual environment is still active (you should see `(qsc_env)` in your terminal), then run:

```bash
python3 -m ipykernel install --user --name=qsc_env --display-name "Python (qsc_env)"
```

### 6. Select the kernel in VS Code

1. Open any `.ipynb` notebook file in VS Code
2. Click the kernel selector in the top-right corner (it may show "Select Kernel" or a Python version)
3. Choose **"Python (qsc_env)"** from the list
4. Apply this same kernel to all notebooks in this repo

### 7. Run the notebooks

- **To see the final submission:** Open `notebooks/submission/07_final_submission.ipynb` and run all cells (Menu → Run → Run All Cells)
- **To reproduce the full pipeline from scratch:** Run the exploration notebooks in order — `01` → `02` → `03` → `04`. Each one saves intermediate files to `data/` that the next notebook reads.

---

## Competition Links

- [Deloitte Quantum Sustainability Challenge 2026](https://us.ekipa.de/deloitte-quantum-2026)
- [Qiskit Documentation](https://quantum.cloud.ibm.com/docs/en/guides)
- [Qiskit Machine Learning](https://qiskit-community.github.io/qiskit-machine-learning/)
