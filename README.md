# Deloitte Quantum Sustainability Challenge 2026

## Wildfire Risk and Insurance Premium Prediction


---

## What is this competition about?

We are given two datasets about California zip codes:
1. **Wildfire dataset** with weather data and fire incidents (2018 to 2023)
2. **Insurance dataset** with premiums, claims, census data, and fire risk scores (2018 to 2021)

We have three tasks:
- **Task 1A:** Build a quantum machine learning model that predicts which zip codes will have wildfires in 2023, using data from 2018 to 2022
- **Task 1B:** Compare the quantum model against classical models (like XGBoost) and discuss the pros and cons
- **Task 2:** Build a time series model that predicts insurance premiums for 2021 using 2018 to 2020 data

---

## What we have done so far

### 1. Data exploration (notebook 01_eda.ipynb)

We explored both datasets and found some important things:

**Wildfire dataset (125,476 rows, 2,599 zip codes, 6 years)**
- Only about 2,211 rows are actual fire events. The rest are monthly weather records per zip code.
- For 2018 to 2021, we have full weather data for all zip codes (about 31,000 weather rows per year).
- For 2022 and 2023, we only have fire event rows. There is NO weather data for those years.
- About 8.5% of zip codes have a fire in any given year. This means the problem is heavily imbalanced.

**Insurance dataset (47,033 rows, 2,251 zip codes, 4 years)**
- Has 76 columns covering premiums, claims, coverage, fire risk scores, and census demographics.
- "Earned Premium" is our prediction target for Task 2. It has no missing values.
- Some census columns have about 30% missing data. Some insurance columns (like Avg PPC) have data only in 2018 and 2019.

**Key findings from our analysis:**
- Weather alone is a weak signal for predicting fires. Fire and no fire zip codes have very similar temperatures and rainfall.
- Fire history is the strongest predictor. If a zip code had a fire before, it is about 4 times more likely to have one again.
- The insurance dataset's "Avg Fire Risk Score" is very useful. Fire zip codes average 1.22 vs 0.71 for no fire zip codes.
- Fire zip codes tend to have lower housing values ($580K vs $752K), lower incomes, and higher insurance premiums.
- Peak fire season is June through October. July has the most fires.
- Top fire causes: unknown, lightning, equipment use, vehicles, and power lines.

### 2. Feature engineering (notebook 02_feature_engineering.ipynb)

We built a feature matrix with one row per zip code per year (2018 to 2021). Each row has 43 features:

**Fire history features (strongest signal)**
- Number of fires in prior years
- Total acres burned in prior years
- Maximum single fire size
- Whether the zip ever had a fire before
- Log transformed versions of acres (because fire sizes are extremely skewed)

**Weather features (weak signal on their own but still useful)**
- Average and max temperature for the full year
- Average and max temperature for fire season only (June to October)
- Total precipitation, minimum precipitation
- Temperature range across the year

**Insurance and risk features (moderate to strong signal)**
- Average Fire Risk Score
- Average PPC (Public Protection Classification, measures fire department quality)
- Earned Exposure and Earned Premium
- Coverage A and C weighted averages
- CAT and Non CAT fire claims (count and dollar losses)
- Number of high, very high, moderate, low, and negligible fire risk exposures

**Census features (moderate signal)**
- Total population, median income
- Housing value, total housing units, vacancy numbers
- Average household size, educational attainment
- Owner vs renter occupied housing

Missing values were handled by filling weather nulls with zip level averages, claims with zeros, and census/scores with the median.

### 3. Classical baselines (notebook 03_classical_baseline.ipynb)

We trained on 2018 to 2020 and tested on 2021. Results:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Random Forest (43 features) | 0.860 | 0.351 | 0.728 | 0.473 |
| XGBoost (43 features) | 0.889 | 0.412 | 0.665 | 0.509 |
| Random Forest (6 PCA features) | 0.852 | 0.327 | 0.670 | 0.439 |
| XGBoost (6 PCA features) | 0.847 | 0.312 | 0.647 | 0.421 |

We also validated the model on 2022 and 2023 fire zip codes:
- Fire zip codes get a predicted probability about 4.7 times higher than non fire zip codes
- The top 500 riskiest zip codes capture about 69% of actual fires in both 2022 and 2023
- This proves the model generalizes well to unseen years

The PCA reduced models (6 features) are needed for the quantum model because quantum circuits can only handle a small number of features. With 6 PCA components, we capture 66.4% of the variance in the original 43 features.

### 4. Quantum model (notebook 04_quantum_model.ipynb)

We built a Variational Quantum Classifier (VQC) using Qiskit. The architecture:
- 6 qubits (one per PCA component)
- ZZFeatureMap with 2 repetitions and linear entanglement to encode data
- RealAmplitudes ansatz with 2 repetitions and 18 trainable parameters
- Trained on 400 balanced samples (50/50 fire and no fire)
- Ran on the Qiskit Aer statevector simulator

Results:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Quantum VQC (COBYLA, 100 iter) | 0.525 | 0.097 | 0.540 | 0.164 |
| Quantum VQC (COBYLA, 150 iter) | 0.542 | 0.093 | 0.491 | 0.156 |
| Quantum VQC (SPSA, 150 iter) | 0.481 | 0.076 | 0.451 | 0.130 |

The quantum model is significantly weaker than the classical baselines. This is expected and normal. Current quantum simulators are limited by the number of qubits, training data size, and optimizer convergence. The competition is not expecting quantum to beat classical. They want to see that we built it correctly and can discuss the tradeoffs thoughtfully.

---

## Project structure

```
deloitte-qsc-2026/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── wildfire_weather.csv              (Task 1A raw data)
│   ├── insurance_fire_census_weather_raw.csv  (Task 2 raw data)
│   ├── Feature_Descsription_FireHistory_Census.csv
│   ├── FeatureDescription_fire_insurance.csv
│   ├── feature_matrix.csv                (43 features, before cleaning)
│   ├── feature_matrix_clean.csv          (43 features, no nulls)
│   ├── X_train_pca.npy                   (6 PCA features, train)
│   ├── X_test_pca.npy                    (6 PCA features, test)
│   ├── y_train.npy                       (labels, train)
│   ├── y_test.npy                        (labels, test)
│   ├── scaler.pkl                        (StandardScaler)
│   └── pca.pkl                           (PCA transformer)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classical_baseline.ipynb
│   └── 04_quantum_model.ipynb
├── src/
│   └── __init__.py
├── results/
└── report/
```

---

## What still needs to be done

### High priority (must do)

**Task 2: Insurance premium prediction**
- Build a model that predicts Earned Premium for each zip code in 2021 using 2018 to 2020 data
- The insurance dataset already has the data we need. We just need to set up a time series or regression model.
- We can use the fire risk score from Task 1 as an input feature to link the two tasks together.
- Options: XGBoost regression, or a simple panel regression with lagged features (prior year premium, claims).

**Submission PDF report**
- Must follow the format in the challenge statement (team overview, abstract, algorithm description, results, future vision)
- Maximum 400 word abstract
- Must include a clickable link to this GitHub repo

### Improvements to try (will make our submission stronger)

**Improve quantum model performance**
- Try different feature maps: PauliFeatureMap instead of ZZFeatureMap
- Try different ansatz circuits: EfficientSU2 instead of RealAmplitudes
- Try full entanglement instead of linear entanglement
- Increase training data from 400 to 800 or 1200 balanced samples (will be slower)
- Increase optimizer iterations to 300 or 500
- Try 8 PCA components (8 qubits) instead of 6 if training time allows
- Try a quantum kernel approach with a balanced training set instead of natural class ratio

**Improve classical baselines**
- Hyperparameter tuning with GridSearchCV or Optuna
- Try different class imbalance strategies: SMOTE oversampling instead of class weights
- Try threshold tuning: instead of the default 0.5 cutoff, find the optimal threshold using the ROC curve

**Add more features to the feature matrix**
- Fire recency: years since the most recent fire (a zip that burned last year is different from one that burned 3 years ago)
- Fire seasonality: average temperature specifically during months when fires historically occur
- Distance or spatial features: if neighboring zip codes have fires, this zip is higher risk
- Interaction features: temperature multiplied by low precipitation (hot and dry together)
- Rolling averages: 2 year or 3 year rolling average temperature and precipitation
- We already have 43 features. Adding more could help the classical models. For the quantum model, PCA will compress them down to 6 to 8 anyway, so more raw features means better PCA components.

**Improve recall specifically**
- Right now recall is our weak spot. We catch 54% to 73% of fires depending on the model.
- For wildfire prediction, missing a fire (false negative) is worse than a false alarm (false positive). So optimizing for recall matters more than precision.
- Try adjusting the decision threshold lower (e.g., 0.3 instead of 0.5) to catch more fires at the cost of more false alarms.
- Try cost sensitive learning: assign a higher penalty for missing fires than for false alarms.

### Scope for improvement (good to discuss in report even if we do not implement)

- **More qubits:** With 10 to 20 qubits, the quantum model could handle more features without PCA, preserving more information.
- **Real quantum hardware:** Running on actual quantum hardware (via AWS Braket during the 72 hour window) introduces noise but shows practical feasibility.
- **Quantum error mitigation:** Techniques like zero noise extrapolation could improve results on noisy hardware.
- **Data encoding strategies:** Amplitude encoding could pack more features into fewer qubits (exponential compression) but is harder to implement.
- **Hybrid ensemble:** Use the quantum model as one member of an ensemble alongside classical models, potentially capturing complementary patterns.
- **Counterfactual analysis for Task 1B:** Use the model to ask "what if" questions, like "how would fire risk change if temperature increased by 2 degrees?" This is great for the evaluation section.
- **Transfer learning:** Train on California data, test generalizability to other fire prone states.
- **Real time weather integration:** In a production system, the model could ingest live weather data for dynamic risk updates.

---

## How to run

1. Clone the repo:
```bash
git clone https://github.com/Akhil-Kambhatla/deloitte-qsc-2026.git
cd deloitte-qsc-2026
```

2. Create a virtual environment:
```bash
python3 -m venv qsc_env
source qsc_env/bin/activate
pip install -r requirements.txt
```

3. If on Mac and XGBoost fails:
```bash
brew install libomp
```

4. Open notebooks in VS Code or Jupyter:
```bash
jupyter notebook notebooks/
```

5. Run notebooks in order: 01, 02, 03, 04. Each one saves outputs that the next one needs.

---

## Who is doing what

| Person | Responsibility |
|---|---|
| Akhil | Quantum model (Task 1A/1B), overall pipeline, report |
| [Teammate 2] | [Assign: e.g., feature engineering improvements, Task 2] |
| [Teammate 3] | [Assign: e.g., classical model tuning, report writing] |

---

## Useful links

- [Competition page](https://us.ekipa.de/deloitte-quantum-2026)
- [Qiskit documentation](https://quantum.cloud.ibm.com/docs/en/guides)
- [Qiskit ML tutorials](https://qiskit-community.github.io/qiskit-machine-learning/)
- [Challenge statement PDF](data/challenge_statement.pdf)
