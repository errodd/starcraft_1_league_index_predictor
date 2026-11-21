# StarCraft 2 League Index Predictor

**University of Carabobo**  
**Experimental Faculty of Sciences and Technology (FACYT)**  
**Computer Science Department**  
**Course:** Machine Learning (Elective)  
**Date:** October 2026  
**Authors:** Juan Suárez & Josue Carrion  

---

## Project Overview
This repository hosts a supervised Machine Learning project aimed at predicting a StarCraft 2 player's competitive rank (`LeagueIndex`, spanning from Bronze [1] to Professional [8]) based on in-game behavioral telemetry and demographic attributes from the SkillCraft1 dataset.

The prediction task is framed as a **post-match Multiclass / Ordinal Classification problem**, where telemetry is extracted from complete game replays (`.SC2Replay`), guaranteeing that all aggregated in-game statistics are available at inference time without lookahead bias.

---

## Repository Structure

The workflow is organized into modular directories and progressive Jupyter Notebooks:

```text
ml_proyecto/
├── data/
│   └── SkillCraft1_Dataset.csv      # Raw SkillCraft1 dataset from UCI ML Repository
├── notebooks/
│   ├── 01_eda.ipynb                 # Comprehensive Exploratory Data Analysis
│   ├── 02_data_preparation.ipynb    # Preprocessing, transformations, and train/test split
│   ├── 03_modeling_and_model_selection.ipynb # Model benchmarking, Optuna tuning, and final test evaluation
│   └── logo-facty.jpeg              # University institutional logo (local reference)
├── figures/                         # Exported exploratory and analytical plots
├── logo-facty.jpeg                  # University institutional logo (root reference)
├── requirements.txt                 # Project environment dependencies
└── README.md                        # Project documentation and summary
```

### Notebook Milestones
* **`notebooks/01_eda.ipynb` (Exploratory Data Analysis):**
  * **Provenance & Population Context:** Formal attribution to the SkillCraft1 study (Thompson et al., 2013, *PLOS ONE*, UCI Repository). Characterization of community self-selection bias in voluntary replay contribution and discussion of player-level clustering (`PlayerID` absence).
  * **Data Quality & Audit:** Conversion of masked nulls (`?`), structured missingness analysis across leagues (preserving all 55 observations of Professional League 8), and targeted filtering of physical outliers.
  * **Univariate & Skewness Analysis:** Distribution modeling, skewness evaluation (`TotalHours` skew ≈ 14.54), and outlier inspection.
  * **Dual Baseline Benchmarking:** Formulated the naive majority-class baseline (**23.90% Accuracy**, **0.0482 Macro-F1**) and a strong expert **Domain Heuristic Baseline** based on `APM` and `ActionLatency` skill brackets (**30.45% Accuracy**, **0.3042 Macro-F1**, **1.0186 Ordinal MAE**).
  * **Feature Engineering & Inventory:** Formulation of `CognitiveSpeedIndex` (CSI = $\frac{\text{APM}}{\text{ActionLatency} + 1}$) and audited analytical inventory table (verifying 55 missing in `Age`, discrete integer range `[5, 58]` for `TotalMapExplored`).

* **`notebooks/02_data_preparation.ipynb` (Data Preparation & Feature Engineering):**
  * **Input Contract Validation:** Defensive verification of primary key uniqueness (`GameID`), non-nullity, and valid target boundaries with strict exception raising (`ValueError`).
  * **Leakage-Free Splitting & Sealed Test Vault:** Stratified 80/20 train/test split preserving the 1.62% proportion of League 8 across partitions. All smoke tests and diagnostic validations restricted to `X_train_diag`, ensuring `X_test` remains a completely sealed vault.
  * **Finiteness & Routing:** Verification of mathematical finiteness (`np.isfinite()`) across all 23 output features. Clean `ColumnTransformer` routing separating median-imputed demographics, robust-scaled telemetry, binary flags, and dropped identifiers.
  * **Pipeline Factory:** Implementation of `build_model_pipeline(estimator)` factory using `sklearn.base.clone(prep_pipeline)` guaranteeing independent parameter estimation within cross-validation folds.
  * **Reproducible Handoff & Environment Locking:** Export of raw partition files (`X_train_raw.csv`, `X_test_raw.csv`, `y_train.csv`, `y_test.csv`) and JSON metadata manifest (`data_preparation_metadata.json`) with exact runtime environment specifications (Python 3.12.3, scikit-learn 1.9.1, Optuna 5.0.0).

* **`notebooks/03_modeling_and_model_selection.ipynb` (Model Selection & Hyperparameter Optimization):**
  * **Evaluation Protocol:** `Macro-F1` as primary metric, complemented by `Balanced Accuracy`, `Accuracy`, `Ordinal MAE`, and boundary proximity ($\le 1$ and $\le 2$ leagues). 5-Fold Stratified Cross-Validation with strict preprocessor encapsulation.
  * **Feature Ablation Study:** Empirical benchmarking across 6 configurations demonstrating that streamlined representation **Var 5** (CSI + `log_WeeksOfPlay` without noisy tactical aggregations) is superior, achieving **0.4346 ± 0.0199 Macro-F1** and lowest MAE (**0.7738**).
  * **Multi-Family Benchmarking:** Evaluated 5 diverse candidates across identical folds on Var 5: Naive Dummy (0.0482), Domain Heuristic (0.3036), Logistic Regression (0.3865), Random Forest (0.4243), and HistGradientBoosting (0.4346).
  * **Hyperparameter Optimization with Optuna:** Bayesian optimization (`TPESampler`, seed=42) conducted for both `RandomForest` (achieving **0.4452 ± 0.0276** Macro-F1 and **0.4614** Balanced Acc) and `HistGradientBoosting`.
  * **Paired Test Set Evaluation:** Unsealed the test vault ($N = 679$) for a single evaluation:
    * `Champion: Random Forest Tuned (Var 5)`: **Macro-F1 = 0.4519**, **Balanced Acc = 0.4706**, **Accuracy = 41.53%**, **MAE = 0.7585**.
    * `Benchmark: HistGradientBoosting (Var 5)`: **Macro-F1 = 0.4256**, **Balanced Acc = 0.4162**, **Accuracy = 42.27%** (vs. 23.86% baseline), **MAE = 0.7320**.
  * **Clinical Error Analysis & Throughput:** Over **86.16%** of predictions fall within $\pm 1$ tier (**98.38%** within $\pm 2$ tiers), and League 8 achieved **90.91% Recall** (10 of 11) with **100.0% Precision**. Empirical batch throughput measured at **0.073 ms per match** (> 13,000 matches/sec).

---

## Key Findings

1. **Structured Missingness in League 8:** All 55 Professional players (League 8) lack demographic data (`Age`, `TotalHours`, `HoursPerWeek`). Preserving these records and imputing post-split without missing indicators successfully prevented shortcut learning.
2. **Telemetry Associations:** In-game mechanical telemetry metrics show the strongest observed linear associations with competitive rank: lower `ActionLatency` ($r = -0.67$) and higher `APM` ($r = 0.66$) strongly correspond to elite tiers.
3. **Engineered Feature Impact:** The streamlined **Var 5** representation (`CognitiveSpeedIndex` and tenure proxy `log_WeeksOfPlay`) proved empirically superior to raw telemetry and full aggregations, confirming that motor-cognitive interaction ratios capture authentic skill signal.
4. **Generalization Performance:** The champion models achieved **42.27% test accuracy** (almost doubling the 23.86% naive baseline and outperforming the 30.04% domain heuristic), an outstanding **90.91% recall on League 8 with 100% precision**, and over **86% of predictions within $\pm 1$ league tier** (severe errors $\ge 3$ tiers restricted to 1.62%).

---

## Installation & Reproducibility

### 1. Environment Setup
Clone the repository and install the verified dependencies using Python 3.10+:

```bash
git clone <repo-url>
cd ml_proyecto
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Running the Notebooks
Launch Jupyter Lab or Notebook from the repository root:

```bash
jupyter lab
```

Navigate to `notebooks/` and execute in sequential order:
1. `01_eda.ipynb`
2. `02_data_preparation.ipynb`
3. `03_modeling_and_model_selection.ipynb`

All relative paths to data (`../data/`) and assets will resolve automatically.