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
  * **Problem Formulation & Prediction Timestamp:** Definition of post-match inference, unit of observation, and documentation of the `PlayerID` limitation regarding I.I.D. assumptions.
  * **Data Quality & Audit:** Identification of hidden nulls (`?`), systematic missingness analysis across leagues (preserving all 55 observations of Professional League 8), conditional filtering of physical outliers with class-by-class audit table.
  * **Univariate Analysis:** Distribution, skewness evaluation (`TotalHours` skew ≈ 14.54), and outlier inspection.
  * **Target Distribution & Baseline:** Skill distribution assessment and derivation of the majority-class naive baseline (**23.90%** for League 4 - Platinum).
  * **Bivariate & Multivariate Associations:** Correlation heatmaps and cross-feature interactions (`ActionLatency` $r = -0.67$, `APM` $r = 0.66$).
  * **Feature Engineering:** Formulation of the candidate feature `CognitiveSpeedIndex` (CSI = $\frac{\text{APM}}{\text{ActionLatency} + 1}$) and definition of the experimental protocol for future modeling.
  * **Feature Inventory:** Structured analytical dictionary summarizing types, missingness, and preprocessing recommendations.

* **`notebooks/02_data_preparation.ipynb` (Data Preparation & Feature Engineering):**
  * **Input Contract Validation:** Defensive verification of primary key uniqueness (`GameID`) and valid target boundaries with strict exception raising (`ValueError`).
  * **Leakage-Free Splitting:** Stratified 80/20 train/test split preserving the 1.62% proportion of League 8 across partitions.
  * **Domain Feature Engineering:** Custom stateless `FunctionTransformer` generating four feature families (interaction ratio `CognitiveSpeedIndex`, tenure proxy `WeeksOfPlay`, tactical aggregations `MinimapUsage` and `ComplexMechanicsRate`, and unit-corrected `log1p` rate transforms).
  * **ColumnTransformer Integration:** Explicit branch routing separating numeric features with missing values (median imputation + robust scaling), standard telemetry (robust scaling), binary indicators (passthrough), and identifiers (dropped).
  * **Cross-Validation Leakage Prevention:** Implementation of `build_model_pipeline(estimator)` factory using `sklearn.base.clone(prep_pipeline)` ensuring parameters are learned strictly within each training fold.
  * **Ablation Study Protocol:** Definition of 5 experimental feature configurations to benchmark CSI and domain aggregations in modeling.
  * **Reproducible Handoff:** Export of raw partition files (`X_train_raw.csv`, `X_test_raw.csv`, `y_train.csv`, `y_test.csv`) and JSON metadata manifest (`data_preparation_metadata.json`).

* **`notebooks/03_modeling_and_model_selection.ipynb` (Model Selection & Hyperparameter Optimization):**
  * **Evaluation Protocol:** `Macro-F1` as primary metric (treating all leagues equitably), complemented by `Balanced Accuracy`, `Accuracy` (vs. 23.90% baseline), and `Ordinal MAE`. 5-Fold Stratified Cross-Validation with strict vaulting of the test set.
  * **Feature Ablation Study:** Empirical benchmarking demonstrating that `CognitiveSpeedIndex` captures vital non-linear telemetry interactions and that the tenure proxy `WeeksOfPlay` boosts cross-validation F1.
  * **Multi-Family Benchmarking:** Evaluated `DummyClassifier`, `LogisticRegression` (balanced), `RandomForestClassifier` (balanced), and `HistGradientBoostingClassifier` (balanced) across identical folds.
  * **Hyperparameter Optimization with Optuna:** 25-trial Bayesian optimization (`TPESampler`) tuning learning rate, regularization, leaf nodes, and minimum leaf samples within cross-validation folds.
  * **Single Test Set Evaluation:** Unsealed the test vault for a single final evaluation, achieving **40.06% Accuracy** (vs. 23.90% baseline), **Macro-F1 = 0.4454**, and **Ordinal MAE = 0.7953** tiers.
  * **Clinical Error Analysis:** Demonstrated that **83.65%** of all test predictions fall within $\pm 1$ adjacent league tier, and **90.9%** (10 of 11) of Professional players (League 8) were correctly identified solely through gameplay telemetry.

---

## Key Findings

1. **Structured Missingness in League 8:** All 55 Professional players (League 8) lack demographic data (`Age`, `TotalHours`, `HoursPerWeek`). Rather than dropping these rows (which would eliminate the entire professional tier), missing values are preserved and handled via stratified post-split imputation without missing indicators.
2. **Telemetry Associations:** In-game mechanical telemetry metrics show the strongest observed linear associations with competitive rank: lower `ActionLatency` ($r = -0.67$) and higher `APM` ($r = 0.66$) strongly correspond to elite tiers.
3. **Engineered Feature Impact:** The `CognitiveSpeedIndex` and domain transforms proved empirically superior to raw telemetry alone in cross-validation ablation testing.
4. **Generalization Performance:** The tuned `HistGradientBoostingClassifier` achieved **40.06% test accuracy** (substantially surpassing the 23.90% naive baseline) and an impressive **90.9% recall on League 8**, with over 83% of predictions falling within $\pm 1$ league tier.

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