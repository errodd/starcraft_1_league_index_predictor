# starcraft_1_league_index_predictor

**University of Carabobo**  
**Experimental Faculty of Sciences and Technology (FACYT)**  
**Course:** Machine Learning (Elective)  
**Date:** August 2026  

---

## Project Overview
This repository contains a comprehensive Exploratory Data Analysis (EDA) of the StarCraft 2 (SkillCraft) dataset. The primary objective of this phase is to evaluate data quality and uncover relationships between a player's mechanical/telemetry metrics (such as Actions Per Minute and Action Latency) and their competitive rank (`LeagueIndex`). 

This analysis serves as the foundational step for building a predictive Multiclass Classification Machine Learning model in subsequent stages of the course.

## Repository Structure
The analysis is encapsulated in a single Jupyter Notebook, structured according to the project's academic rubric:
*   **4.1 Context and Problem Definition:** Domain explanation and target variable identification.
*   **4.2 Guiding Questions:** Core analytical questions driving the EDA.
*   **4.3 & 4.4 Configuration and Loading:** Environment setup and dataset ingestion.
*   **4.5 & 4.6 Data Quality:** Structural inspection, handling of hidden nulls, and removal of physically impossible outliers without causing data leakage.
*   **4.7 Univariate Analysis:** Distribution analysis of key demographic and mechanical features.
*   **4.8 Target Variable:** Analysis of the `LeagueIndex` class balance and definition of the naive baseline (24.08%).
*   **4.9 Bivariate and Multivariate Analysis:** Correlation mapping and cross-feature interactions.
*   **4.10 Feature Engineering:** Creation of the `CognitiveSpeedIndex` (CSI) to address multicollinearity and boost predictive signals.
*   **4.11 General Conclusions:** Summary of insights and recommendations for the modeling phase.

## Key Findings
1.  **Data Quality:** Demographic variables contained minor missing values (`?`) and extreme logical errors (e.g., 1,000,000 Total Hours), which were deterministically filtered. Mechanical outliers were retained as they represent elite professional performance.
2.  **Mechanical Dominance:** Telemetry metrics, specifically `ActionLatency` (r = -0.66) and `APM` (r = 0.62), are the strongest linear predictors of a player's skill tier. 
3.  **Engineered Feature:** A strong multicollinearity between APM and Latency was resolved by creating the `CognitiveSpeedIndex`, which exponentially separates high-tier players from lower leagues.
4.  **Class Imbalance:** The dataset follows a bell-shaped distribution regarding skill, leading to underrepresentation at the extreme ends (League 1 and League 8), requiring specific balancing strategies during future modeling.

## Requirements & Reproducibility
To run this notebook, the following Python libraries are required:
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`

Simply place the dataset , the university logo image (`logo-facty.jpeg`), and the Jupyter Notebook (`.ipynb`) in the same root directory and execute all cells.
