# Predicting Prospective Customer Risk in Sparse Data Environments

## Overview

This project investigates whether variables with high levels of missingness can still provide useful predictive information when modelling prospective customer risk.

Using a customer dataset containing 9,289 records and 26 variables, I compared **Random Forest** and **Extreme Gradient Boosting (XGBoost)** classifiers for predicting whether customer accounts would be classified as **GOOD** or **BAD**.

A key challenge was that several predictors contained more than 50% missing values. Rather than removing these variables automatically, the analysis examined whether their missingness itself contained useful predictive information.

## Objectives

The project aimed to:

- investigate patterns of missingness within the data
- compare Random Forest and XGBoost classifiers
- assess whether high-missingness variables contribute predictive value
- reduce model complexity using recursive feature elimination (RFE)
- interpret important predictors using feature importance and partial dependence
- apply probability thresholds to distinguish confident predictions from uncertain cases
- evaluate the final model on an unseen test set

## Methods

The workflow included:

1. Data cleaning and preprocessing
2. Dummy encoding of categorical variables
3. Exploratory analysis of missingness
4. Five-fold cross-validation
5. Random Forest and XGBoost model comparison
6. Hyperparameter tuning
7. Recursive feature elimination within cross-validation
8. Feature-importance analysis
9. Partial-dependence analysis
10. Chi-squared tests of association for selected missingness patterns
11. Out-of-fold probability estimation
12. GOOD / BAD / PASS classification thresholds
13. Final evaluation using a held-out test set

## Informative Missingness

Several variables contained substantial missingness, including:

- `I_01`
- `I_02`
- `D_01`
- `D_02`
- `S_01`
- `CA_02`

Exploratory analysis showed that accounts were consistently more likely to be labelled GOOD when values were missing for these variables.

Removing individual high-missingness variables produced relatively small changes in model performance, while removing them collectively reduced cross-validation accuracy. This suggests that information was distributed across combinations of missingness patterns rather than being dependent on a single variable.

## Model Comparison

Random Forest and XGBoost achieved broadly similar baseline performance, but XGBoost retained stronger predictive performance following feature reduction.

Feature-importance and partial-dependence analysis also suggested that XGBoost was able to make useful predictions from variables containing substantial missingness.

## Final Model

Recursive feature elimination identified an **11-feature XGBoost model**:

- `disp_income`
- `cust_age`
- `time_emp`
- `D_01`
- `ER_01`
- `ER_02`
- `S_01`
- `CA_02`
- `CA_01`
- `res_indicator_P`
- `res_indicator_R`

The reduced model achieved approximately **70% cross-validation accuracy**.

## Decision Thresholds

Rather than forcing every observation into GOOD or BAD, predicted probabilities were divided into three decision groups:

- **GOOD:** probability ≥ 0.70
- **BAD:** probability ≤ 0.40
- **PASS:** probability between 0.40 and 0.70

The PASS category represents observations where model confidence is insufficient for an automatic classification.

Using out-of-fold cross-validation predictions, the thresholded model achieved approximately:

- **78.6% accuracy**
- **59% coverage**

This provides a practical trade-off between prediction accuracy and the proportion of cases automatically classified.

## Final Evaluation

The final model and thresholds were evaluated once using an unseen test set.

The thresholded test-set performance was approximately:

- **Accuracy:** 0.78
- **Macro precision:** 0.72
- **Macro recall:** 0.70

The similarity between cross-validation and held-out test performance suggests reasonable generalisation.

## Key Findings

- Missing data should not automatically be treated as uninformative.
- Several highly incomplete variables contained useful predictive information.
- XGBoost performed particularly well in the sparse-data environment.
- Recursive feature elimination substantially reduced the feature set while maintaining predictive performance.
- Probability thresholds improved classification accuracy by separating uncertain cases into a PASS category.
- Missingness itself appeared to contribute information about customer account outcome.

## Tools

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- matplotlib
- Jupyter Notebook

##  Report

The report is available here:

[MissingnessReport.pdf](https://github.com/user-attachments/files/31231267/MissingnessReport.pdf)
