# Recidivism Risk Prediction

Supervised classification of two-year recidivism using criminal-history features. The notebook compares logistic regression and a random forest, calibrates the forest, and scores an example person without treating the output as a verdict.

**Notebook:** [ethical_predictive_model_prison_parole_system.ipynb](ethical_predictive_model_prison_parole_system.ipynb)

## Brief description

This project asks a single question: given prior record, age, charge degree, and related attributes, how likely is two-year reoffending? 
I cleaned mixed-type justice data, encoded categories, scaled numeric fields in a sklearn pipeline, trained two classifiers, and reported 
a **calibrated random-forest probability of 0.58** for a worked example. 

Logistic regression collapsed to a near-zero score on the same row, so I discarded that probability and documented why the models disagreed.

## What I built

- Defined `two_year_recid` as the binary target
- Selected features: `sex`, `age`, `race`, juvenile counts, `priors_count`, `c_charge_degree`
- Handled missing values, date-like fields, and non-numeric columns before modelling
- Compared **unsupervised clustering** (K-Means / hierarchical) with **supervised classification**, then chose classification because a label exists
- Fitted logistic regression and random forest in a `Pipeline` with `ColumnTransformer`, `OneHotEncoder`, and `StandardScaler`
- Evaluated with a stratified train/test split, classification report, ROC AUC, and a calibration curve
- Used `CalibratedClassifierCV` so the example score moved from a raw **0.76** to a calibrated **0.58**
- Result is indicated as “somewhat more likely than not,” and not as certainty

## Outcome

| Model | Example P(reoffend) | How I used it |
|---|---|---|
| Logistic regression | ~0.0004% | Rejected — unusable probability |
| Raw random forest | 76% | Too confident |
| **Calibrated random forest** | **58%** | **Reported estimate** |

The useful result is not a yes/no label. It is a calibrated probability (A statistical estimate, not a parole decision).

## Tools

- Python 3 / Anaconda
- Jupyter Notebook
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (`LogisticRegression`, `RandomForestClassifier`, `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `train_test_split`, `CalibratedClassifierCV`)
- Git / GitHub

## How to run

```bash
git clone https://github.com/PleromaRay/recidivism-risk-prediction.git
cd recidivism-risk-prediction
jupyter notebook ethical_predictive_model_prison_parole_system.ipynb
