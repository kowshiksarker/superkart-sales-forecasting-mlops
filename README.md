# SuperKart Sales Forecasting MLOps Project

## Project Overview

This project implements an end-to-end MLOps workflow for the
SuperKart dataset.

The objective is to develop a machine learning regression model
that predicts Product-Store Sales using product and store
characteristics.

> **Important:** The supplied dataset does not contain a date or
> time variable. Therefore, this implementation performs
> product-store sales prediction using regression rather than
> chronological time-series forecasting.

---

## MLOps Workflow

The project follows the following lifecycle:

1. Data Registration
2. Data Preparation
3. Train/Test Split
4. Dataset Registration
5. Model Training
6. Hyperparameter Tuning
7. Model Evaluation
8. Model Registration
9. Application Development
10. Containerized Deployment
11. CI/CD Automation

---

## Project Architecture

```text
                         SuperKart.csv
                              |
                              v
                    Hugging Face Dataset Hub
                              |
                              v
                       Data Preparation
                              |
                       +------+------+
                       |             |
                       v             v
                    train.csv      test.csv
                       |             |
                       +------+------+
                              |
                              v
                       Model Training
                              |
                    Random Forest Regression
                              |
                              v
                       Model Evaluation
                              |
                              v
                    Hugging Face Model Hub
                              |
                       best_model.joblib
                              |
                              v
                        Streamlit App
                              |
                              v
                       Public Deployment
```

---

## Repository Structure

```text
superkart_sales_forecasting_mlops/
|
+-- .github/
|   +-- workflows/
|
+-- data/
|   +-- SuperKart.csv
|   +-- train.csv
|   +-- test.csv
|
+-- model_building/
|   +-- README.md
|   +-- best_model.joblib
|
+-- deployment/
|   +-- Dockerfile
|   +-- README.md
|   +-- app.py
|   +-- host.py
|   +-- requirements.txt
|
+-- .gitignore
+-- README.md
```

---

## Dataset

The dataset contains 8,763 observations and 12 original columns.

The target variable is:

`Product_Store_Sales_Total`

The raw identifier `Product_Id` was removed during data
preparation.

The inconsistent category `reg` in `Product_Sugar_Content` was
normalized to `Regular`.

The cleaned dataset contains 11 columns.

---

## Data Preparation

The cleaned data was split into:

- Training set: 7,010 observations
- Test set: 1,753 observations

The split uses an 80/20 ratio with `random_state=42`.

The train and test datasets were uploaded to the Hugging Face
Dataset Hub.

---

## Model Development

Two regression algorithms were evaluated:

- Random Forest Regression
- Gradient Boosting Regression

Random Forest produced the stronger validation and test
performance.

The final model is implemented as a scikit-learn Pipeline
containing:

1. ColumnTransformer
2. OneHotEncoder for categorical variables
3. Random Forest Regressor

---

## Final Model Performance

### Random Forest

Test-set results:

| Metric | Value |
|---|---:|
| MAE | 105.10 |
| RMSE | 278.11 |
| R² | 0.9322 |

The tuned Random Forest improved the test RMSE from 283.30
to 278.11 compared with the baseline Random Forest.

---

## Registered Model

The trained model is registered in the Hugging Face Model Hub.

Model repository:

`ksarker/superkart-sales-regressor`

Model artifact:

`best_model.joblib`

The deployed application downloads the registered model rather
than requiring the model artifact to be manually packaged with
the app.

---

## Deployment

The application is implemented using Streamlit.

Deployment assets include:

- `app.py`
- `requirements.txt`
- `Dockerfile`
- `README.md`
- `host.py`

The Docker configuration uses Python 3.11 and exposes port 7860.

The application retrieves the registered model from the Hugging
Face Model Hub when it starts.

---

## Deployment Validation

A sample inference test was successfully executed using the
registered model.

Sample input produced:

`Estimated Product-Store Sales = 3,857.48`

The deployment inference test passed successfully.

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Docker
- Hugging Face Dataset Hub
- Hugging Face Model Hub
- GitHub
- GitHub Actions

---

## Reproducibility

The deployment dependencies are explicitly specified in:

`deployment/requirements.txt`

Authentication credentials are supplied through environment
variables or secure platform secrets and are never stored
directly in source code.

---

## Project Status

Current implementation status:

- [x] Data registered on Hugging Face Dataset Hub
- [x] Data cleaned and prepared
- [x] Train/test datasets created
- [x] Train/test datasets registered on Hugging Face
- [x] Candidate models evaluated
- [x] Hyperparameter tuning completed
- [x] Best model selected
- [x] Model registered on Hugging Face Model Hub
- [x] Deployment application created
- [x] Dockerfile created
- [x] Deployment requirements created
- [x] Hosting script created
- [ ] Live application deployment
- [ ] GitHub Actions CI/CD workflow
- [ ] Final automated end-to-end pipeline

---

## Deployment Note

Hugging Face Docker Spaces require a paid plan for the current
account configuration. The Docker deployment assets have
therefore been implemented independently of the live hosting
environment.

The final public Streamlit application will be deployed through
a GitHub-based Streamlit hosting workflow.
