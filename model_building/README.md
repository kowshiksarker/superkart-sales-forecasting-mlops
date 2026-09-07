
# SuperKart Sales Prediction Model

## Model Overview

This repository contains the trained machine learning pipeline for
predicting `Product_Store_Sales_Total` for the SuperKart dataset.

The model is a Random Forest regression pipeline containing:

1. Numerical feature passthrough
2. One-hot encoding for categorical features
3. Random Forest regression

The preprocessing and trained model are stored together in the
`best_model.joblib` artifact.

## Dataset

Hugging Face Dataset Repository:

ksarker/superkart-sales-dataset

Training observations: 7,010
Testing observations: 1,753

## Features

### Numerical Features

- Product_Weight
- Product_Allocated_Area
- Product_MRP
- Store_Establishment_Year

### Categorical Features

- Product_Sugar_Content
- Product_Type
- Store_Id
- Store_Size
- Store_Location_City_Type
- Store_Type

Target:

`Product_Store_Sales_Total`

## Model Selection

Two ensemble regression algorithms were evaluated:

- Random Forest Regressor
- Gradient Boosting Regressor

Three-fold cross-validation was used for hyperparameter tuning on
the training dataset.

Random Forest achieved the lowest cross-validation RMSE.

### Cross-Validation Results

| Model | Best CV RMSE |
|---|---:|
| Random Forest | 285.89 |
| Gradient Boosting | 308.60 |

## Selected Random Forest Parameters

- n_estimators: 200
- max_depth: None
- min_samples_split: 5
- min_samples_leaf: 2
- random_state: 42

## Final Test Performance

The final model was evaluated on the untouched test dataset.

| Metric | Result |
|---|---:|
| MAE | 105.10 |
| RMSE | 278.11 |
| R² | 0.9322 |

## Artifact

`best_model.joblib` contains the complete fitted preprocessing and
Random Forest pipeline required for inference.

## Important Dataset Note

The available dataset does not contain a date or time-period variable.
Therefore, this model predicts product-store sales rather than performing
a chronological time-series forecast.
