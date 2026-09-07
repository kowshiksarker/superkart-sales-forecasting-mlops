---
library_name: scikit-learn
tags:
- tabular-regression
- random-forest
- superkart
- mlops
---

# SuperKart Sales Regression Model

## Model

Selected model: **Random Forest**

The model is implemented as a scikit-learn Pipeline containing feature preprocessing and the selected regression estimator.

## Cross-Validation Performance

CV RMSE: **285.89**

## Test Performance

- MAE: **105.10**

- RMSE: **278.11**

- R2: **0.9322**

## Hyperparameters

`{'model__max_depth': None, 'model__min_samples_leaf': 2, 'model__min_samples_split': 5, 'model__n_estimators': 200}`

## Numerical Features

- Product_Weight
- Product_Allocated_Area
- Product_MRP
- Store_Establishment_Year

## Categorical Features

- Product_Sugar_Content
- Product_Type
- Store_Id
- Store_Size
- Store_Location_City_Type
- Store_Type

## Target

`Product_Store_Sales_Total`

## Dataset Note

The supplied dataset does not contain a date or time variable. Therefore, this model performs product-store sales prediction using regression rather than chronological time-series forecasting.
