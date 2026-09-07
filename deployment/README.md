---
title: SuperKart Sales Prediction
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# SuperKart Sales Prediction

An MLOps-based machine learning application for predicting
Product-Store Sales for SuperKart.

## Application

The application accepts product and store characteristics as input
and uses a trained Random Forest regression pipeline to estimate
Product-Store Sales.

## Model

The model is hosted separately on the Hugging Face Model Hub:

**ksarker/superkart-sales-regressor**

The Streamlit application downloads the registered model from the
Model Hub when the application starts.

## Input Features

The application uses the following ten features:

- Product_Weight
- Product_Sugar_Content
- Product_Allocated_Area
- Product_Type
- Product_MRP
- Store_Id
- Store_Establishment_Year
- Store_Size
- Store_Location_City_Type
- Store_Type

## Model Performance

Test-set performance:

- MAE: 105.10
- RMSE: 278.11
- R²: 0.9322

## Technology Stack

- Python
- Scikit-learn
- Pandas
- Joblib
- Streamlit
- Docker
- Hugging Face Model Hub
- Hugging Face Spaces

## Important Note

The supplied dataset does not contain a time or date variable.
Therefore, this application performs product-store sales prediction
(regression) rather than chronological time-series forecasting.
