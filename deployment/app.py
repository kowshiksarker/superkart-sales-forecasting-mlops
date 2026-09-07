# TEMPORARY DEPLOYMENT DIAGNOSTIC
import sys
import importlib.util

print("=" * 70)
print("STREAMLIT DEPLOYMENT ENVIRONMENT")
print("=" * 70)
print("Python:", sys.version)
print("Executable:", sys.executable)
print("joblib spec:", importlib.util.find_spec("joblib"))
print("pandas spec:", importlib.util.find_spec("pandas"))
print("sklearn spec:", importlib.util.find_spec("sklearn"))
print("huggingface_hub spec:", importlib.util.find_spec("huggingface_hub"))
print("=" * 70)

import streamlit as st
import pandas as pd
import joblib

from huggingface_hub import hf_hub_download


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

MODEL_REPO_ID = "ksarker/superkart-sales-regressor"
MODEL_FILENAME = "best_model.joblib"


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="centered"
)


# -------------------------------------------------------------------
# Application title
# -------------------------------------------------------------------

st.title("🛒 SuperKart Sales Prediction")

st.write(
    "Enter the product and store characteristics below to estimate "
    "the expected product-store sales."
)


# -------------------------------------------------------------------
# Load the trained model from Hugging Face Model Hub
# -------------------------------------------------------------------

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model"
    )

    return joblib.load(model_path)


model = load_model()

st.success("Model loaded successfully from Hugging Face Model Hub.")


# -------------------------------------------------------------------
# Prediction input form
# -------------------------------------------------------------------

st.subheader("Prediction Inputs")

product_weight = st.number_input(
    "Product Weight",
    min_value=4.0,
    max_value=22.0,
    value=12.65,
    step=0.01
)

product_sugar_content = st.selectbox(
    "Product Sugar Content",
    [
        "Low Sugar",
        "Regular",
        "No Sugar"
    ]
)

product_allocated_area = st.number_input(
    "Product Allocated Area",
    min_value=0.004,
    max_value=0.298,
    value=0.069,
    step=0.001
)

product_type = st.selectbox(
    "Product Type",
    [
        "Baking Goods",
        "Breads",
        "Breakfast",
        "Canned",
        "Dairy",
        "Frozen Foods",
        "Fruits and Vegetables",
        "Hard Drinks",
        "Health and Hygiene",
        "Household",
        "Meat",
        "Others",
        "Seafood",
        "Snack Foods",
        "Soft Drinks",
        "Starchy Foods"
    ]
)

product_mrp = st.number_input(
    "Product MRP",
    min_value=31.0,
    max_value=266.0,
    value=147.0,
    step=0.1
)

store_id = st.selectbox(
    "Store ID",
    [
        "OUT001",
        "OUT002",
        "OUT003",
        "OUT004"
    ]
)

store_establishment_year = st.number_input(
    "Store Establishment Year",
    min_value=1987,
    max_value=2009,
    value=2002,
    step=1
)

store_size = st.selectbox(
    "Store Size",
    [
        "Small",
        "Medium",
        "High"
    ]
)

store_location_city_type = st.selectbox(
    "Store Location City Type",
    [
        "Tier 1",
        "Tier 2",
        "Tier 3"
    ]
)

store_type = st.selectbox(
    "Store Type",
    [
        "Departmental Store",
        "Food Mart",
        "Supermarket Type1",
        "Supermarket Type2"
    ]
)


# -------------------------------------------------------------------
# Display input summary
# -------------------------------------------------------------------

st.subheader("Selected Inputs")

input_data = pd.DataFrame({
    "Product_Weight": [product_weight],
    "Product_Sugar_Content": [product_sugar_content],
    "Product_Allocated_Area": [product_allocated_area],
    "Product_Type": [product_type],
    "Product_MRP": [product_mrp],
    "Store_Id": [store_id],
    "Store_Establishment_Year": [store_establishment_year],
    "Store_Size": [store_size],
    "Store_Location_City_Type": [store_location_city_type],
    "Store_Type": [store_type]
})

st.dataframe(
    input_data,
    use_container_width=True
)


# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------

if st.button("Predict Sales", type="primary"):

    prediction = model.predict(input_data)[0]

    st.subheader("Predicted Product-Store Sales")

    st.success(
        f"Estimated Sales: {prediction:,.2f}"
    )
