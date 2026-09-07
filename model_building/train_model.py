import os
from pathlib import Path

import joblib
import pandas as pd

from huggingface_hub import HfApi, hf_hub_download

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HF_DATASET_REPO = "ksarker/superkart-sales-dataset"
HF_MODEL_REPO = "ksarker/superkart-sales-regressor"

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"

TARGET_COLUMN = "Product_Store_Sales_Total"

RANDOM_STATE = 42


# -------------------------------------------------------------------
# Authenticate with Hugging Face
# -------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN environment variable is not set."
    )

api = HfApi(token=HF_TOKEN)


# -------------------------------------------------------------------
# Download train/test data directly from Hugging Face
# -------------------------------------------------------------------

print("=" * 70)
print("SUPERCART MODEL TRAINING")
print("=" * 70)

print("\nDownloading train/test datasets from Hugging Face...")

train_path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename=TRAIN_FILENAME,
    repo_type="dataset",
    token=HF_TOKEN
)

test_path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename=TEST_FILENAME,
    repo_type="dataset",
    token=HF_TOKEN
)

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print(f" - Train shape: {train_data.shape}")
print(f" - Test shape:  {test_data.shape}")


# -------------------------------------------------------------------
# Separate features and target
# -------------------------------------------------------------------

if TARGET_COLUMN not in train_data.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found in training data."
    )

if TARGET_COLUMN not in test_data.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found in test data."
    )

X_train = train_data.drop(columns=[TARGET_COLUMN])
y_train = train_data[TARGET_COLUMN]

X_test = test_data.drop(columns=[TARGET_COLUMN])
y_test = test_data[TARGET_COLUMN]


# -------------------------------------------------------------------
# Define feature types
# -------------------------------------------------------------------

numerical_features = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year"
]

categorical_features = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]

all_features = numerical_features + categorical_features

if set(all_features) != set(X_train.columns):
    raise ValueError(
        "Feature definitions do not match the training dataset."
    )

if len(all_features) != len(set(all_features)):
    raise ValueError(
        "Duplicate feature names detected."
    )


# -------------------------------------------------------------------
# Create preprocessing pipeline
# -------------------------------------------------------------------

def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                "passthrough",
                numerical_features
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            )
        ]
    )


# -------------------------------------------------------------------
# Baseline Random Forest
# -------------------------------------------------------------------

baseline_rf = Pipeline([
    (
        "preprocessor",
        create_preprocessor()
    ),
    (
        "model",
        RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
])

baseline_rf.fit(X_train, y_train)

baseline_predictions = baseline_rf.predict(X_test)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = mean_squared_error(
    y_test,
    baseline_predictions
) ** 0.5

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)

print("\nBaseline Random Forest:")
print(f" - MAE:  {baseline_mae:.2f}")
print(f" - RMSE: {baseline_rmse:.2f}")
print(f" - R2:   {baseline_r2:.4f}")


# -------------------------------------------------------------------
# Baseline Gradient Boosting
# -------------------------------------------------------------------

baseline_gb = Pipeline([
    (
        "preprocessor",
        create_preprocessor()
    ),
    (
        "model",
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    )
])

baseline_gb.fit(X_train, y_train)

gb_predictions = baseline_gb.predict(X_test)

gb_mae = mean_absolute_error(
    y_test,
    gb_predictions
)

gb_rmse = mean_squared_error(
    y_test,
    gb_predictions
) ** 0.5

gb_r2 = r2_score(
    y_test,
    gb_predictions
)

print("\nBaseline Gradient Boosting:")
print(f" - MAE:  {gb_mae:.2f}")
print(f" - RMSE: {gb_rmse:.2f}")
print(f" - R2:   {gb_r2:.4f}")


# -------------------------------------------------------------------
# Random Forest hyperparameter tuning
# -------------------------------------------------------------------

print("\nTuning Random Forest...")

rf_pipeline = Pipeline([
    (
        "preprocessor",
        create_preprocessor()
    ),
    (
        "model",
        RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
])

rf_param_grid = {
    "model__n_estimators": [200],
    "model__max_depth": [None, 15],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

rf_grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=1
)

rf_grid_search.fit(X_train, y_train)

best_rf_cv_rmse = -rf_grid_search.best_score_

print("\nBest Random Forest parameters:")
print(rf_grid_search.best_params_)

print(f"Best RF CV RMSE: {best_rf_cv_rmse:.2f}")


# -------------------------------------------------------------------
# Gradient Boosting hyperparameter tuning
# -------------------------------------------------------------------

print("\nTuning Gradient Boosting...")

gb_pipeline = Pipeline([
    (
        "preprocessor",
        create_preprocessor()
    ),
    (
        "model",
        GradientBoostingRegressor(
            random_state=RANDOM_STATE
        )
    )
])

gb_param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.10],
    "model__max_depth": [2, 3]
}

gb_grid_search = GridSearchCV(
    estimator=gb_pipeline,
    param_grid=gb_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=1
)

gb_grid_search.fit(X_train, y_train)

best_gb_cv_rmse = -gb_grid_search.best_score_

print("\nBest Gradient Boosting parameters:")
print(gb_grid_search.best_params_)

print(f"Best GB CV RMSE: {best_gb_cv_rmse:.2f}")


# -------------------------------------------------------------------
# Select best model using CV RMSE
# -------------------------------------------------------------------

if best_rf_cv_rmse <= best_gb_cv_rmse:

    best_model = rf_grid_search.best_estimator_
    best_model_name = "Random Forest"
    best_params = rf_grid_search.best_params_
    best_cv_rmse = best_rf_cv_rmse

else:

    best_model = gb_grid_search.best_estimator_
    best_model_name = "Gradient Boosting"
    best_params = gb_grid_search.best_params_
    best_cv_rmse = best_gb_cv_rmse


print("\nSelected model:")
print(f" - {best_model_name}")
print(f" - CV RMSE: {best_cv_rmse:.2f}")


# -------------------------------------------------------------------
# Evaluate selected model on test data
# -------------------------------------------------------------------

test_predictions = best_model.predict(X_test)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = mean_squared_error(
    y_test,
    test_predictions
) ** 0.5

test_r2 = r2_score(
    y_test,
    test_predictions
)

print("\nFinal test performance:")
print(f" - MAE:  {test_mae:.2f}")
print(f" - RMSE: {test_rmse:.2f}")
print(f" - R2:   {test_r2:.4f}")


# -------------------------------------------------------------------
# Save best model locally
# -------------------------------------------------------------------

model_dir = PROJECT_ROOT / "model_building"
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "best_model.joblib"

joblib.dump(
    best_model,
    model_path
)

print(f"\nModel saved:")
print(f" - {model_path}")


# -------------------------------------------------------------------
# Create model card
# -------------------------------------------------------------------

model_readme = (
    "---\n"
    "library_name: scikit-learn\n"
    "tags:\n"
    "- tabular-regression\n"
    "- random-forest\n"
    "- superkart\n"
    "- mlops\n"
    "---\n"
    "\n"
    "# SuperKart Sales Regression Model\n"
    "\n"
    f"## Model\n\nSelected model: **{best_model_name}**\n"
    "\n"
    "The model is implemented as a scikit-learn Pipeline containing "
    "feature preprocessing and the selected regression estimator.\n"
    "\n"
    "## Cross-Validation Performance\n"
    f"\nCV RMSE: **{best_cv_rmse:.2f}**\n"
    "\n"
    "## Test Performance\n"
    f"\n- MAE: **{test_mae:.2f}**\n"
    f"\n- RMSE: **{test_rmse:.2f}**\n"
    f"\n- R2: **{test_r2:.4f}**\n"
    "\n"
    "## Hyperparameters\n"
    f"\n`{best_params}`\n"
    "\n"
    "## Numerical Features\n"
    "\n"
    "- Product_Weight\n"
    "- Product_Allocated_Area\n"
    "- Product_MRP\n"
    "- Store_Establishment_Year\n"
    "\n"
    "## Categorical Features\n"
    "\n"
    "- Product_Sugar_Content\n"
    "- Product_Type\n"
    "- Store_Id\n"
    "- Store_Size\n"
    "- Store_Location_City_Type\n"
    "- Store_Type\n"
    "\n"
    "## Target\n"
    "\n"
    "`Product_Store_Sales_Total`\n"
    "\n"
    "## Dataset Note\n"
    "\n"
    "The supplied dataset does not contain a date or time variable. "
    "Therefore, this model performs product-store sales prediction "
    "using regression rather than chronological time-series "
    "forecasting.\n"
)

model_readme_path = model_dir / "README.md"

with open(
    model_readme_path,
    "w",
    encoding="utf-8"
) as file:
    file.write(model_readme)


# -------------------------------------------------------------------
# Register model on Hugging Face Model Hub
# -------------------------------------------------------------------

print("\nUploading best model to Hugging Face Model Hub...")

api.create_repo(
    repo_id=HF_MODEL_REPO,
    repo_type="model",
    exist_ok=True
)

api.upload_file(
    path_or_fileobj=str(model_path),
    path_in_repo="best_model.joblib",
    repo_id=HF_MODEL_REPO,
    repo_type="model",
    commit_message="Register updated SuperKart regression model"
)

api.upload_file(
    path_or_fileobj=str(model_readme_path),
    path_in_repo="README.md",
    repo_id=HF_MODEL_REPO,
    repo_type="model",
    commit_message="Update SuperKart model card"
)

print(" - Model uploaded successfully")
print(" - Model card uploaded successfully")


# -------------------------------------------------------------------
# Completion
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL TRAINING AND REGISTRATION : PASSED")
print("=" * 70)
