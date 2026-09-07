import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HF_DATASET_REPO = "ksarker/superkart-sales-dataset"

RAW_DATA_FILENAME = "SuperKart.csv"

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"

TARGET_COLUMN = "Product_Store_Sales_Total"

RANDOM_STATE = 42
TEST_SIZE = 0.20


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
# Download raw dataset directly from Hugging Face
# -------------------------------------------------------------------

print("=" * 70)
print("SUPERCART DATA PREPARATION")
print("=" * 70)

print("\nDownloading raw dataset from Hugging Face...")

raw_data_path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename=RAW_DATA_FILENAME,
    repo_type="dataset",
    token=HF_TOKEN
)

df = pd.read_csv(raw_data_path)

print(f" - Loaded shape: {df.shape}")


# -------------------------------------------------------------------
# Validate raw dataset
# -------------------------------------------------------------------

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found."
    )

if df.empty:
    raise ValueError("Raw dataset is empty.")


# -------------------------------------------------------------------
# Clean dataset
# -------------------------------------------------------------------

df_clean = df.copy()

# Normalize inconsistent sugar-content category
df_clean["Product_Sugar_Content"] = (
    df_clean["Product_Sugar_Content"]
    .replace({"reg": "Regular"})
)

# Remove identifier column
if "Product_Id" in df_clean.columns:
    df_clean = df_clean.drop(columns=["Product_Id"])


# -------------------------------------------------------------------
# Validate cleaned dataset
# -------------------------------------------------------------------

if df_clean.isnull().sum().sum() != 0:
    raise ValueError(
        "Missing values detected after data cleaning."
    )

if df_clean.duplicated().sum() != 0:
    raise ValueError(
        "Duplicate rows detected after data cleaning."
    )

if df_clean[TARGET_COLUMN].isnull().any():
    raise ValueError(
        "Missing target values detected."
    )

print(f"\nCleaned shape: {df_clean.shape}")
print(" - Missing values: 0")
print(" - Duplicate rows: 0")
print(" - Product_Id removed")
print(" - 'reg' normalized to 'Regular'")


# -------------------------------------------------------------------
# Train/test split
# -------------------------------------------------------------------

train_df, test_df = train_test_split(
    df_clean,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("\nTrain/test split:")
print(f" - Train shape: {train_df.shape}")
print(f" - Test shape:  {test_df.shape}")


# -------------------------------------------------------------------
# Save locally
# -------------------------------------------------------------------

data_dir = PROJECT_ROOT / "data"
data_dir.mkdir(parents=True, exist_ok=True)

train_path = data_dir / TRAIN_FILENAME
test_path = data_dir / TEST_FILENAME

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("\nLocal files created:")
print(f" - {train_path}")
print(f" - {test_path}")


# -------------------------------------------------------------------
# Validate train/test datasets
# -------------------------------------------------------------------

if list(train_df.columns) != list(test_df.columns):
    raise ValueError(
        "Train and test columns do not match."
    )

if TARGET_COLUMN not in train_df.columns:
    raise ValueError(
        "Target column missing from training data."
    )

if len(train_df) + len(test_df) != len(df_clean):
    raise ValueError(
        "Train/test row counts do not match cleaned dataset."
    )


# -------------------------------------------------------------------
# Upload train/test datasets to Hugging Face
# -------------------------------------------------------------------

print("\nUploading train/test datasets to Hugging Face...")

api.upload_file(
    path_or_fileobj=str(train_path),
    path_in_repo=TRAIN_FILENAME,
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    commit_message="Update training dataset"
)

api.upload_file(
    path_or_fileobj=str(test_path),
    path_in_repo=TEST_FILENAME,
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    commit_message="Update test dataset"
)


# -------------------------------------------------------------------
# Remote verification
# -------------------------------------------------------------------

remote_train = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename=TRAIN_FILENAME,
    repo_type="dataset",
    token=HF_TOKEN,
    force_download=True
)

remote_test = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename=TEST_FILENAME,
    repo_type="dataset",
    token=HF_TOKEN,
    force_download=True
)

remote_train_df = pd.read_csv(remote_train)
remote_test_df = pd.read_csv(remote_test)

if remote_train_df.shape != train_df.shape:
    raise ValueError(
        "Remote training dataset shape does not match local dataset."
    )

if remote_test_df.shape != test_df.shape:
    raise ValueError(
        "Remote test dataset shape does not match local dataset."
    )

print(" - Remote train dataset verified")
print(" - Remote test dataset verified")


# -------------------------------------------------------------------
# Completion
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA PREPARATION : PASSED")
print("=" * 70)
