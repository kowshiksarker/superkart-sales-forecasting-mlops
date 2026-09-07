import os
from pathlib import Path

from huggingface_hub import HfApi


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "SuperKart.csv"

HF_DATASET_REPO = "ksarker/superkart-sales-dataset"


# -------------------------------------------------------------------
# Validate input data
# -------------------------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )


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
# Create or verify dataset repository
# -------------------------------------------------------------------

print("=" * 70)
print("SUPERCART DATA REGISTRATION")
print("=" * 70)

print(f"\nDataset:")
print(f" - {DATA_FILE}")

print(f"\nHugging Face repository:")
print(f" - {HF_DATASET_REPO}")

api.create_repo(
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    exist_ok=True
)

print("\nDataset repository ready.")


# -------------------------------------------------------------------
# Upload raw dataset
# -------------------------------------------------------------------

api.upload_file(
    path_or_fileobj=str(DATA_FILE),
    path_in_repo="SuperKart.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    commit_message="Register SuperKart dataset"
)

print("\nRaw dataset uploaded successfully.")

print("\n" + "=" * 70)
print("DATA REGISTRATION : PASSED")
print("=" * 70)
