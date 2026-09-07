import os
from pathlib import Path

from huggingface_hub import HfApi


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

SPACE_REPO_ID = "ksarker/superkart-sales-prediction"
DEPLOYMENT_DIR = Path(__file__).resolve().parent


# -------------------------------------------------------------------
# Authenticate with Hugging Face
# -------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN environment variable is not set. "
        "Please provide a Hugging Face token through a secure secret."
    )

api = HfApi(token=HF_TOKEN)


# -------------------------------------------------------------------
# Create or verify Hugging Face Space
# -------------------------------------------------------------------

print("=" * 70)
print("HUGGING FACE SPACE DEPLOYMENT")
print("=" * 70)

print(f"\nSpace repository:")
print(f" - {SPACE_REPO_ID}")

print("\nCreating or verifying Docker Space...")

api.create_repo(
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True
)

print(" - Space ready")


# -------------------------------------------------------------------
# Upload deployment files
# -------------------------------------------------------------------

print("\nUploading deployment files...")

api.upload_folder(
    folder_path=str(DEPLOYMENT_DIR),
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    commit_message="Deploy SuperKart sales prediction application"
)

print(" - Deployment files uploaded successfully")


# -------------------------------------------------------------------
# Display Space URL
# -------------------------------------------------------------------

SPACE_URL = f"https://huggingface.co/spaces/{SPACE_REPO_ID}"

print("\n" + "=" * 70)
print("DEPLOYMENT SUBMITTED")
print("=" * 70)

print(f"\nHugging Face Space:")
print(f" - {SPACE_URL}")

print("\nThe Space will now build the Docker image and start the app.")
print("=" * 70)
