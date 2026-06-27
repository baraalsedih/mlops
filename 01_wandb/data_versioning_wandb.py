"""
Data versioning with W&B Artifacts — METADATA ONLY approach.

WHY metadata only?
- Real datasets are often large (GBs+) or sensitive (PII, client data)
- Uploading raw data to W&B is inappropriate in those cases
- Instead: log what the data IS (schema, statistics, split sizes, hash)
  so experiments are reproducible without moving the actual data

For large / on-prem datasets, see dvc_demo/ which shows how DVC handles
the actual data files while W&B tracks the experiment results.

Run:
    python data_versioning_wandb.py
"""

import hashlib
import json
import pandas as pd
import wandb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def compute_hash(df: pd.DataFrame) -> str:
    """Deterministic hash of a DataFrame — used as a reproducibility fingerprint."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Load and split (simulating "we have a dataset on our server")
# ---------------------------------------------------------------------------
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])

# ---------------------------------------------------------------------------
# Build a metadata dict — this is what we log, not the raw rows
# ---------------------------------------------------------------------------
metadata = {
    "name": "wine-dataset",
    "version_tag": "v1.0",
    "source": "sklearn.datasets.load_wine",  # replace with actual path on your servers
    "n_samples_total": len(df),
    "n_train": len(train_df),
    "n_test": len(test_df),
    "n_features": len(data.feature_names),
    "feature_names": list(data.feature_names),
    "target_names": list(data.target_names),
    "class_distribution_train": train_df["target"].value_counts().to_dict(),
    "class_distribution_test": test_df["target"].value_counts().to_dict(),
    "hash_train": compute_hash(train_df),
    "hash_test": compute_hash(test_df),
    "random_state": 42,
}

print("Dataset metadata:")
print(json.dumps(metadata, indent=2))

# ---------------------------------------------------------------------------
# Log as a W&B Artifact (metadata only — no CSV upload)
# ---------------------------------------------------------------------------
run = wandb.init(project="mlops-wine-classifier", job_type="data-versioning")

artifact = wandb.Artifact(
    name="wine-dataset-metadata",
    type="dataset",
    description="Metadata fingerprint for the Wine dataset split. Raw data stays on source server.",
    metadata=metadata,
)

# We add a small JSON file as the artifact content (not the raw data)
with open("dataset_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
artifact.add_file("dataset_metadata.json")

run.log_artifact(artifact)
wandb.finish()

print("\nArtifact logged. Raw data was NOT uploaded to W&B.")
print("To reproduce: use hash_train / hash_test to verify you have the same split.")
print("\nFor large/sensitive datasets, use DVC (see dvc_demo/) to version the actual files")
print("while keeping them on your own servers.")
