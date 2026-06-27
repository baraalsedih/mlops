"""
W&B Model Registry demo.

Shows:
1. How to link a logged model artifact to the W&B Model Registry
2. How to promote a version to the "production" alias
3. How a deployment script pulls the production model by alias (not by version number)

Prerequisites:
    Run train_with_wandb.py first so the artifact exists.

Run:
    python model_registry.py
"""

import joblib
import wandb

# ---------------------------------------------------------------------------
# 1. Fetch the latest logged model artifact from the project
# ---------------------------------------------------------------------------
print("=== Step 1: Fetch latest model artifact ===")

run = wandb.init(project="mlops-wine-classifier", job_type="model-promotion")

# Use the artifact logged by train_with_wandb.py
artifact = run.use_artifact("wine-classifier:latest", type="model")
artifact_dir = artifact.download()
print(f"Downloaded artifact to: {artifact_dir}")

model = joblib.load(f"{artifact_dir}/model.pkl")
print(f"Model loaded: {model}")
print(f"Artifact metadata: {artifact.metadata}")

# ---------------------------------------------------------------------------
# 2. Promote to "production" alias in the Model Registry
#
#    In the W&B UI this is done via:
#      Registry → Models → wine-classifier → version → Add alias → production
#
#    Programmatically (requires org-level registry setup):
# ---------------------------------------------------------------------------
print("\n=== Step 2: Promote to production ===")

# Link artifact to the Model Registry collection
wandb.run.link_artifact(
    artifact,
    target_path="mlops-wine-classifier/model-registry/wine-classifier",
    aliases=["production"],
)

print("Artifact linked to registry with alias 'production'.")
print("Any downstream system can now pull 'wine-classifier:production'")
print("without knowing the exact version number.")

wandb.finish()

# ---------------------------------------------------------------------------
# 3. Simulate a deployment script pulling by alias
# ---------------------------------------------------------------------------
print("\n=== Step 3: Deployment — pull by alias, not version ===")

deploy_run = wandb.init(
    project="mlops-wine-classifier", job_type="deployment"
)

# This will always resolve to the current "production" model,
# regardless of which version number is behind the alias.
prod_artifact = deploy_run.use_artifact(
    "wine-classifier:production", type="model"
)
prod_dir = prod_artifact.download()
prod_model = joblib.load(f"{prod_dir}/model.pkl")

print(f"Loaded production model (v{prod_artifact.version}): {prod_model}")
print("\nTo roll back: simply reassign the 'production' alias in the registry UI.")

wandb.finish()
