"""Stage 1 — prepare: split raw data into train/test."""

import os
import yaml
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["prepare"]

# In a real project, load from: pd.read_csv("data/wine_raw.csv")
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

train_df, test_df = train_test_split(
    df,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=df["target"],
)

os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
