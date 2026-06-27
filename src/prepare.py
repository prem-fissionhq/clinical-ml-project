# src/prepare.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
# Read raw data
df = pd.read_csv("data/patient_outcomes.csv")
# Basic cleaning
df = df.dropna()
df["bmi"] = df["bmi"].clip(15, 50)
# One-hot encode treatment group
df = pd.get_dummies(df, columns=["treatment_group"])
# Split
train, test = train_test_split(df, test_size=0.2, random_state=42,
    stratify=df["outcome"])
# Save prepared data
os.makedirs("data/prepared", exist_ok=True)
train.to_csv("data/prepared/train.csv", index=False)
test.to_csv("data/prepared/test.csv", index=False)
print(f"Train: {len(train)}, Test: {len(test)}")