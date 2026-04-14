#!/usr/bin/env conda run -n ds python3
"""
Export trained Random Forest model as JSON for the browser demo.
Outputs demo/public/model.json with tree structures + preprocessing params.

Usage:
    conda run -n ds python scripts/export_model.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE / "raw-data"
NULL_VALS = ["NULL", "PrivacySuppressed", "NA", "PS"]

# "STEM Score" fields — STEM + Business (high-earning in Scorecard data)
HIGH_EARN_PCIP_COLS = ["PCIP11", "PCIP14", "PCIP26", "PCIP27", "PCIP40", "PCIP52"]
RAW_NUMERIC_COLS = [
    "UGDS", "ADM_RATE",
    "TUITIONFEE_IN", "TUITIONFEE_OUT", "NPT4_PUB", "NPT4_PRIV",
    "PCTPELL", "PCTFLOAN",
    "UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP", "UGDS_ASIAN",
]
CATEGORICAL_FEATURES = ["CONTROL", "PREDDEG", "HIGHDEG", "LOCALE", "REGION"]
TARGET_COMPONENTS = ["GRAD_DEBT_MDN", "MD_EARN_WNE_P10"]

BASE_NUMERIC_FEATURES = [
    "UGDS_log", "ADM_RATE", "ADM_RATE_missing",
    "TUITIONFEE_OUT", "NPT4",
    "PCTPELL", "PCTFLOAN",
    "UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP", "UGDS_ASIAN",
    "PCIP_reported",
]

TAX_BRACKETS = [
    (25_000, 0.15), (50_000, 0.25), (75_000, 0.29),
    (100_000, 0.35), (150_000, 0.37), (np.inf, 0.40),
]
BASE_LIVING = 24_200
LIFESTYLE_K = 22_000
REPAYMENT_FRAC = 0.50
MIN_PAYMENT = 500
MAX_PAYOFF = 25
LOAN_RATE = 0.055


def compute_tax(income):
    tax = np.zeros(len(income), dtype=float)
    prev = 0
    for limit, rate in TAX_BRACKETS:
        taxable = np.clip(income, prev, limit) - prev
        tax += taxable * rate
        prev = limit
    return tax


def amortize_years(debt, annual_pmt, rate=LOAN_RATE, cap=MAX_PAYOFF):
    r = rate / 12
    m_pmt = annual_pmt / 12
    interest_floor = debt * r
    can_pay = m_pmt > interest_floor
    months = np.full_like(debt, cap * 12, dtype=float)
    months[can_pay] = -np.log(1 - r * debt[can_pay] / m_pmt[can_pay]) / np.log(1 + r)
    return np.minimum(months / 12, cap)


# ── Load & engineer (same as generate_figures.py) ────────────────────

csv_path = sorted(RAW_DATA_DIR.rglob("Most-Recent-Cohorts-Institution.csv"))[0]
load_cols = RAW_NUMERIC_COLS + CATEGORICAL_FEATURES + TARGET_COMPONENTS + HIGH_EARN_PCIP_COLS
df = pd.read_csv(csv_path, usecols=load_cols, na_values=NULL_VALS, low_memory=False)

for col in RAW_NUMERIC_COLS + TARGET_COMPONENTS + HIGH_EARN_PCIP_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=TARGET_COMPONENTS).copy()

gross = df["MD_EARN_WNE_P10"]
after_tax = gross - compute_tax(gross)
living = BASE_LIVING + LIFESTYLE_K * np.log1p(np.maximum(0, after_tax - BASE_LIVING) / BASE_LIVING)
disp = np.maximum(0, after_tax - living)
pmt = np.maximum(MIN_PAYMENT, disp * REPAYMENT_FRAC)
df["payoff"] = np.minimum(amortize_years(df["GRAD_DEBT_MDN"].values, pmt.values), MAX_PAYOFF)

df["NPT4"] = df["NPT4_PUB"].fillna(df["NPT4_PRIV"])
df["UGDS_log"] = np.log1p(df["UGDS"])
df["ADM_RATE_missing"] = df["ADM_RATE"].isna().astype(float)
df["PCIP_reported"] = df[HIGH_EARN_PCIP_COLS].notna().any(axis=1).astype(float)
for col in HIGH_EARN_PCIP_COLS:
    df[col] = df[col].fillna(0)
df["high_earning_share"] = df[HIGH_EARN_PCIP_COLS].sum(axis=1)

for c in CATEGORICAL_FEATURES:
    df[c] = df[c].astype(str)

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ["high_earning_share"]
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df["payoff"]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train RF with known-good hyperparams ────────────────────────────

numeric_tx = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_tx = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_tx, NUMERIC_FEATURES),
    ("cat", categorical_tx, CATEGORICAL_FEATURES),
])

rf = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_leaf=1,
                           random_state=42, n_jobs=-1)
pipe = Pipeline([("prep", preprocessor), ("model", rf)])
pipe.fit(X_train, y_train)
print(f"Trained RF on {len(X_train)} rows")

# ── Extract preprocessing params ────────────────────────────────────

num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
num_scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]

export = {
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "imputer_medians": num_imputer.statistics_.tolist(),
    "scaler_means": num_scaler.mean_.tolist(),
    "scaler_stds": num_scaler.scale_.tolist(),
    "onehot_categories": [cats.tolist() for cats in cat_encoder.categories_],
    "trees": [],
}

# ── Extract tree structures ─────────────────────────────────────────

for i, tree in enumerate(rf.estimators_):
    t = tree.tree_
    export["trees"].append({
        "feature": t.feature.tolist(),
        "threshold": [round(float(x), 6) for x in t.threshold],
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "value": [round(float(v[0][0]), 4) for v in t.value],
    })

# ── Save ────────────────────────────────────────────────────────────

out_dir = BASE / "demo" / "public"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "model.json"
with open(out_path, "w") as f:
    json.dump(export, f)

size_mb = out_path.stat().st_size / 1024 / 1024
print(f"Exported {len(export['trees'])} trees to {out_path} ({size_mb:.1f} MB)")
