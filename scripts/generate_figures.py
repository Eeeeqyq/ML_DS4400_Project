#!/usr/bin/env conda run -n ds python3
"""
Generate all EDA, model, and analysis figures for the DS4400 Final Project.

Usage:
    conda run -n ds python scripts/generate_figures.py

Outputs PNGs to figures/ and model_results.json to the repo root.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE / "raw-data"


def _find_scorecard_csv(raw_data_dir: Path) -> Path:
    candidates = sorted(raw_data_dir.rglob("Most-Recent-Cohorts-Institution.csv"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find 'Most-Recent-Cohorts-Institution.csv'.\n"
            f"Expected it somewhere under: {raw_data_dir}\n"
            "Download from collegescorecard.ed.gov/data and place in raw-data/."
        )
    return candidates[0]


DATA = _find_scorecard_csv(RAW_DATA_DIR)
SCRIPTS_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

NULL_VALS = ["NULL", "PrivacySuppressed", "NA", "PS"]

# ── Feature definitions ─────────────────────────────────────────────

# "STEM Score" fields — STEM + Business (included because it is consistently
# high-earning in the Scorecard data). Summed into high_earning_share.
HIGH_EARN_PCIP_COLS = [
    "PCIP11",  # Computer & Information Sciences
    "PCIP14",  # Engineering
    "PCIP26",  # Biological & Biomedical Sciences
    "PCIP27",  # Mathematics & Statistics
    "PCIP40",  # Physical Sciences
    "PCIP52",  # Business, Management & Marketing
]

BASE_NUMERIC_FEATURES = [
    "UGDS_log",          # log1p(UGDS) — raw skew=7.7
    "ADM_RATE",
    "ADM_RATE_missing",  # 63% missing; absence is informative
    "TUITIONFEE_OUT",    # dropped TUITIONFEE_IN (r=0.933 duplicate)
    "NPT4",             # merged NPT4_PUB + NPT4_PRIV (mutually exclusive, 0 overlap)
    "PCTPELL", "PCTFLOAN",
    "UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP", "UGDS_ASIAN",
    "PCIP_reported",     # 1 if any PCIP column was non-NaN in raw data
]
CATEGORICAL_FEATURES = ["CONTROL", "PREDDEG", "HIGHDEG", "LOCALE", "REGION"]
TARGET_COMPONENTS = ["GRAD_DEBT_MDN", "MD_EARN_WNE_P10"]

# Human-readable labels for figures (availability, correlation heatmap, axes, importances)
DISPLAY_NAMES = {
    "UGDS": "Undergrad enrollment",
    "UGDS_log": "Undergrad enrollment (log)",
    "ADM_RATE": "Admission rate",
    "ADM_RATE_missing": "Admission rate missing",
    "SAT_AVG": "Average SAT",
    "TUITIONFEE_IN": "In-state tuition",
    "TUITIONFEE_OUT": "Out-of-state tuition",
    "NPT4_PUB": "Net price (4-yr, public)",
    "NPT4_PRIV": "Net price (4-yr, private)",
    "NPT4": "Net price (4-yr)",
    "PCTPELL": "% students on Pell",
    "PCTFLOAN": "% students with federal loan",
    "UGDS_WHITE": "% undergrad White",
    "UGDS_BLACK": "% undergrad Black",
    "UGDS_HISP": "% undergrad Hispanic",
    "UGDS_ASIAN": "% undergrad Asian",
    "PCIP_reported": "Major data reported",
    "high_earning_share": "STEM Score",
    "CONTROL": "Institution type (control)",
    "PREDDEG": "Predominant credential",
    "HIGHDEG": "Highest degree awarded",
    "LOCALE": "Locale (urban / rural)",
    "REGION": "Census region",
    "GRAD_DEBT_MDN": "Median graduate debt",
    "MD_EARN_WNE_P10": "Median earnings at 10 yrs",
    "estimated_payoff_years": "Estimated payoff (years)",
}

TARGET_LABEL = DISPLAY_NAMES["estimated_payoff_years"]


CATEGORY_VALUE_LABELS = {
    "CONTROL": {"1": "Public", "2": "Private Nonprofit", "3": "For-Profit"},
    "PREDDEG": {"0": "Unclassified", "1": "Certificate", "2": "Associate's",
                "3": "Bachelor's", "4": "Graduate"},
    "HIGHDEG": {"0": "Non-degree", "1": "Certificate", "2": "Associate's",
                "3": "Bachelor's", "4": "Graduate"},
    "LOCALE": {
        "11.0": "City: Large", "12.0": "City: Midsize", "13.0": "City: Small",
        "21.0": "Suburb: Large", "22.0": "Suburb: Midsize", "23.0": "Suburb: Small",
        "31.0": "Town: Fringe", "32.0": "Town: Distant", "33.0": "Town: Remote",
        "41.0": "Rural: Fringe", "42.0": "Rural: Distant", "43.0": "Rural: Remote",
    },
    "REGION": {
        "0": "US Service Schools", "1": "New England", "2": "Mid East",
        "3": "Great Lakes", "4": "Plains", "5": "Southeast",
        "6": "Southwest", "7": "Rocky Mountains", "8": "Far West", "9": "Outlying Areas",
    },
}


def display_feature_name(raw: str) -> str:
    """Map raw pipeline feature names (numeric + one-hot) to display labels."""
    if raw in DISPLAY_NAMES:
        return DISPLAY_NAMES[raw]
    for cat in CATEGORICAL_FEATURES:
        prefix = cat + "_"
        if raw.startswith(prefix):
            suffix = raw[len(prefix):]
            val_labels = CATEGORY_VALUE_LABELS.get(cat, {})
            label = val_labels.get(suffix, suffix)
            return f"{DISPLAY_NAMES[cat]}: {label}"
    return raw


# ── Target engineering constants ────────────────────────────────────

TAX_BRACKETS = [
    ( 25_000, 0.15),
    ( 50_000, 0.25),
    ( 75_000, 0.29),
    (100_000, 0.35),
    (150_000, 0.37),
    (np.inf,  0.40),
]

BASE_LIVING = 24_200      # BLS CE 2022 per-capita floor (fitted to quintile data)
LIFESTYLE_K = 22_000      # log1p scaling factor (fitted to BLS per-capita expenditures)
REPAYMENT_FRAC = 0.50
MIN_PAYMENT = 500
MAX_PAYOFF = 25            # federal IDR forgiveness ceiling (34 CFR 685.209)
LOAN_RATE = 0.055         # federal student loan rate (~5.5% historical avg)


def compute_tax(income: pd.Series) -> np.ndarray:
    """Marginal-bracket tax: each slice of income is taxed at its own rate."""
    tax = np.zeros(len(income), dtype=float)
    prev_limit = 0
    for limit, rate in TAX_BRACKETS:
        taxable = np.clip(income, prev_limit, limit) - prev_limit
        tax += taxable * rate
        prev_limit = limit
    return tax


def amortize_years(debt: np.ndarray, annual_pmt: np.ndarray,
                   rate: float = LOAN_RATE, cap: float = MAX_PAYOFF) -> np.ndarray:
    """Years to pay off debt with compound interest (standard amortization)."""
    r = rate / 12
    m_pmt = annual_pmt / 12
    interest_floor = debt * r
    can_pay = m_pmt > interest_floor
    months = np.full_like(debt, cap * 12, dtype=float)
    months[can_pay] = (
        -np.log(1 - r * debt[can_pay] / m_pmt[can_pay])
        / np.log(1 + r)
    )
    return np.minimum(months / 12, cap)


# ── Load & engineer ─────────────────────────────────────────────────

RAW_NUMERIC_COLS = [
    "UGDS", "ADM_RATE",
    "TUITIONFEE_IN", "TUITIONFEE_OUT", "NPT4_PUB", "NPT4_PRIV",
    "PCTPELL", "PCTFLOAN",
    "UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP", "UGDS_ASIAN",
]
load_cols = (
    RAW_NUMERIC_COLS + CATEGORICAL_FEATURES + TARGET_COMPONENTS
    + HIGH_EARN_PCIP_COLS + ["SAT_AVG"]
)
df = pd.read_csv(DATA, usecols=load_cols, na_values=NULL_VALS, low_memory=False)

# Convert target components to numeric and drop rows where either is missing
# (amortize_years silently maps NaN to cap, so we must filter BEFORE engineering)
for tc in TARGET_COMPONENTS:
    df[tc] = pd.to_numeric(df[tc], errors="coerce")
n_before = len(df)
df = df.dropna(subset=TARGET_COMPONENTS).copy()
n_dropped = n_before - len(df)
print(f"Rows loaded: {n_before:,} | Dropped (missing debt/earnings): {n_dropped:,} "
      f"({n_dropped/n_before*100:.1f}%) | Remaining: {len(df):,}")

# Engineer target
gross = df["MD_EARN_WNE_P10"]
after_tax = gross - compute_tax(gross)
df["living_cost"] = BASE_LIVING + LIFESTYLE_K * np.log1p(
    np.maximum(0, after_tax - BASE_LIVING) / BASE_LIVING
)
df["annual_disposable"] = np.maximum(0, after_tax - df["living_cost"])
df["annual_payment"] = np.maximum(MIN_PAYMENT, df["annual_disposable"] * REPAYMENT_FRAC)
df["estimated_payoff_years"] = amortize_years(
    df["GRAD_DEBT_MDN"].values, df["annual_payment"].values
)
df["estimated_payoff_years"] = df["estimated_payoff_years"].clip(upper=MAX_PAYOFF)

# ── Feature engineering ──────────────────────────────────────────────

# Convert remaining raw numeric columns
for col in RAW_NUMERIC_COLS + ["SAT_AVG"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# D2: Merge NPT4_PUB + NPT4_PRIV (mutually exclusive, 0 overlap)
df["NPT4"] = df["NPT4_PUB"].fillna(df["NPT4_PRIV"])

# D4: Log-transform UGDS (skew=7.7)
df["UGDS_log"] = np.log1p(df["UGDS"])

# D5: ADM_RATE missingness indicator (63% missing)
df["ADM_RATE_missing"] = df["ADM_RATE"].isna().astype(float)
df["PCIP_reported"] = df[HIGH_EARN_PCIP_COLS].notna().any(axis=1).astype(float)
for col in HIGH_EARN_PCIP_COLS:
    df[col] = df[col].fillna(0)
df["high_earning_share"] = df[HIGH_EARN_PCIP_COLS].sum(axis=1)

for c in CATEGORICAL_FEATURES:
    df[c] = df[c].astype(str)

# Control labels (used in several figures)
control_map = {"1": "Public", "2": "Private Nonprofit", "3": "For-Profit"}
df["CONTROL_label"] = df["CONTROL"].map(control_map).fillna("Other")

# Features WITH and WITHOUT high_earning_share
NUMERIC_WITH_STEM = BASE_NUMERIC_FEATURES + ["high_earning_share"]
NUMERIC_WITHOUT_STEM = BASE_NUMERIC_FEATURES

# =====================================================================
# EDA FIGURES
# =====================================================================

# ── Figure 1: Target distribution ───────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["estimated_payoff_years"], bins=50, edgecolor="white", alpha=0.85)
med = df["estimated_payoff_years"].median()
mn = df["estimated_payoff_years"].mean()
ax.axvline(med, color="red", ls="--", label=f"median = {med:.1f}")
ax.axvline(mn, color="orange", ls="--", label=f"mean = {mn:.1f}")
ax.set_xlabel(TARGET_LABEL)
ax.set_ylabel("Count")
ax.set_title("Distribution of Estimated College Debt Payoff Time")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "01_target_distribution.png", dpi=150)
plt.close(fig)
print("Saved 01_target_distribution.png")

# ── Figure 2: Feature availability ──────────────────────────────────

all_features = NUMERIC_WITH_STEM + CATEGORICAL_FEATURES
avail = {f: df[f].notna().mean() * 100 for f in all_features}
avail_s = pd.Series(avail).sort_values()
avail_labeled = avail_s.rename(lambda c: DISPLAY_NAMES.get(c, c))

fig, ax = plt.subplots(figsize=(9, 7))
avail_labeled.plot.barh(ax=ax, color="steelblue", edgecolor="white", linewidth=0.5)
ax.set_xlim(0, 105)
ax.set_xlabel("% of institutions with non-missing value")
ax.set_title(f"Feature availability in the modeling cohort (n = {len(df):,})")
ax.grid(axis="x", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG_DIR / "02_feature_availability.png", dpi=150)
plt.close(fig)
print("Saved 02_feature_availability.png")

# ── Figure 3: Correlation heatmap ───────────────────────────────────

corr_cols = NUMERIC_WITH_STEM + TARGET_COMPONENTS + ["estimated_payoff_years"]
corr = df[corr_cols].corr()
corr = corr.rename(index=lambda c: DISPLAY_NAMES.get(c, c), columns=lambda c: DISPLAY_NAMES.get(c, c))

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    ax=ax,
    annot_kws={"size": 7},
    linewidths=0.4,
    linecolor="white",
)
ax.set_title(f"Pearson correlation matrix (n = {len(df):,})")
plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(FIG_DIR / "03_correlation_heatmap.png", dpi=150)
plt.close(fig)
print("Saved 03_correlation_heatmap.png")

# ── Figure 4: Payoff by institution type ────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
order = ["Public", "Private Nonprofit", "For-Profit"]
existing = [o for o in order if o in df["CONTROL_label"].values]
sns.boxplot(data=df, x="CONTROL_label", y="estimated_payoff_years",
            order=existing, ax=ax)
ax.set_xlabel(DISPLAY_NAMES["CONTROL"])
ax.set_ylabel(TARGET_LABEL)
ax.set_title("Payoff years by institution type")
fig.tight_layout()
fig.savefig(FIG_DIR / "04_payoff_by_control.png", dpi=150)
plt.close(fig)
print("Saved 04_payoff_by_control.png")

# ── Figure 5: Tuition vs payoff ────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
for label, color in zip(existing, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
    sub = df[df["CONTROL_label"] == label]
    ax.scatter(sub["TUITIONFEE_OUT"], sub["estimated_payoff_years"],
               alpha=0.35, s=12, label=label, color=color)
ax.set_xlabel(f"{DISPLAY_NAMES['TUITIONFEE_OUT']} ($)")
ax.set_ylabel(TARGET_LABEL)
ax.set_title("Tuition vs. estimated payoff time")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "05_tuition_vs_payoff.png", dpi=150)
plt.close(fig)
print("Saved 05_tuition_vs_payoff.png")

# ── Figure 6: SAT reporters vs non-reporters ────────────────────────

sat_reported = df["SAT_AVG"].notna()
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df.loc[sat_reported, "estimated_payoff_years"], bins=40, alpha=0.6,
        label=f"SAT reported (n={sat_reported.sum()})", edgecolor="white")
ax.hist(df.loc[~sat_reported, "estimated_payoff_years"], bins=40, alpha=0.6,
        label=f"SAT not reported (n={(~sat_reported).sum()})", edgecolor="white")
ax.set_xlabel(TARGET_LABEL)
ax.set_ylabel("Count")
ax.set_title("Payoff distribution: SAT reporters vs. non-reporters")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "06_sat_reporters.png", dpi=150)
plt.close(fig)
print("Saved 06_sat_reporters.png")

# ── Figure 7: High-earning field share vs payoff ──────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
for label, color in zip(existing, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
    sub = df[df["CONTROL_label"] == label]
    ax.scatter(sub["high_earning_share"], sub["estimated_payoff_years"],
               alpha=0.35, s=12, label=label, color=color)
ax.set_xlabel(DISPLAY_NAMES["high_earning_share"])
ax.set_ylabel(TARGET_LABEL)
ax.set_title("STEM Score vs. estimated payoff time")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "07_stem_vs_payoff.png", dpi=150)
plt.close(fig)
print("Saved 07_stem_vs_payoff.png")

# =====================================================================
# MODEL TRAINING UTILITIES
# =====================================================================

def make_preprocessor(numeric_features):
    numeric_tx = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_tx = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_tx, numeric_features),
        ("cat", categorical_tx, CATEGORICAL_FEATURES),
    ])


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PyTorchRegressor:
    """Sklearn-compatible wrapper around a PyTorch MLP for use in pipelines."""

    def __init__(self, hidden_size=128, lr=1e-3, epochs=200, batch_size=128,
                 patience=25, weight_decay=0, dropout=0.2):
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.model = None
        self.train_losses_ = []
        self.val_losses_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        n_val = max(1, int(0.15 * len(X)))
        idx = np.random.RandomState(42).permutation(len(X))
        X_tr, X_val = X[idx[n_val:]], X[idx[:n_val]]
        y_tr, y_val = y[idx[n_val:]], y[idx[:n_val]]

        self.model = MLP(X.shape[1], self.hidden_size, self.dropout)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.MSELoss()

        X_tr_t = torch.from_numpy(X_tr)
        y_tr_t = torch.from_numpy(y_tr)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)

        best_val, patience_ctr = float("inf"), 0
        best_state = None
        self.train_losses_ = []
        self.val_losses_ = []

        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(X_tr_t))
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, len(X_tr_t), self.batch_size):
                batch_idx = perm[i:i + self.batch_size]
                xb, yb = X_tr_t[batch_idx], y_tr_t[batch_idx]
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            self.model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model(X_val_t), y_val_t).item()
            self.train_losses_.append(epoch_loss / n_batches)
            self.val_losses_.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                patience_ctr = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(X)).numpy()

    def get_params(self, deep=True):
        return {"hidden_size": self.hidden_size, "lr": self.lr,
                "epochs": self.epochs, "batch_size": self.batch_size,
                "patience": self.patience, "weight_decay": self.weight_decay,
                "dropout": self.dropout}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def eval_model(pipe, X_tr, y_tr, X_te, y_te, cv=True):
    """Train, predict, and compute metrics for a sklearn pipeline."""
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_pred_train = pipe.predict(X_tr)
    metrics = {
        "Train_MAE": round(mean_absolute_error(y_tr, y_pred_train), 3),
        "MAE": round(mean_absolute_error(y_te, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_te, y_pred)), 3),
        "R2": round(r2_score(y_te, y_pred), 3),
    }
    if cv:
        cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=CV_FULL,
                                    scoring="neg_mean_absolute_error")
        metrics["CV_MAE"] = round(-cv_scores.mean(), 3)
    return y_pred, metrics


# =====================================================================
# HYPERPARAMETER TUNING (sweep plots)
# =====================================================================

NUMERIC_FEATURES = NUMERIC_WITH_STEM  # use high_earning_share for all final models
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df["estimated_payoff_years"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n── Hyperparameter Tuning ──")

# Ridge / DT are cheap → 5-fold CV, fine grids
# RF / NN are expensive → 3-fold CV, moderate grids
CV_FAST = 3
CV_FULL = 5

# ── Ridge alpha sweep ───────────────────────────────────────────────

alphas = np.logspace(-4, 2, 25)
ridge_cv_maes = []
for a in alphas:
    pipe = Pipeline([("prep", make_preprocessor(NUMERIC_FEATURES)),
                     ("model", Ridge(alpha=a))])
    scores = cross_val_score(pipe, X_train, y_train, cv=CV_FULL,
                             scoring="neg_mean_absolute_error")
    ridge_cv_maes.append(-scores.mean())

best_alpha = alphas[np.argmin(ridge_cv_maes)]
print(f"Ridge best alpha: {best_alpha:.4f} (CV MAE: {min(ridge_cv_maes):.3f})")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(alphas, ridge_cv_maes, "o-")
ax.axvline(best_alpha, color="red", ls="--", label=f"best α = {best_alpha:.2f}")
ax.set_xscale("log")
ax.set_xlabel("Alpha (regularization strength)")
ax.set_ylabel(f"{CV_FULL}-Fold CV MAE")
ax.set_title("Ridge Regression: Alpha Sweep")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "08_ridge_alpha_sweep.png", dpi=150)
plt.close(fig)
print("Saved 08_ridge_alpha_sweep.png")

# ── Decision Tree sweeps ────────────────────────────────────────────

depths = range(2, 30)
dt_depth_maes = []
for d in depths:
    pipe = Pipeline([("prep", make_preprocessor(NUMERIC_FEATURES)),
                     ("model", DecisionTreeRegressor(max_depth=d,
                                                      min_samples_leaf=10,
                                                      random_state=42))])
    scores = cross_val_score(pipe, X_train, y_train, cv=CV_FULL,
                             scoring="neg_mean_absolute_error")
    dt_depth_maes.append(-scores.mean())

best_depth = list(depths)[np.argmin(dt_depth_maes)]

leafs = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
dt_leaf_maes = []
for lf in leafs:
    pipe = Pipeline([("prep", make_preprocessor(NUMERIC_FEATURES)),
                     ("model", DecisionTreeRegressor(max_depth=best_depth,
                                                      min_samples_leaf=lf,
                                                      random_state=42))])
    scores = cross_val_score(pipe, X_train, y_train, cv=CV_FULL,
                             scoring="neg_mean_absolute_error")
    dt_leaf_maes.append(-scores.mean())

best_leaf = leafs[np.argmin(dt_leaf_maes)]
print(f"DT best depth: {best_depth}, best leaf: {best_leaf}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(depths), dt_depth_maes, "o-")
axes[0].axvline(best_depth, color="red", ls="--", label=f"best = {best_depth}")
axes[0].set_xlabel("max_depth")
axes[0].set_ylabel(f"{CV_FULL}-Fold CV MAE")
axes[0].set_title("Decision Tree: max_depth Sweep")
axes[0].legend()

axes[1].plot(leafs, dt_leaf_maes, "o-")
axes[1].axvline(best_leaf, color="red", ls="--", label=f"best = {best_leaf}")
axes[1].set_xlabel("min_samples_leaf")
axes[1].set_ylabel(f"{CV_FULL}-Fold CV MAE")
axes[1].set_title("Decision Tree: min_samples_leaf Sweep")
axes[1].legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "09_dt_sweeps.png", dpi=150)
plt.close(fig)
print("Saved 09_dt_sweeps.png")

# ── Random Forest sweeps ────────────────────────────────────────────

n_est_vals = [50, 75, 100, 150, 200, 300, 500]
rf_nest_maes = []
for n in n_est_vals:
    pipe = Pipeline([("prep", make_preprocessor(NUMERIC_FEATURES)),
                     ("model", RandomForestRegressor(n_estimators=n, max_depth=12,
                                                      random_state=42, n_jobs=-1))])
    scores = cross_val_score(pipe, X_train, y_train, cv=CV_FAST,
                             scoring="neg_mean_absolute_error")
    rf_nest_maes.append(-scores.mean())

best_nest = n_est_vals[np.argmin(rf_nest_maes)]

rf_depths = list(range(4, 22, 2)) + [None]  # None = unlimited depth
rf_depth_maes = []
for d in rf_depths:
    pipe = Pipeline([("prep", make_preprocessor(NUMERIC_FEATURES)),
                     ("model", RandomForestRegressor(n_estimators=best_nest,
                                                      max_depth=d, random_state=42,
                                                      n_jobs=-1))])
    scores = cross_val_score(pipe, X_train, y_train, cv=CV_FAST,
                             scoring="neg_mean_absolute_error")
    rf_depth_maes.append(-scores.mean())

best_rf_depth = rf_depths[np.argmin(rf_depth_maes)]

# Phase 3: min_samples_leaf sweep (with best n_est + depth)
rf_leafs = [1, 2, 3, 5, 8, 10, 15, 20]
rf_leaf_maes = []
for lf in rf_leafs:
    pipe = Pipeline([("prep", make_preprocessor(NUMERIC_FEATURES)),
                     ("model", RandomForestRegressor(n_estimators=best_nest,
                                                      max_depth=best_rf_depth,
                                                      min_samples_leaf=lf,
                                                      random_state=42, n_jobs=-1))])
    scores = cross_val_score(pipe, X_train, y_train, cv=CV_FAST,
                             scoring="neg_mean_absolute_error")
    rf_leaf_maes.append(-scores.mean())

best_rf_leaf = rf_leafs[np.argmin(rf_leaf_maes)]
print(f"RF best n_estimators: {best_nest}, best depth: {best_rf_depth}, "
      f"best leaf: {best_rf_leaf}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(n_est_vals, rf_nest_maes, "o-")
axes[0].axvline(best_nest, color="red", ls="--", label=f"best = {best_nest}")
axes[0].set_xlabel("n_estimators")
axes[0].set_ylabel(f"{CV_FAST}-Fold CV MAE")
axes[0].set_title("Random Forest: n_estimators Sweep")
axes[0].legend()

rf_depth_labels = [str(d) if d is not None else "None" for d in rf_depths]
axes[1].plot(rf_depth_labels, rf_depth_maes, "o-")
best_label = str(best_rf_depth) if best_rf_depth is not None else "None"
axes[1].axvline(rf_depth_labels.index(best_label), color="red", ls="--",
                label=f"best = {best_label}")
axes[1].set_xlabel("max_depth")
axes[1].set_ylabel(f"{CV_FAST}-Fold CV MAE")
axes[1].set_title("Random Forest: max_depth Sweep")
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)

axes[2].plot(rf_leafs, rf_leaf_maes, "o-")
axes[2].axvline(best_rf_leaf, color="red", ls="--", label=f"best = {best_rf_leaf}")
axes[2].set_xlabel("min_samples_leaf")
axes[2].set_ylabel(f"{CV_FAST}-Fold CV MAE")
axes[2].set_title("Random Forest: min_samples_leaf Sweep")
axes[2].legend()

fig.tight_layout()
fig.savefig(FIG_DIR / "10_rf_sweeps.png", dpi=150)
plt.close(fig)
print("Saved 10_rf_sweeps.png")

# ── Neural Network sweeps ───────────────────────────────────────────

preprocessor_nn = make_preprocessor(NUMERIC_FEATURES)
X_train_nn = preprocessor_nn.fit_transform(X_train).astype(np.float32)
X_test_nn = preprocessor_nn.transform(X_test).astype(np.float32)
y_train_np = y_train.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)

# Hold out 20% of training data for NN hyperparameter selection
# (never touch X_test during sweeps to avoid data leakage)
nn_val_n = int(0.2 * len(X_train_nn))
nn_rng = np.random.RandomState(42)
nn_shuffle = nn_rng.permutation(len(X_train_nn))
X_sweep_val = X_train_nn[nn_shuffle[:nn_val_n]]
y_sweep_val = y_train_np[nn_shuffle[:nn_val_n]]
X_sweep_train = X_train_nn[nn_shuffle[nn_val_n:]]
y_sweep_train = y_train_np[nn_shuffle[nn_val_n:]]

NN_SWEEP_EPOCHS = 150
NN_FINAL_EPOCHS = 300

# Phase 1: hidden size sweep (at sensible defaults)
hidden_sizes = [32, 64, 96, 128, 192, 256, 384]
nn_hidden_maes = []
for hs in hidden_sizes:
    model = PyTorchRegressor(hidden_size=hs, lr=1e-3, epochs=NN_SWEEP_EPOCHS,
                             weight_decay=1e-4)
    model.fit(X_sweep_train, y_sweep_train)
    preds = model.predict(X_sweep_val)
    nn_hidden_maes.append(mean_absolute_error(y_sweep_val, preds))

best_hidden = hidden_sizes[np.argmin(nn_hidden_maes)]

# Phase 2: learning rate sweep (with best hidden size)
lrs = [1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3]
nn_lr_maes = []
for lr_val in lrs:
    model = PyTorchRegressor(hidden_size=best_hidden, lr=lr_val,
                             epochs=NN_SWEEP_EPOCHS, weight_decay=1e-4)
    model.fit(X_sweep_train, y_sweep_train)
    preds = model.predict(X_sweep_val)
    nn_lr_maes.append(mean_absolute_error(y_sweep_val, preds))

best_lr = lrs[np.argmin(nn_lr_maes)]

# Phase 3: weight decay sweep (with best hidden + lr)
wds = [0, 1e-5, 1e-4, 5e-4, 1e-3, 3e-3, 1e-2]
nn_wd_maes = []
for wd in wds:
    model = PyTorchRegressor(hidden_size=best_hidden, lr=best_lr,
                             epochs=NN_SWEEP_EPOCHS, weight_decay=wd)
    model.fit(X_sweep_train, y_sweep_train)
    preds = model.predict(X_sweep_val)
    nn_wd_maes.append(mean_absolute_error(y_sweep_val, preds))

best_wd = wds[np.argmin(nn_wd_maes)]

# Phase 4: dropout sweep (with best hidden + lr + wd)
dropouts = [0.0, 0.1, 0.2, 0.3, 0.4]
nn_do_maes = []
for do_val in dropouts:
    model = PyTorchRegressor(hidden_size=best_hidden, lr=best_lr,
                             epochs=NN_SWEEP_EPOCHS, weight_decay=best_wd,
                             dropout=do_val)
    model.fit(X_sweep_train, y_sweep_train)
    preds = model.predict(X_sweep_val)
    nn_do_maes.append(mean_absolute_error(y_sweep_val, preds))

best_do = dropouts[np.argmin(nn_do_maes)]
print(f"NN best hidden: {best_hidden}, lr: {best_lr}, wd: {best_wd}, dropout: {best_do}")

# Train final NN to get loss curves
final_nn = PyTorchRegressor(hidden_size=best_hidden, lr=best_lr,
                            epochs=NN_FINAL_EPOCHS, weight_decay=best_wd,
                            dropout=best_do)
final_nn.fit(X_train_nn, y_train_np)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].plot(hidden_sizes, nn_hidden_maes, "o-")
axes[0, 0].axvline(best_hidden, color="red", ls="--", label=f"best = {best_hidden}")
axes[0, 0].set_xlabel("Hidden Layer Size")
axes[0, 0].set_ylabel("Validation MAE")
axes[0, 0].set_title("Hidden Size Sweep")
axes[0, 0].legend()

axes[0, 1].plot(lrs, nn_lr_maes, "o-")
axes[0, 1].axvline(best_lr, color="red", ls="--", label=f"best = {best_lr:.0e}")
axes[0, 1].set_xscale("log")
axes[0, 1].set_xlabel("Learning Rate")
axes[0, 1].set_ylabel("Validation MAE")
axes[0, 1].set_title("Learning Rate Sweep")
axes[0, 1].legend()

axes[0, 2].plot(wds, nn_wd_maes, "o-")
axes[0, 2].axvline(best_wd, color="red", ls="--",
                    label=f"best = {best_wd:.0e}" if best_wd > 0 else "best = 0")
axes[0, 2].set_xscale("symlog", linthresh=1e-6)
axes[0, 2].set_xlabel("Weight Decay")
axes[0, 2].set_ylabel("Validation MAE")
axes[0, 2].set_title("Weight Decay (L2) Sweep")
axes[0, 2].legend()

axes[1, 0].plot(dropouts, nn_do_maes, "o-")
axes[1, 0].axvline(best_do, color="red", ls="--", label=f"best = {best_do}")
axes[1, 0].set_xlabel("Dropout Rate")
axes[1, 0].set_ylabel("Validation MAE")
axes[1, 0].set_title("Dropout Sweep")
axes[1, 0].legend()

axes[1, 1].plot(final_nn.train_losses_, label="Train Loss", alpha=0.8)
axes[1, 1].plot(final_nn.val_losses_, label="Val Loss", alpha=0.8)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("MSE Loss")
axes[1, 1].set_title("Training Curves (final model)")
axes[1, 1].legend()

axes[1, 2].axis("off")
summary_text = (
    f"Best hyperparameters:\n"
    f"  Hidden size: {best_hidden}\n"
    f"  Learning rate: {best_lr}\n"
    f"  Weight decay: {best_wd}\n"
    f"  Dropout: {best_do}\n"
    f"  Epochs: {NN_FINAL_EPOCHS}"
)
axes[1, 2].text(0.1, 0.5, summary_text, fontsize=13, family="monospace",
                verticalalignment="center", transform=axes[1, 2].transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

fig.suptitle("Neural Network Hyperparameter Tuning", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "11_nn_sweeps.png", dpi=150)
plt.close(fig)
print("Saved 11_nn_sweeps.png")

# =====================================================================
# TRAIN FINAL MODELS (with tuned hyperparameters)
# =====================================================================

print("\n── Training Final Models ──")

results = {}
preds_dict = {}

# Ridge
ridge_pipe = Pipeline([("prep", make_preprocessor(NUMERIC_FEATURES)),
                        ("model", Ridge(alpha=best_alpha))])
y_pred_ridge, results["Ridge"] = eval_model(
    ridge_pipe, X_train, y_train, X_test, y_test
)
preds_dict["Ridge"] = y_pred_ridge

# Decision Tree
dt_pipe = Pipeline([("prep", make_preprocessor(NUMERIC_FEATURES)),
                     ("model", DecisionTreeRegressor(max_depth=best_depth,
                                                      min_samples_leaf=best_leaf,
                                                      random_state=42))])
y_pred_dt, results["Decision Tree"] = eval_model(
    dt_pipe, X_train, y_train, X_test, y_test
)
preds_dict["Decision Tree"] = y_pred_dt

# Random Forest
rf_pipe = Pipeline([("prep", make_preprocessor(NUMERIC_FEATURES)),
                     ("model", RandomForestRegressor(n_estimators=best_nest,
                                                      max_depth=best_rf_depth,
                                                      min_samples_leaf=best_rf_leaf,
                                                      random_state=42, n_jobs=-1))])
y_pred_rf, results["Random Forest"] = eval_model(
    rf_pipe, X_train, y_train, X_test, y_test
)
preds_dict["Random Forest"] = y_pred_rf

# Neural Network (already trained above, just evaluate)
y_pred_nn = final_nn.predict(X_test_nn)
y_pred_nn_train = final_nn.predict(X_train_nn)
nn_cv_scores = []
for fold in range(CV_FAST):
    rng = np.random.RandomState(fold)
    idx = rng.permutation(len(X_train_nn))
    n_val = len(X_train_nn) // CV_FAST
    xv, yv = X_train_nn[idx[:n_val]], y_train_np[idx[:n_val]]
    xt, yt = X_train_nn[idx[n_val:]], y_train_np[idx[n_val:]]
    fold_model = PyTorchRegressor(hidden_size=best_hidden, lr=best_lr,
                                  epochs=NN_SWEEP_EPOCHS, weight_decay=best_wd,
                                  dropout=best_do)
    fold_model.fit(xt, yt)
    nn_cv_scores.append(mean_absolute_error(yv, fold_model.predict(xv)))

results["Neural Network"] = {
    "Train_MAE": round(mean_absolute_error(y_train_np, y_pred_nn_train), 3),
    "MAE": round(mean_absolute_error(y_test_np, y_pred_nn), 3),
    "RMSE": round(np.sqrt(mean_squared_error(y_test_np, y_pred_nn)), 3),
    "R2": round(r2_score(y_test_np, y_pred_nn), 3),
    "CV_MAE": round(np.mean(nn_cv_scores), 3),
}
preds_dict["Neural Network"] = y_pred_nn

with open(SCRIPTS_DIR / "model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nModel results:\n{json.dumps(results, indent=2)}")

# =====================================================================
# CAPPED vs NON-CAPPED DIAGNOSTIC
# =====================================================================

print("\n── Capped vs Non-Capped Diagnostic ──")

capped_mask = y_test >= MAX_PAYOFF - 0.01
noncapped_mask = ~capped_mask
n_capped_test = capped_mask.sum()
n_noncapped_test = noncapped_mask.sum()
print(f"Test set: {n_capped_test} capped ({n_capped_test/len(y_test)*100:.1f}%), "
      f"{n_noncapped_test} non-capped ({n_noncapped_test/len(y_test)*100:.1f}%)")

split_results = {}
for name, yp in preds_dict.items():
    yp_arr = np.asarray(yp)
    split_results[name] = {
        "capped_MAE": round(mean_absolute_error(
            y_test[capped_mask], yp_arr[capped_mask.values]), 3),
        "noncapped_MAE": round(mean_absolute_error(
            y_test[noncapped_mask], yp_arr[noncapped_mask.values]), 3),
        "noncapped_R2": round(r2_score(
            y_test[noncapped_mask], yp_arr[noncapped_mask.values]), 3),
        "pct_predicted_capped": round(
            (yp_arr > MAX_PAYOFF - 1).sum() / len(yp_arr) * 100, 1),
    }

print(f"\n{'Model':<20s} {'Capped MAE':>12s} {'NonCap MAE':>12s} "
      f"{'NonCap R²':>10s} {'% Pred≥24':>10s}")
print("-" * 66)
for name, s in split_results.items():
    print(f"{name:<20s} {s['capped_MAE']:12.3f} {s['noncapped_MAE']:12.3f} "
          f"{s['noncapped_R2']:10.3f} {s['pct_predicted_capped']:9.1f}%")

with open(SCRIPTS_DIR / "capped_diagnostic.json", "w") as f:
    json.dump(split_results, f, indent=2)
print("\nSaved capped_diagnostic.json")

# =====================================================================
# MODEL COMPARISON FIGURES
# =====================================================================

# ── Figure 12: Model comparison bar chart ────────────────────────────

metrics_list = ["MAE", "RMSE", "R2", "CV_MAE"]
model_names = list(results.keys())
x = np.arange(len(metrics_list))
width = 0.18

fig, ax = plt.subplots(figsize=(10, 5))
for i, name in enumerate(model_names):
    vals = [results[name][m] for m in metrics_list]
    ax.bar(x + i * width, vals, width, label=name)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics_list)
ax.set_ylabel("Score")
ax.set_title("Model Comparison (Test Set)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "12_model_comparison.png", dpi=150)
plt.close(fig)
print("Saved 12_model_comparison.png")

# ── Figure 13: Predicted vs actual (all 4 models) ───────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
for ax, (name, yp) in zip(axes.flat, preds_dict.items()):
    ax.scatter(y_test, yp, alpha=0.3, s=10)
    ax.plot([0, MAX_PAYOFF], [0, MAX_PAYOFF], "r--", lw=1)
    ax.set_xlabel("Actual payoff (years)")
    ax.set_ylabel("Predicted payoff (years)")
    ax.set_title(name)
fig.suptitle("Predicted vs. actual", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "13_predicted_vs_actual.png", dpi=150)
plt.close(fig)
print("Saved 13_predicted_vs_actual.png")

# =====================================================================
# ENGINEERED FEATURE ON/OFF COMPARISON
# =====================================================================

print("\n── Feature Engineering Comparison ──")

comparison = {"With STEM Score": {}, "Without STEM Score": {}}

for name_tag, num_feats in [("With STEM Score", NUMERIC_WITH_STEM),
                             ("Without STEM Score", NUMERIC_WITHOUT_STEM)]:
    X_comp = df[num_feats + CATEGORICAL_FEATURES]
    Xtr, Xte, ytr, yte = train_test_split(X_comp, y, test_size=0.2, random_state=42)

    for model_name, model_obj in [
        ("Ridge", Ridge(alpha=best_alpha)),
        ("Decision Tree", DecisionTreeRegressor(max_depth=best_depth,
                                                 min_samples_leaf=best_leaf,
                                                 random_state=42)),
        ("Random Forest", RandomForestRegressor(n_estimators=best_nest,
                                                 max_depth=best_rf_depth,
                                                 min_samples_leaf=best_rf_leaf,
                                                 random_state=42, n_jobs=-1)),
    ]:
        pipe = Pipeline([("prep", make_preprocessor(num_feats)),
                          ("model", model_obj)])
        _, metrics = eval_model(pipe, Xtr, ytr, Xte, yte, cv=False)
        comparison[name_tag][model_name] = metrics

    # NN comparison
    prep_comp = make_preprocessor(num_feats)
    Xtr_nn = prep_comp.fit_transform(Xtr).astype(np.float32)
    Xte_nn = prep_comp.transform(Xte).astype(np.float32)
    nn_comp = PyTorchRegressor(hidden_size=best_hidden, lr=best_lr,
                               epochs=NN_SWEEP_EPOCHS, weight_decay=best_wd,
                               dropout=best_do)
    nn_comp.fit(Xtr_nn, ytr.values.astype(np.float32))
    p = nn_comp.predict(Xte_nn)
    comparison[name_tag]["Neural Network"] = {
        "MAE": round(mean_absolute_error(yte, p), 3),
        "RMSE": round(np.sqrt(mean_squared_error(yte, p)), 3),
        "R2": round(r2_score(yte, p), 3),
    }

# ── Figure 14: Feature on/off comparison ─────────────────────────────

comp_models = ["Ridge", "Decision Tree", "Random Forest", "Neural Network"]
x = np.arange(len(comp_models))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric in zip(axes, ["MAE", "R2"]):
    with_vals = [comparison["With STEM Score"][m][metric] for m in comp_models]
    without_vals = [comparison["Without STEM Score"][m][metric] for m in comp_models]
    ax.bar(x - width / 2, with_vals, width,
           label=f"With {DISPLAY_NAMES['high_earning_share']}")
    ax.bar(x + width / 2, without_vals, width,
           label=f"Without {DISPLAY_NAMES['high_earning_share']}")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_models, rotation=15, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}: with vs. without engineered STEM share")
    ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "14_feature_onoff.png", dpi=150)
plt.close(fig)
print("Saved 14_feature_onoff.png")

with open(SCRIPTS_DIR / "feature_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)

# =====================================================================
# ROC CURVES + CONFUSION MATRICES
# =====================================================================

print("\n── Classification Metrics (threshold-based) ──")

BURDEN_YEARS = 20  # federal extended repayment max; policy-meaningful cutoff
threshold = BURDEN_YEARS
y_test_binary = (y_test > threshold).astype(int)

# ── Figure 15: ROC curves ───────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))
auc_scores = {}
for name, yp in preds_dict.items():
    scores = np.asarray(yp, dtype=float)
    fpr, tpr, _ = roc_curve(y_test_binary, scores)
    auc_val = roc_auc_score(y_test_binary, scores)
    auc_scores[name] = round(auc_val, 3)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curves (threshold = {threshold:.1f} years)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "15_roc_curves.png", dpi=150)
plt.close(fig)
print("Saved 15_roc_curves.png")

# ── Figure 16: Confusion matrices ───────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, yp) in zip(axes.flat, preds_dict.items()):
    yp_binary = (np.asarray(yp) > threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_test_binary, yp_binary,
        labels=[0, 1],
        display_labels=["Low Burden", "High Burden"],
        cmap="Blues", ax=ax
    )
    ax.set_title(name)
fig.suptitle(f"Confusion Matrices (threshold = {threshold} years)", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "16_confusion_matrices.png", dpi=150)
plt.close(fig)
print("Saved 16_confusion_matrices.png")

# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================

print("\n── Feature Importance ──")

# Get feature names from the preprocessor
prep_fitted = ridge_pipe.named_steps["prep"]
num_names = NUMERIC_FEATURES
cat_names = list(prep_fitted.named_transformers_["cat"]
                 .named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES))
all_feat_names = num_names + cat_names

# ── Figure 17: Ridge coefficients ───────────────────────────────────

ridge_coefs = ridge_pipe.named_steps["model"].coef_
coef_series = pd.Series(ridge_coefs, index=all_feat_names)
top_coefs = coef_series.abs().nlargest(15)
top_coefs_signed = coef_series[top_coefs.index].sort_values()
top_coefs_signed.index = [display_feature_name(i) for i in top_coefs_signed.index]

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#d62728" if v > 0 else "#1f77b4" for v in top_coefs_signed]
top_coefs_signed.plot.barh(ax=ax, color=colors)
ax.set_xlabel("Standardized coefficient")
ax.set_title("Ridge regression: top 15 feature coefficients")
fig.tight_layout()
fig.savefig(FIG_DIR / "17_ridge_importance.png", dpi=150)
plt.close(fig)
print("Saved 17_ridge_importance.png")

# ── Figure 18: DT + RF feature importances ──────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, (name, pipe) in zip(axes, [("Decision Tree", dt_pipe),
                                     ("Random Forest", rf_pipe)]):
    importances = pipe.named_steps["model"].feature_importances_
    imp_series = pd.Series(importances, index=all_feat_names).nlargest(15).sort_values()
    imp_series.index = [display_feature_name(i) for i in imp_series.index]
    imp_series.plot.barh(ax=ax)
    ax.set_xlabel("Feature importance")
    ax.set_title(f"{name}: top 15 features")
fig.tight_layout()
fig.savefig(FIG_DIR / "18_tree_importance.png", dpi=150)
plt.close(fig)
print("Saved 18_tree_importance.png")

# ── Figure 19: NN permutation importance ─────────────────────────────

class NNWrapper:
    """Minimal wrapper so sklearn's permutation_importance can call .predict()."""
    def __init__(self, nn_model):
        self._nn = nn_model
    def fit(self, X, y):
        return self
    def predict(self, X):
        return self._nn.predict(X)

perm_result = permutation_importance(
    NNWrapper(final_nn), X_test_nn, y_test_np,
    n_repeats=5, random_state=42, scoring="neg_mean_absolute_error"
)
perm_series = pd.Series(perm_result.importances_mean, index=all_feat_names)
top_perm = perm_series.nlargest(15).sort_values()
top_perm.index = [display_feature_name(i) for i in top_perm.index]

fig, ax = plt.subplots(figsize=(8, 6))
top_perm.plot.barh(ax=ax)
ax.set_xlabel("Mean MAE increase (permutation)")
ax.set_title("Neural network: top 15 features (permutation importance)")
fig.tight_layout()
fig.savefig(FIG_DIR / "19_nn_importance.png", dpi=150)
plt.close(fig)
print("Saved 19_nn_importance.png")

# =====================================================================
# SUBGROUP ERROR ANALYSIS
# =====================================================================

print("\n── Subgroup Error Analysis ──")

# ── Figure 20: MAE by institution type ──────────────────────────────

test_control = df.loc[y_test.index, "CONTROL_label"]
subgroup_results = {}

for name, yp in preds_dict.items():
    yp_arr = np.asarray(yp)
    for label in existing:
        mask = test_control == label
        if mask.sum() == 0:
            continue
        mae = mean_absolute_error(y_test[mask], yp_arr[mask])
        subgroup_results.setdefault(name, {})[label] = round(mae, 3)

sub_df = pd.DataFrame(subgroup_results).T
fig, ax = plt.subplots(figsize=(10, 5))
sub_df.plot.bar(ax=ax)
ax.set_ylabel("MAE")
ax.set_title("MAE by institution type")
ax.legend(title=DISPLAY_NAMES["CONTROL"])
plt.xticks(rotation=15, ha="right")
fig.tight_layout()
fig.savefig(FIG_DIR / "20_subgroup_mae.png", dpi=150)
plt.close(fig)
print("Saved 20_subgroup_mae.png")

# ── Figure 21: Residual distributions by institution type ───────────

best_model_name = min(results, key=lambda k: results[k]["MAE"])
best_preds = preds_dict[best_model_name]
residuals = y_test.values - np.asarray(best_preds)

fig, ax = plt.subplots(figsize=(8, 5))
for label in existing:
    mask = test_control == label
    if mask.sum() == 0:
        continue
    ax.hist(residuals[mask.values], bins=30, alpha=0.5, label=label, edgecolor="white")
ax.axvline(0, color="black", ls="--", lw=1)
ax.set_xlabel("Residual (actual − predicted payoff, years)")
ax.set_ylabel("Count")
ax.set_title(f"Residual distribution by institution type ({best_model_name})")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "21_residuals_by_type.png", dpi=150)
plt.close(fig)
print("Saved 21_residuals_by_type.png")

# =====================================================================
# DONE
# =====================================================================

print(f"\nDone. All figures saved to {FIG_DIR}/")
print(f"Total figures: {len(list(FIG_DIR.glob('*.png')))}")
