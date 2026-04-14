# Pipeline Decisions

Rationale for the feature pipeline. **Implementation:** `scripts/generate_figures.py` (and `scripts/export_model.py` for the demo export).

---

## D1: Drop rows with missing target components BEFORE engineering

**Data**: 1,822 / 6,322 rows (28.8%) had NaN for GRAD_DEBT_MDN or MD_EARN_WNE_P10.
`amortize_years` silently mapped NaN inputs to the 25-year cap, inflating capped %
from the true 42.0% to a false 58.7% and corrupting every downstream model.

**Decision**: Filter on `dropna(subset=TARGET_COMPONENTS)` before any target math.
**Status**: IMPLEMENTED.

---

## D2: Merge NPT4_PUB and NPT4_PRIV into a single NPT4 feature

**Data**: These are mutually exclusive (overlap: 0 rows).
- NPT4_PUB: 68.2% missing (only public schools report it)
- NPT4_PRIV: 45.2% missing (only private/FP schools report it)
- Neither column: 600 rows (13.3%)

Median-imputing a 68% missing column is injecting noise. A merged
`NPT4 = NPT4_PUB.fillna(NPT4_PRIV)` has only 13.3% missing.

**Decision**: Create NPT4, drop NPT4_PUB and NPT4_PRIV from feature list.
**Status**: IMPLEMENTED.

---

## D3: Drop TUITIONFEE_IN (keep TUITIONFEE_OUT only)

**Data**: TUITIONFEE_IN ↔ TUITIONFEE_OUT correlation r=0.933.
TUITIONFEE_IN ↔ NPT4_PUB r=0.701 (if we keep the merged NPT4).
Target correlations: TUITIONFEE_OUT r=-0.409 vs TUITIONFEE_IN r=-0.302.

Near-duplicate features waste capacity in Ridge (unstable coefficients)
and split importance in trees. TUITIONFEE_OUT is both more predictive
and less collinear with NPT4.

**Decision**: Drop TUITIONFEE_IN, keep TUITIONFEE_OUT.
**Status**: IMPLEMENTED.

---

## D4: Log-transform UGDS (enrollment)

**Data**: Skewness = 7.70. Range: 0 to 163,164. Mean=3,320 vs Median=956.
The p95 is 14,884 but the max is 163,164 (10x the p95). StandardScaler
after log will produce a much more normal distribution.

Note: UGDS has 402 zeros (8.9%). Use log1p to handle them.

**Decision**: Replace UGDS with log1p(UGDS) before scaling.
**Status**: IMPLEMENTED.

---

## D5: Handle ADM_RATE (62.8% missing) with missingness indicator

**Data**: 2,828 / 4,500 values are missing (62.8%). Target correlation r=0.200.
Only 1,672 schools report admission rate — mostly 4-year selective institutions.
Open-admission and for-profit schools typically don't report.

Median-imputing 63% of a feature is mostly noise. But the *absence* of a
reported admission rate is itself a strong signal (for-profit and open-admission
schools don't report, and those schools have longer payoff).

**Decision**: Keep ADM_RATE (median-imputed) AND add ADM_RATE_missing binary indicator.
**Status**: IMPLEMENTED.

---

## D6: Add PCIP_missing indicator for High-Earning Score (high_earning_share)

**Note**: "High-Earning Score" = sum of degree shares across CS, engineering, biology,
math, physical sciences, and business (PCIP11/14/26/27/40/52). Includes business
because it is consistently high-earning in the Scorecard data.

**Data**: 41.7% of institutions have high_earning_share=0. By type:
- Public: 14.3% zero
- Private Nonprofit: 23.5% zero
- For-Profit: 84.4% zero

Most for-profit zeros are real (vocational/trade schools). But some
zeros are from NaN PCIP fields that we filled with 0. The model can't
distinguish "no high-earning programs" from "data not reported."

**Decision**: Add PCIP_reported binary feature (1 if any PCIP column was non-NaN
in the raw data, 0 if all were NaN before our fillna).
**Status**: IMPLEMENTED.

---

## D7: Consider dropping UGDS_HISP

**Data**: Target correlation r=0.006 — essentially zero signal.
Keeping it adds a dimension with no predictive value.

**Decision**: KEEP. r=0.006 linearly doesn't mean no signal in trees/NN.
It's a proportion bounded [0,1] so it's not harmful. Low priority.
**Status**: NO ACTION.

---

## D8: Consider dropping PCTFLOAN

**Data**: Target correlation r=0.191. Not the weakest, but this feature
has conceptual overlap with PCTPELL (r between them should be checked).
If they're collinear, one may be redundant.

Profile shows PCTPELL ↔ PCTFLOAN was NOT flagged at r>0.70 threshold,
so they carry distinct information.

**Decision**: KEEP both.
**Status**: NO ACTION.

---

## D9: Target distribution — 42% capped at 25 years

**Data**: 1,890 / 4,500 rows have target = 25.0 (the cap).
This is a censored distribution. Standard regression treats 25.0
as a known value, but it really means "25 or more."

Options considered:
- (a) Treat as-is: simple, defensible for a class project.
- (b) Log-transform target: would compress the cap region.
- (c) Two-stage model (classify capped vs not, then regress on non-capped).
- (d) Tobit regression (censored regression model).

**Decision**: (a) for the project, note (c) and (d) as future work.
The 42% cap rate is a real reflection of the data (for-profit schools
genuinely have extremely long payoff times). Acknowledging it in the
presentation is more valuable than adding model complexity.
**Status**: NO ACTION (acknowledged in presentation).

---

## Summary

| # | Change | Status |
|---|--------|--------|
| D1 | Drop NaN target rows before engineering | IMPLEMENTED |
| D2 | Merge NPT4 | IMPLEMENTED |
| D3 | Drop TUITIONFEE_IN | IMPLEMENTED |
| D4 | Log-transform UGDS | IMPLEMENTED |
| D5 | Add ADM_RATE_missing indicator | IMPLEMENTED |
| D6 | Add PCIP_reported indicator | IMPLEMENTED |
| D7 | Keep UGDS_HISP | NO ACTION |
| D8 | Keep PCTFLOAN | NO ACTION |
| D9 | Acknowledge capped target | NO ACTION (presentation) |
