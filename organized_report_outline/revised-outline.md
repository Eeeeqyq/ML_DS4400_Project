# Final Report: Revised Outline

Headers match the professor's required sections exactly. Subheaders within each are ours.

---

## Problem Description

Open with the real-world framing: a high school senior choosing between schools has no simple way to compare long-term financial outcomes. Median debt and median earnings exist publicly, but nobody combines them into a single "how long until I'm debt-free?" number.

State the prediction task: given pre-enrollment institutional characteristics, predict estimated years to pay off student debt.

Note that after our proposal phase, the professor asked us to incorporate an engineered variable. We chose to engineer the target itself, since no payoff-time column exists in the data. This turned out to be the defining challenge of the project.

Briefly mention the interactive demo we built so a student can actually use this (link it here, revisit in Conclusion).

---

## Dataset and Exploratory Data Analysis

### The College Scorecard
Source, size (6,400+ institutions, 3,300+ columns), what it contains, the October 2025 release. After filtering to institutions with complete debt and earnings data: 4,500 rows, 13 numeric + 5 categorical features, 80/20 train-test split.

### Features and Selection
Walk through categorical and numeric features. Emphasize the choices that required judgment:
- Merging mutually exclusive net price columns (68% missing each, 13% after merge)
- Dropping collinear in-state tuition (r = 0.93 with out-of-state)
- Log-transforming enrollment (skewness 7.7)
- Missingness indicators for admission rate and degree-field data (informative absence)
- Dropping SAT (only 23% reporting)

### High-Earning Score (Engineered Feature)
Sum of degree shares across six high-earning fields: CS, engineering, biology, math, physical sciences, and business. Why these six. Tested with and without across all four models: small but consistent improvement everywhere.

### Exploratory Findings
42% of institutions hit the 25-year cap. For-profits: 67% capped, median at 25 years. Public/nonprofit: ~28% capped, medians around 11-12 years. Pell Grant share, tuition, and institution type are the strongest correlates. Tuition alone is misleading (for-profits cluster at moderate tuition but long payoff). Reference figures.

### Engineering the Target Variable
Why it was necessary (no payoff column exists; professor asked us to engineer a variable). The formula, step by step, each grounded in a specific data source:

1. After-tax income. Progressive marginal brackets approximating combined federal + state + FICA. Cite IRS 2024 brackets and SSA FICA rates.
2. Living costs. Floor of $24,200 scaled with a log curve. Both parameters fitted to BLS Consumer Expenditure Survey 2022 per-capita expenditure data across income quintiles. Cite specific BLS table. Matteo (our TA) pushed us to calibrate against real government data here rather than use a flat assumption.
3. Repayment. 50% of remaining disposable income, $500/year minimum (prevents infinite amortization for near-zero surplus schools).
4. Amortization. Standard compound interest at 5.50%, the 2023-24 federal direct loan rate. Cite studentaid.gov.
5. Cap at 25 years. Federal income-driven repayment forgiveness policy. Cite 34 CFR 685.209.

### Iterating on the Target
The milestone version used a flat 22% tax, flat $20K living costs, 20% repayment, 40-year cap. Less than 1% of institutions hit that cap. After calibrating to real data, 42% hit the 25-year cap. This completely changed the modeling problem. Professor Amir's observation: engineering a target is inherently risky because small parameter tweaks have outsized downstream effects. We agree. The target is the project.

Also: caught a NaN bug where the amortization function silently capped rows with missing inputs, contaminating 1,822 training rows. Data profiling caught it.

---

## Approach and Methodology

### Preprocessing
Scikit-learn Pipeline. Median imputation + z-score normalization for numerics. Constant imputation + one-hot encoding for categoricals. All inside the pipeline to prevent leakage.

### Models
Four models, four different families:

*Ridge Regression*: L2-regularized linear model. Interpretable baseline. Tuned regularization strength.

*Decision Tree*: Single CART regressor. Captures nonlinear splits and interactions. Tuned depth and leaf size.

*Random Forest*: Ensemble of decision trees via bagging. Reduces variance relative to a single tree. Tuned tree count, depth, and leaf size.

*Neural Network*: Two-layer MLP in PyTorch with BatchNorm, dropout, and cosine annealing. Tuned architecture, learning rate, weight decay, and dropout. Validation split carved from training data to avoid leakage during tuning.

All four tuned via systematic hyperparameter search with 5-fold cross-validation. Expanded ranges when optima hit edges, narrowed when they converged.

### Evaluation Strategy
MAE, RMSE, R squared on a held-out 20% test set. 5-fold CV MAE and training MAE for overfitting diagnostics. Capped vs. non-capped split evaluation to diagnose the censored target. Binary classification framing (payoff above/below 20 years) with ROC curves and confusion matrices.

---

## Discussion and Result Interpretation

### Model Comparison
Results table (Train MAE, Test MAE, RMSE, R squared, CV MAE for all four). Random Forest wins: 3.95 MAE, 0.63 R squared. Neural Network second. Ridge weakest. Consistent with literature on tree ensembles vs. deep learning for tabular data (cite Grinsztajn et al. 2022).

### Feature Importance
Ridge coefficients highlight credential type and region. Tree importance highlights tuition, Pell share, enrollment. Consensus across families: Pell Grant share, tuition, and credential type are the real drivers.

### The Censored Target Problem
Split evaluation table. Ridge and Decision Tree have negative R squared on non-capped schools (worse than the mean). They learned a binary capped-or-not signal and stopped. Random Forest and Neural Network show R squared ~0.28 on non-capped data: genuine but modest regression ability. The best model is part classifier, part regressor. Tobit regression or a two-stage approach would be more principled (future work).

### Why Models Make Errors
Three sources: (1) the censored target inflates headline metrics, (2) institutional features are aggregate proxies for individual outcomes, (3) the engineered target's assumptions are themselves imperfect.

### Classification Results
Binary classification (payoff >= 20 years). All four models perform well. Random Forest highest AUC. Even where regression is noisy, models reliably separate high-burden from low-burden schools.

---

## Conclusion

Random Forest is the best model (3.95 MAE, 0.63 R squared). Pre-enrollment features carry real signal about debt burden.

The 42% cap means all models partially act as classifiers. Honest split evaluation was essential.

The biggest lesson: target engineering is the project. We started with a rough formula, calibrated it to government data, and watched each refinement reshape the entire modeling landscape. Getting the target right (or being honest about where it falls short) matters more than model selection.

We also built an interactive demo deployed on Vercel where a student can plug in school characteristics and get an estimated payoff time. It runs the full Random Forest in the browser. Link it. This is the product: making a machine learning model useful to a 17-year-old.

Future work: location-adjusted living costs, earnings growth trajectories, Tobit/two-stage censored regression, and loan-level NSLDS data for ground-truth outcomes.

---

## References
Expanded list with precise citations for every calibration parameter (BLS table, IRS rev. proc., SSA FICA, studentaid.gov rate, 34 CFR 685.209, Grinsztajn et al., scikit-learn, PyTorch, Stanford CS229 project, Dean GBM analysis).

---

## Team Member Contributions
Same as before.
