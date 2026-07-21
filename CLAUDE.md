# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This repository generates multiple coherent, synthetic datasets for educational purposes — each one built for teaching **Exploratory Data Analysis (EDA)** and then applied **Machine Learning (ML)** (scikit-learn) in an ML unit.

Each dataset lives in its own top-level folder (e.g. [online-gaming/](online-gaming/), [coffee-health/](coffee-health/)). See [README.md](README.md) for the full repository structure and the step-by-step process used to build a new dataset — read it before starting work on a new or existing dataset folder.

**Why synthetic datasets**: real Kaggle-style datasets used as a starting point are often generated with no logical correlations or patterns between features (fine for applied ML, unsuitable for teaching EDA). The fix is a synthetic dataset with similar features but realistic, logical relationships and patterns students can discover through analysis — while remaining suitable for applied ML.

## Established Process (per dataset)

Follow this sequence when building or extending a dataset folder — see [README.md](README.md) for full detail:

1. **Limitations analysis** (optional — skip if the dataset is being created from scratch rather than improving an existing one): analyze the source/original dataset and document why it's unsuitable for EDA teaching as-is.
2. **Spec**: built iteratively through dialogue with the user, not written solo — expect requests to pull stats on the existing dataset or fetch domain research mid-conversation. Document target features, their real-world/research grounding, feature relationships/correlations to build in, and the intentional data quality issues (missing values, noise/anomalies, duplicates) with precise rates, affected columns, and detection/cleaning approach. Open checklist items / open questions in the spec doc are a normal byproduct of this process, not a defect — don't tidy them away just to close them out.
3. **Generation scripts**: Python (`generate_dataset.py`, plus a `validate_dataset.py` that checks ranges, distributions, correlations and quality-issue counts against the spec), seeded for reproducibility. Loops with step 2: after generating, proactively validate against the spec yourself before handing off — the user does a manual check once it looks aligned, and mismatches typically send you back to adjusting the spec/algorithm and regenerating.
4. **Notebook**: a Jupyter notebook covering EDA (data quality assessment/cleaning, distributions, correlations, segmentation) and applied ML with scikit-learn (classification tasks, feature engineering, handling the injected data quality issues and any class imbalance). **This defaults to a review/validation notebook for the user, not student-facing teaching material** — no exercises or "for students" framing unless explicitly asked for as its own separate task. Don't assume a student-facing notebook is a pending deliverable just because a validation notebook exists.

[online-gaming/](online-gaming/) is the reference implementation of this process — its README.md, `DATA_GENERATION_SPEC.md`, `DATA_GENERATION_ALGORITHM.md`, `generate_dataset.py`, `validate_dataset.py`, and `analysis_demo.ipynb` show the expected shape and depth for each artifact in a new dataset folder.

## Repository Conventions

- One root `CLAUDE.md` (this file) and one root `README.md` — dataset folders do **not** get their own `CLAUDE.md`, only their own more specific `README.md`.
- Each dataset folder keeps its own `data/` subfolder for any original/source data it starts from, its own `requirements.txt`, and its own generation/validation scripts and notebook — folders are self-contained and don't share code.

## Datasets

### online-gaming/ (complete — reference implementation)

Synthetic online gaming behavior dataset, ~10,000 player-game combination rows across 21 features, improving on the original Kaggle ["Predict Online Gaming Behavior Dataset"](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset/data) which lacked logical correlations.

- Original dataset: `online-gaming/data/online_gaming_original.csv` (~40,000 records: PlayerID, Age, Gender, Location, GameGenre, PlayTimeHours, InGamePurchases, GameDifficulty, SessionsPerWeek, AvgSessionDurationMinutes, PlayerLevel, AchievementsUnlocked, EngagementLevel)
- Generated dataset: `online-gaming/generated_gaming_dataset.csv` — player-level attributes (PlayerID, Age, Gender, Location), game-specific attributes (GameID, GameName, GameGenre, GameDifficulty), behavioral & spending metrics, and two ML target variables (SpendingPropensity: NonSpender/Occasional/Whale; PlayerExpertise: Beginner/Intermediate/Expert)
- Design goals achieved: logical correlations (e.g. PlayTimeHours ↔ PlayerLevel), demographic influences on genre/spending preferences, multi-factorial (non-deterministic) target variables, and intentional data quality issues (duplicates, age-typo anomalies, missing values) for teaching data cleaning
- Full detail: `online-gaming/README.md`, `online-gaming/DATA_GENERATION_SPEC.md`, `online-gaming/DATA_GENERATION_ALGORITHM.md`

### coffee-health/ (first version complete, pending manual review)

Synthetic coffee consumption / lifestyle / self-rated health dataset, 10,040 rows (10,000 people + intentional duplicates) across Italy, France, UK, and Norway, improving on a modified Kaggle "Global Coffee Health Dataset" whose features had no meaningful correlations (e.g. Sleep Quality was purely a function of Sleep Hours, Stress Level was a 100% deterministic relabeling of Sleep Quality).

- Original dataset: `coffee-health/data/coffee_health_original.csv`; limitations quantified in `coffee-health/analyze_original_dataset.py`
- Single target: `SelfRatedHealth` {Poor, Fair, Good, Very Good, Excellent} — a real public-health survey construct, built as a multi-factor composite (Health Issues, BMI, Sleep Quality, Physical Activity, Smoking, Stress, Age, Country, Gender), calibrated against sourced country-level statistics
- Full detail: `coffee-health/README.md`, `coffee-health/DATA_GENERATION_SPEC.md`, `coffee-health/DATA_GENERATION_ALGORITHM.md`
- **Complete.** Spec, `generate_dataset.py` + `validate_dataset.py` (all checks passing), and `dataset_validation.ipynb` (the step-d review notebook — executes cleanly end-to-end, ~65% RF accuracy on the 5-class target, confirms all the intentional data quality issues and EDA patterns behave as designed) are all done, and the user has reviewed and approved the dataset. No student-facing notebook is planned — that was never requested; this dataset's notebook deliverable is the review notebook, full stop.