# Data Generation Specification — Coffee & Health

## Overview

This document captures the requirements for a redesigned coffee-consumption-and-health dataset, replacing the local `data/coffee_health_original.csv` (itself a modified-for-assignment version of Kaggle's ["Global Coffee Health Dataset"](https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset)).

**Target Size**: ~10,000 rows, **one row per person** (no multi-entity structure like online-gaming's player-game rows — there's no natural repeating unit here).

**Status**: First draft from the spec dialogue. Treat unchecked items and open questions as normal, not as things to rush to close — see the repo root README.md's note on the iterative spec process.

---

## Step (a): Limitations of the Current Dataset

Full quantitative analysis: `analyze_original_dataset.py` (run with `python3 analyze_original_dataset.py`). Summary of confirmed issues:

- **Sleep Quality** is a pure function of Sleep Hours (near-zero overlap between buckets), with ~0 relationship to caffeine, smoking, or alcohol.
- **Physical Activity** (numeric 0-15) correlates ~0 with Sleep Hours, BMI, Heart Rate, or Occupation — effectively random noise, contradicting its own documentation.
- **Stress Level** is **100% deterministic from Sleep Quality** (High stress → Poor sleep always, Medium → Fair always, Low → Good/Excellent). Zero independent signal — this is why it "felt like a feature" when used as a target last year.
- **Smoker / Drinks Alcohol** (binary) show no relationship to BMI, Heart Rate, or Stress — decorative, not linked to any outcome.
- **Daily Coffees / Caffeine Intake** are well-correlated (r=0.85, ~94.6mg/cup) but rigidly deterministic (cups × constant), with no cultural/brew-method variation.
- Missing-value rates are inconsistent (~0.7% baseline on most columns, but Sleep Hours 22.6%, Sleep Quality 15.7%, Health Issues 59.8%) and look like assignment-injected QA issues rather than something to preserve — this dataset will define its own missingness scheme from scratch (see Data Quality Issues below).
- Doc/data mismatches: Physical Activity is documented as categorical but stored as numeric; Age is documented as "whole years" but stored as float with a typo-style anomaly (max 199.7).

---

## Target Variable

**Variable name**: `SelfRatedHealth`
**Type**: Ordinal categorical
**Values**: `{Poor, Fair, Good, Very Good, Excellent}`

**Why this construct**: Self-rated/self-perceived health is a real, widely-used survey measure (Eurostat, WHO, national health surveys), not an invented label. It's subjective by design — which is realistic, since it doesn't reduce to one deterministic formula — and has a well-documented predictor set: sleep quality, physical activity, BMI, smoking, age, and psychological stress ([Sleep & self-rated health, elderly](https://doi.org/10.3390/su10113918), [Determinants of Self-Rated Health, Greece](https://pmc.ncbi.nlm.nih.gov/articles/PMC3367289/), [CDC longitudinal predictors](https://www.cdc.gov/pcd/issues/2014/13_0241.htm)).

### Contributing Factors (draft weights — open to tuning once generated & validated)
1. **Health Issues** (chronic condition severity) — strong negative weight
2. **BMI** — moderate negative weight outside a healthy range
3. **Sleep Quality** — moderate positive weight
4. **Physical Activity Level** — moderate positive weight
5. **Smoking Status** — negative weight, Current > Former > Never
6. **Stress Level** — moderate negative weight
7. **Age** — mild negative weight (older skews slightly lower, but not linear — captures general health decline without being deterministic)
8. **Country** — small direct effect on top of the above, reflecting real cross-national self-report differences (see Country Profiles below) — agreed in dialogue as a deliberate, modest effect

**Open question — new suggestion, not yet agreed**: real data shows a "gender-health paradox" — women tend to self-report lower health than men despite typically living longer. Worth a small `Gender` effect on the target for realism/discoverability? Flagging for your call rather than adding it unilaterally.

### Target Distribution
Should skew positive (matches real self-rated health surveys, where "Good" or better is the majority response) rather than being uniform across 5 classes. Country baselines below give a rough anchor; exact class thresholds to be tuned against validation once first-pass generation exists.

---

## Country Profiles

Four countries, chosen for meaningfully different coffee/lifestyle/health profiles while staying recognizable to UK students — Italy and Norway show that high coffee consumption doesn't predict poor health outcomes, while the UK has the lowest coffee consumption *and* the highest obesity, a genuinely non-obvious pattern for EDA.

| Country | Coffee (kg/yr) | Obesity (adult) | Smoking (current, adult) | Alcohol (L pure/capita) | Self-rated health "good+" | Happiness rank 2025 |
|---|---|---|---|---|---|---|
| Norway | 9.9 | ~23% | 14.2% | ~6 (est.) | ~80% | Top 10 |
| Italy | 5.9 (espresso culture) | ~13% (lowest) | 23.4% | 7.7 | 75.5% | mid-table |
| France | 5.4 | ~24% | 34.6% (highest) | 10.4 (highest) | 68.5% (declining — was 75.2% in 2017) | #33 |
| UK | 3.1, rising | ~30% (highest) | 12.5% (lowest) | 9.7 | not confidently sourced yet — **TODO** | Top 25 |

**Open item**: need a sourced UK self-rated-health figure before finalizing the country-effect magnitude (currently a gap, not a guess).

**Row distribution**: proposed roughly equal (~2,500/country) for balanced cross-country comparison — open to weighting by real population if you'd prefer that instead.

### Feature-to-country mapping (proposed)
- **Caffeine mg/cup**: vary by country (Italian espresso = smaller serving, higher concentration; Nordic filter coffee = larger volume, different mg profile) rather than a flat 95mg constant.
- **Smoking Status base rates**: Current-smoker % sourced above (France highest, UK lowest); Never/Former split per country is an **estimate**, not sourced — needs either research or an agreed simplifying assumption.
- **Alcohol Level base rates**: banded from L/capita figures above into `{None, Light, Moderate, Heavy}` — banding thresholds are an estimate, open to discussion.
- **Physical Activity baseline**: Norway skews more active (outdoor culture); others closer to WHO's ~28%-of-adults-insufficiently-active baseline — not yet country-differentiated beyond Norway, open item.
- **BMI baseline**: shifted per country's obesity rate above.

---

## Dataset Structure (proposed feature list)

### Demographics
1. **ID** — unique identifier, int, no missing, no duplicates (immune field)
2. **Age** — int, adult range (proposed 18-75)
3. **Gender** — {Male, Female, Other}
4. **Country** — {Italy, France, UK, Norway}
5. **Occupation** — {Office, Healthcare, Student, Service, Other} (kept from original)

### Coffee & Caffeine
6. **Daily Coffees** — cups, float
7. **Caffeine Intake** — mg, correlated with Daily Coffees but with country-varying mg/cup

### Lifestyle
8. **Smoking Status** — {Never, Former, Current} (replaces binary Smoker)
9. **Alcohol Level** — {None, Light, Moderate, Heavy} (replaces binary Drinks Alcohol)
10. **Physical Activity Level** — {Sedentary, Lightly Active, Moderately Active, Very Active} (standard 4-tier activity classification; replaces noisy 0-15 numeric) — driven by Age, Occupation, Country, Stress
11. **Sleep Hours** — float
12. **Sleep Quality** — {Poor, Fair, Good, Excellent} — rebuilt from Sleep Hours + Caffeine (esp. late-day) + Stress + Smoking + Alcohol + Age
13. **Stress Level** — {Low, Medium, High} — generated independently from Occupation, Age, Country (not derived from Sleep Quality)

### Physiology
14. **BMI** — float, country-shifted baseline
15. **Heart Rate** — float, linked to Physical Activity/Smoking/Stress
16. **Health Issues** — {None, Mild, Moderate, Severe} — chronic condition indicator, strengthened links to Age, BMI, Smoking

### Target
17. **SelfRatedHealth** — {Poor, Fair, Good, Very Good, Excellent} — see Target Variable section above

**Open question**: any features to drop entirely (vs. rework)? Current proposal reworks everything from the original 16 columns rather than dropping any — worth confirming that's the right call before locking in generation logic.

---

## Data Quality Issues (proposed — not yet discussed with you)

Following the online-gaming pattern (duplicates + typo-style anomalies + mixed feature/row-level missingness), proposed as a starting point:

- **Duplicate rows**: ~0.3-0.5%, exact copies
- **Age anomalies**: typo-style doubled-digit errors (e.g. 45 → 445), similar to online-gaming's approach — Age is the natural candidate since it's the one plausibly self-entered/manually-keyed field
- **Missing values**: feature-level missingness on 2-3 columns (candidates: Sleep Hours, Health Issues — both plausible to be genuinely under-reported in real health surveys) at a deliberately chosen rate (~5-10%, TBD) rather than inheriting the original's inconsistent rates; plus a handful of row-level incomplete records
- **Immune fields**: ID, Country, SelfRatedHealth (target)

This whole section is a placeholder for discussion, not a decision — flagging it now so it's not forgotten, per your point about the process.

---

## Open Questions

- [ ] UK self-rated health % — need a sourced figure
- [ ] Gender-health-paradox effect on target — add or skip?
- [ ] Never/Former smoker split per country — estimate acceptable, or worth deeper research?
- [ ] Alcohol Level banding thresholds from L/capita — estimate acceptable, or refine?
- [ ] Row distribution across countries — equal (~2,500 each) or population-weighted?
- [ ] Data quality issue rates — placeholder proposal above, needs your sign-off
- [ ] Exact target-variable weight formula — to be tuned once first-pass generation + validation exists (mirrors online-gaming's iterative b↔c loop)

## Notes & Decisions Log

- **2026-07-21**: Confirmed source dataset is Kaggle's "Global Coffee Health Dataset" (uom190346a); local copy has been further modified with different QA issues than the Kaggle original.
- **2026-07-21**: Target variable decided: `SelfRatedHealth` (5-level ordinal), chosen over "Overall Health Risk Tier" and "rebuilt Sleep Quality" alternatives.
- **2026-07-21**: Countries decided: Italy, France, UK, Norway.
- **2026-07-21**: Smoking/Alcohol to use categorical levels (not continuous units).
- **2026-07-21**: Country gets a small direct effect on the target, on top of indirect effects via lifestyle features.
- **2026-07-21**: Physical Activity to use the standard 4-tier classification {Sedentary, Lightly Active, Moderately Active, Very Active} rather than 3-tier Low/Medium/High, replacing the current noisy 0-15 numeric scale.
