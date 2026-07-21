# Dataset Generation Guide

## Overview

This directory contains scripts to generate and validate a coherent synthetic online gaming behavior dataset for educational purposes (EDA and Applied Machine Learning).

## Files

- `DATA_GENERATION_SPEC.md` - High-level specification of dataset requirements
- `DATA_GENERATION_ALGORITHM.md` - Detailed algorithm specification with formulas
- `generate_dataset.py` - Python script that generates the dataset
- `validate_dataset.py` - Python script that validates the generated dataset
- `requirements.txt` - Python dependencies
- `generated_gaming_dataset.csv` - The generated dataset (after running scripts)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Requirements: Python 3.8+, numpy, pandas, scipy

### 2. Generate Dataset

```bash
python3 generate_dataset.py --rows 10000 --seed 42 --output generated_gaming_dataset.csv
```

Options:
- `--rows`: Target number of rows (default: 10000)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output`: Output CSV filename (default: generated_gaming_dataset.csv)

### 3. Validate Dataset

```bash
python3 validate_dataset.py generated_gaming_dataset.csv --eda
```

Options:
- `--eda`: Show additional EDA insights (optional)

## Dataset Features

The generated dataset contains 20 features:

### Identifiers
- PlayerID, GameID

### Demographics
- Age, Gender, Location

### Game Context
- GameGenre, GameDifficulty

### Behavioral Metrics
- PlayTimeHours, SessionsPerWeek, AvgSessionDurationMinutes
- PlayerLevel, AchievementsUnlocked
- EngagementLevel, DaysPlayed

### Spending Behavior
- PurchaseCount, TotalSpend, AvgPurchasesPerMonth, AvgPurchaseValue

### Target Variables (for ML)
- **PlayerExpertise**: {Beginner, Intermediate, Expert} - Multi-factorial skill level
- **SpendingPropensity**: {NonSpender, Occasional, Whale} - Spending behavior category

## Key Patterns Built Into the Dataset

### For EDA Discovery
- Strong correlation between PlayTimeHours and PlayerLevel
- Location-genre preferences (USA→Action, Europe→Strategy, Asia→RPG)
- Gender-genre patterns (RPG most balanced, Action male-dominated)
- Age-genre preferences (younger→Action, older→Strategy)
- Spending correlates with engagement, playtime, game type, location

### For ML Classification
- **PlayerExpertise**: Harder task requiring feature engineering
  - Factors: GameDifficulty, efficiency metrics, consistency, progression rate
- **SpendingPropensity**: Easier task with clearer patterns
  - Factors: Engagement, playtime, game type, location, age

## Validation Checks

The validation script verifies:
1. No missing values
2. Correct data types
3. Values within expected ranges
4. Distribution targets (gender, location, genre, targets)
5. Key correlations (PlayTime↔Level, PurchaseCount↔TotalSpend, etc.)
6. Logical consistency (NonSpenders have £0 spend, derived fields calculated correctly)
7. Multi-game player consistency (same demographics across games)

## Customization

To adjust parameters, edit `generate_dataset.py`:
- **Line 19-67**: `Config` class - adjust distributions and ranges
- **Line 70-110**: `GAMES` catalog - modify game specifications
- **Line 679-723**: Expertise scoring - adjust weights and thresholds
- **Line 561-670**: Spending generation - adjust monetization logic

## Notes

- Dataset size: Target is 10,000 rows, but actual may be ~7,000 due to multi-game distribution
- Unique players: ~4,500 players with 1-4 games each (realistic distribution)
- Reproducibility: Same seed produces same dataset
- Currency: All spending in GBP (normalized across global regions)
- Games: 9 total (3 RPG, 3 Action, 3 Strategy)

## Troubleshooting

**Issue**: Distributions don't match targets exactly

- Small variations (±5%) are acceptable and add realism
- Adjust thresholds in the scoring functions if needed
- Larger datasets (>10k rows) converge closer to targets

**Issue**: Python command not found

- Use `python3` instead of `python`
- Ensure Python 3.8+ is installed

**Issue**: Module not found errors

- Run `pip install -r requirements.txt`
- May need `pip3` instead of `pip`

## Educational Use

This dataset is designed for teaching:

### EDA Exercises
- Distribution analysis
- Correlation exploration
- Demographic segmentation
- Cross-tabulation and pivot tables
- Visualization practice

### ML Exercises
- Binary/multi-class classification
- Feature engineering (aggregate by player, derive ratios)
- Handling class imbalance
- Model comparison (different algorithms)
- Cross-validation and evaluation

### Key Learning Points
- Real-world patterns (location biases, demographic preferences)
- Business context (monetization, player retention)
- Multi-table thinking (player-game granularity)
- Feature engineering opportunities

## Next Steps

1. Generate the dataset with your desired parameters
2. Run validation to verify quality
3. Perform initial EDA to confirm patterns
4. Use for teaching EDA and ML concepts
5. Iterate on parameters based on student feedback
