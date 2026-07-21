# Synthetic Online Gaming Behavior Dataset

A coherent synthetic dataset designed for teaching **Exploratory Data Analysis (EDA)** and **Machine Learning classification** concepts to students. This dataset simulates realistic player behavior patterns across multiple online games, with built-in data quality issues for teaching data cleaning techniques.

## Dataset Overview

- **Size**: 10,037 rows × 21 columns
- **Structure**: Player-game combinations (~6,425 unique players across 9 games)
- **Format**: CSV file with intentional data quality issues
- **Purpose**: Educational - teaching EDA, data cleaning, and ML classification
- **Targets**: Two classification tasks (PlayerExpertise, SpendingPropensity)

## Features (21 columns)

### Player-Level Attributes (Consistent across all games for each PlayerID)
1. **PlayerID** (int): Unique player identifier (1001-7425)
2. **Age** (int): Player age in years (13-65, with intentional anomalies)
3. **Gender** (str): Player gender {Male, Female, Other}
   - Distribution varies by game genre (Action: 73% Male, RPG: 53% Male, Strategy: 58% Male)
4. **Location** (str): Player location {USA, Europe, Asia}

### Game-Specific Attributes
5. **GameID** (str): Unique game identifier (e.g., RPG_001, ACT_001, STR_001)
6. **GameName** (str): Human-readable game name (e.g., "Dragon's Quest", "Zombie Apocalypse")
7. **GameGenre** (str): Game genre {RPG, Action, Strategy}
8. **GameDifficulty** (str): Game difficulty level {Easy, Medium, Hard}

### Behavioral Metrics
9. **PlayTimeHours** (float): Total hours played for this game (0-500)
10. **SessionsPerWeek** (float): Average sessions per week (1-10)
11. **AvgSessionDurationMinutes** (float): Average session length in minutes (30-300)
    - *Note: Has intentional missing values (~8%)*
12. **PlayerLevel** (float): Current player level in game (1-100)
13. **AchievementsUnlocked** (float): Number of achievements earned (0-50)
    - *Note: Has intentional missing values (~6%)*
14. **EngagementLevel** (str): Engagement category {Low, Medium, High}
15. **DaysPlayed** (float): Total days with activity (1-365)

### Spending Metrics
16. **PurchaseCount** (float): Total in-game purchases (0-20)
17. **TotalSpend** (float): Total amount spent in USD ($0-200)
18. **AvgPurchasesPerMonth** (float): Average purchases per month
19. **AvgPurchaseValue** (float): Average value per purchase

### Target Variables (for ML Classification)
20. **SpendingPropensity** (str): Spending category {None, Occasional, Regular, High}
   - Derived from TotalSpend (data leakage warning for ML)
21. **PlayerExpertise** (str): Skill level {Beginner, Intermediate, Advanced}
   - Derived from PlayTimeHours and PlayerLevel

## Game Catalog

The dataset includes 9 games across 3 genres:

**RPG Games:**
- Dragon's Quest (RPG_001)
- Mystic Realms Online (RPG_002)
- Dungeon Crawler Deluxe (RPG_003)

**Action Games:**
- Battle Royale Extreme (ACT_001)
- Zombie Apocalypse (ACT_002)
- Street Fighter Ultimate (ACT_003)

**Strategy Games:**
- Empire Builder (STR_001)
- Tower Defense Masters (STR_002)
- Chess Legends Online (STR_003)

## Built-in Patterns for Discovery

The dataset contains realistic patterns for students to discover during EDA:

1. **Player Consistency**: Age, Gender, and Location remain constant for each PlayerID across all games
2. **Genre-Specific Demographics**: Gender distributions vary by game genre
3. **Engagement Patterns**: High engagement correlates with more playtime and achievements
4. **Spending Behavior**: Spending patterns correlate with engagement levels
5. **Game Difficulty Effects**: Difficulty influences achievement rates and playtime
6. **Regional Preferences**: Location affects game preferences and spending patterns

## Intentional Data Quality Issues

This dataset includes realistic data quality problems for teaching data cleaning:

### 1. Duplicate Rows (~0.37%)
- **Count**: ~37 exact duplicate rows
- **Detection**: Use `df.duplicated()`
- **Cleaning**: Remove with `df.drop_duplicates()`

### 2. Age Anomalies (~0.73%)
- **Type**: Typo-style errors (e.g., 166 instead of 16)
- **Pattern**: Double-typed digits, values 100-199
- **Count**: ~73 anomalous values
- **Detection**: Filter Age > 100 or Age < 13
- **Cleaning**: Pattern-based correction (e.g., 166 → 16)

### 3. Missing Values
**Feature-Level Missing Values:**
- `AvgSessionDurationMinutes`: ~8% missing
- `AchievementsUnlocked`: ~6% missing

**Row-Level Missing Values:**
- 5-10 rows with 4-6 missing values each
- Simulates incomplete records

**Immune Fields** (never missing):
- PlayerID, GameID, GameName
- PlayerExpertise, SpendingPropensity

**Detection**: Use `df.isnull().sum()` and `df.isnull().sum(axis=1)`

**Cleaning Approaches**:
- Drop rows with excessive missing values (>50% of columns)
- Impute feature-level missing values with median/mode
- Consider domain context when imputing

## Machine Learning Tasks

### Task 1: PlayerExpertise Classification (Easier)
**Target**: Predict player skill level {Beginner, Intermediate, Advanced}

**Relevant Features**:
- PlayTimeHours, PlayerLevel, AchievementsUnlocked
- SessionsPerWeek, AvgSessionDurationMinutes
- EngagementLevel, DaysPlayed

**Expected Performance**: ~85-90% accuracy with Random Forest

### Task 2: SpendingPropensity Classification (Harder)
**Target**: Predict spending behavior {None, Occasional, Regular, High}

**Important**: Exclude spending-related features to avoid data leakage!
- **Exclude**: TotalSpend, PurchaseCount, AvgPurchasesPerMonth, AvgPurchaseValue
- **Use**: Demographics, behavioral metrics, engagement patterns

**Expected Performance**: ~70-75% accuracy with Random Forest

**Teaching Point**: Demonstrates data leakage concept - including TotalSpend would give unrealistically high accuracy.

## Educational Applications

This dataset is ideal for teaching:

1. **Exploratory Data Analysis (EDA)**
   - Summary statistics and distributions
   - Correlation analysis
   - Visualization techniques (histograms, boxplots, heatmaps)
   - Group-by operations and aggregations

2. **Data Cleaning**
   - Handling duplicate records
   - Identifying and correcting anomalies
   - Managing missing values
   - Data validation techniques

3. **Feature Engineering**
   - Creating derived features
   - Encoding categorical variables
   - Feature scaling and normalization

4. **Machine Learning**
   - Classification algorithms (Random Forest, Logistic Regression)
   - Train-test splitting
   - Model evaluation metrics
   - Feature importance analysis
   - Data leakage concepts

5. **Critical Thinking**
   - Understanding data generation processes
   - Identifying biases and limitations
   - Making informed modeling decisions

## Usage Example

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('generated_gaming_dataset.csv')

# Explore the data
print(df.info())
print(df.describe())
print(df.head())

# Check data quality issues
print(f"Duplicates: {df.duplicated().sum()}")
print(f"Age anomalies: {len(df[df['Age'] > 100])}")
print(f"Missing values:\n{df.isnull().sum()}")

# Clean the data
df_clean = df.drop_duplicates()
df_clean = df_clean[df_clean['Age'] <= 100]
df_clean = df_clean.dropna(subset=['AvgSessionDurationMinutes'])

# Visualize patterns
sns.boxplot(data=df_clean, x='GameGenre', y='PlayTimeHours')
plt.show()

# Train a classifier for PlayerExpertise
feature_cols = ['PlayTimeHours', 'PlayerLevel', 'SessionsPerWeek',
                'EngagementLevel', 'AchievementsUnlocked']
X = pd.get_dummies(df_clean[feature_cols], drop_first=True)
y = df_clean['PlayerExpertise']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print(classification_report(y_test, rf.predict(X_test)))
```

## Dataset Generation

This dataset was generated using a sophisticated synthetic data generation algorithm that ensures:
- Realistic correlations between features
- Player-level consistency across multiple games
- Genre-specific behavioral patterns
- Controlled data quality issues for educational purposes

For technical details, see:
- `DATA_GENERATION_SPEC.md` - High-level specification
- `DATA_GENERATION_ALGORITHM.md` - Detailed algorithm documentation
- `generate_dataset.py` - Implementation code

## Validation

The dataset includes a validation script (`validate_dataset.py`) that checks:
- Data types and value ranges
- Player consistency (Age, Gender, Location per PlayerID)
- Statistical distributions and correlations
- Data quality issue counts
- Feature relationships

Run validation: `python validate_dataset.py`

## Demo Notebook

An interactive Jupyter notebook (`analysis_demo.ipynb`) demonstrates:
- Complete EDA workflow
- Data quality assessment and cleaning
- Visualization techniques
- ML model training and evaluation
- Best practices for both classification tasks

## Files Included

- `generated_gaming_dataset.csv` - The main dataset
- `generate_dataset.py` - Dataset generation script
- `validate_dataset.py` - Validation and quality checks
- `analysis_demo.ipynb` - Interactive demo notebook
- `DATA_GENERATION_SPEC.md` - Feature specifications
- `DATA_GENERATION_ALGORITHM.md` - Algorithm details
- `CLAUDE.md` - AI assistant context

## License

This is a synthetic dataset created for educational purposes. Feel free to use it in teaching, tutorials, workshops, or any educational context.

## Citation

If you use this dataset in educational materials or publications, please cite:

```
Synthetic Online Gaming Behavior Dataset (2025)
Generated for teaching EDA and Machine Learning classification
https://github.com/[your-repo]/online-gaming-dataset
```

## Questions or Feedback?

This dataset is designed to provide a realistic, coherent learning experience while maintaining controlled complexity suitable for educational environments. The intentional data quality issues mirror real-world challenges students will encounter in professional data science work.