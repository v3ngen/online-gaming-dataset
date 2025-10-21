# Data Generation Algorithm Specification

## Overview

This document specifies the exact algorithm for generating the coherent online gaming behavior dataset. The implementation will be in Python using numpy, pandas, and scipy.

**Target**: 10,000 player-game combination rows

---

## Generation Order & Dependencies

Features must be generated in dependency order:

```
Level 0 (Independent):
  - PlayerID, Age, Gender, Location

Level 1 (Player-Game Assignment):
  - GameID, GameGenre

Level 2 (Core Behavioral - Time-based):
  - DaysPlayed, SessionsPerWeek, AvgSessionDurationMinutes

Level 3 (Derived Behavioral):
  - PlayTimeHours

Level 4 (Game Progression):
  - PlayerLevel, GameDifficulty

Level 5 (Achievements & Engagement):
  - AchievementsUnlocked, EngagementLevel

Level 6 (Spending):
  - PurchaseCount, TotalSpend

Level 7 (Spending Derived):
  - AvgPurchasesPerMonth, AvgPurchaseValue

Level 8 (Target Variables):
  - PlayerExpertise, SpendingPropensity
```

---

## Step-by-Step Generation Process

### LEVEL 0: Player Demographics (Independent Features)

#### Step 1: Generate Unique Players
```
num_players = ~4,000-5,000 (to get 10,000 rows with multi-game)
PlayerID = range(1, num_players + 1)
```

#### Step 2: Generate Gender
```
Gender distribution (overall):
- Male: 63%
- Female: 35%
- Other: 2%

Implementation: np.random.choice(['Male', 'Female', 'Other'], p=[0.63, 0.35, 0.02])
```

#### Step 3: Generate Location
```
Location distribution:
- USA: 40%
- Europe: 35%
- Asia: 25%

Implementation: np.random.choice(['USA', 'Europe', 'Asia'], p=[0.40, 0.35, 0.25])
```

#### Step 4: Generate Age
```
Age distribution by genre preference (we'll use to bias later):
- Overall range: 13-65 years
- Distribution: Beta distribution shifted and scaled
  - Mode around 25-30 years (peak gaming demographic)
  - Long tail to 65

Implementation:
base_age = np.random.beta(2, 5, size=n) * 52 + 13  # Beta(2,5) scaled to 13-65
Age = np.round(base_age).astype(int)
```

---

### LEVEL 1: Game Assignment (Location & Demographics Dependent)

#### Step 5: Determine Number of Games per Player
```
Distribution:
- 1 game: 60%
- 2 games: 25%
- 3 games: 12%
- 4+ games: 3%

Implementation: Weighted random choice, then create player-game rows
```

#### Step 6: Assign GameID and GameGenre

**Game Catalog**:
```python
GAMES = {
    'RPG_001': {'genre': 'RPG', 'max_level': 100, 'max_achievements': 120,
                'typical_hours': (200, 800), 'monetization': 'high'},
    'RPG_002': {'genre': 'RPG', 'max_level': 80, 'max_achievements': 150,
                'typical_hours': (300, 2000), 'monetization': 'very_high'},
    'RPG_003': {'genre': 'RPG', 'max_level': 75, 'max_achievements': 85,
                'typical_hours': (50, 300), 'monetization': 'medium'},
    'ACT_001': {'genre': 'Action', 'max_level': 100, 'max_achievements': 110,
                'typical_hours': (100, 1000), 'monetization': 'high'},
    'ACT_002': {'genre': 'Action', 'max_level': 50, 'max_achievements': 75,
                'typical_hours': (30, 200), 'monetization': 'low_medium'},
    'ACT_003': {'genre': 'Action', 'max_level': 60, 'max_achievements': 90,
                'typical_hours': (50, 500), 'monetization': 'medium'},
    'STR_001': {'genre': 'Strategy', 'max_level': 50, 'max_achievements': 130,
                'typical_hours': (100, 800), 'monetization': 'medium'},
    'STR_002': {'genre': 'Strategy', 'max_level': 75, 'max_achievements': 80,
                'typical_hours': (40, 200), 'monetization': 'medium'},
    'STR_003': {'genre': 'Strategy', 'max_level': 30, 'max_achievements': 50,
                'typical_hours': (50, 500), 'monetization': 'low_medium'},
}
```

**Genre Selection with Location Bias**:
```python
def get_genre_probabilities(location):
    if location == 'USA':
        return {'RPG': 0.35, 'Action': 0.45, 'Strategy': 0.20}
    elif location == 'Europe':
        return {'RPG': 0.35, 'Action': 0.35, 'Strategy': 0.30}
    elif location == 'Asia':
        return {'RPG': 0.50, 'Action': 0.30, 'Strategy': 0.20}
```

**Gender Distribution by Genre** (validate after generation):
```
Action: 73% Male, 25% Female, 2% Other
RPG: 53% Male, 45% Female, 2% Other
Strategy: 58% Male, 40% Female, 2% Other
```
*Note: This is achieved through biased sampling, not hard constraints*

**Age Bias by Genre**:
```python
def adjust_age_for_genre(age, genre):
    # Younger players prefer Action
    if genre == 'Action' and age > 35:
        # Reduce probability of Action for older players
        pass
    # Older players prefer Strategy
    elif genre == 'Strategy' and age < 22:
        # Reduce probability of Strategy for younger players
        pass
    # RPG middle-age
    # Neutral, wide age range
```

**Specific Game Selection**:
- Within genre, use weighted random (popular games more likely)
- RPG_002 (MMORPG) more popular in Asia
- ACT_001 (Battle Royale) more popular in USA
- STR_001 (Grand Strategy) more popular in Europe

---

### LEVEL 2: Core Behavioral Metrics

#### Step 7: Generate DaysPlayed
```
Range: 7-1095 days (1 week to 3 years)
Distribution: Log-normal (most players 30-180 days, some long-term veterans)

Implementation:
mu = 4.5  # log-space mean
sigma = 1.0  # log-space std
days = np.random.lognormal(mu, sigma)
DaysPlayed = np.clip(days, 7, 1095).astype(int)
```

#### Step 8: Generate SessionsPerWeek
```
Range: 1-21 sessions/week
Distribution: Varies by genre

Genre-specific parameters:
- Action: mean=8, std=4, range(3-18)
- RPG: mean=6, std=3, range(2-15)
- Strategy: mean=5, std=2.5, range(2-12)

Implementation: Gamma distribution, clipped and rounded
```

#### Step 9: Generate AvgSessionDurationMinutes
```
Range: 15-240 minutes
Distribution: Varies by genre

Genre-specific parameters:
- Action: mean=60, std=25, range(30-120)
- RPG: mean=120, std=40, range(60-240)
- Strategy: mean=150, std=50, range(60-240)

Implementation: Gamma distribution, clipped and rounded
```

---

### LEVEL 3: Derived Behavioral

#### Step 10: Generate PlayTimeHours
```
Base formula:
PlayTimeHours ≈ (SessionsPerWeek × AvgSessionDuration × DaysPlayed) / (7 × 60)

Add variance:
- Noise factor: ±20% random variation
- Clamp to game-specific ranges (from GAMES catalog)

Implementation:
estimated_hours = (sessions_per_week * avg_duration * days_played) / 420
noise = np.random.uniform(0.8, 1.2)
PlayTimeHours = estimated_hours * noise
# Clamp to game-specific typical_hours range
```

---

### LEVEL 4: Game Progression

#### Step 11: Generate PlayerExpertise (EARLY - needed for other calculations)
```
This is a TARGET variable, but we generate it early to influence other features.

Three-stage approach:
1. Generate "expertise_seed" score (0-100)
2. Add noise and dependencies later
3. Convert to {Beginner, Intermediate, Expert} at the end

Expertise seed factors:
- Base: Random uniform(0-100)
- Age influence: Slight boost for 20-35 age range
- Genre experience: If player has multiple games in same genre, boost

Distribution target:
- Beginner: 35%
- Intermediate: 45%
- Expert: 20%

Thresholds (to be applied in Step 19):
- Beginner: expertise_score < 40
- Intermediate: 40 ≤ expertise_score < 75
- Expert: expertise_score ≥ 75
```

#### Step 12: Generate PlayerLevel
```
Formula:
PlayerLevel = f(PlayTimeHours, expertise_seed, max_level, noise)

Base progression rate (hours to max level):
- High expertise (>75): 0.7x time needed
- Medium expertise (40-75): 1.0x time needed
- Low expertise (<40): 1.5x time needed

Example for RPG_001 (max level 100):
- Expert: reaches 100 in ~300 hours
- Intermediate: reaches 100 in ~500 hours
- Beginner: reaches 100 in ~800 hours

Implementation:
progression_rate = get_progression_rate(expertise_seed)
expected_max_hours = typical_hours_mean * progression_rate
level_progress = (PlayTimeHours / expected_max_hours)
PlayerLevel = min(max_level, level_progress * max_level + noise)
PlayerLevel = max(1, round(PlayerLevel))
```

#### Step 13: Generate GameDifficulty
```
Distribution:
- Easy: 40%
- Medium: 40%
- Hard: 20%

Biases:
- Higher PlayerLevel → more likely Hard
- Higher expertise_seed → more likely Hard
- Beginners rarely choose Hard

Implementation:
base_probs = [0.40, 0.40, 0.20]  # Easy, Medium, Hard

# Adjust based on expertise_seed
if expertise_seed > 70:
    probs = [0.20, 0.40, 0.40]  # Experts prefer Hard
elif expertise_seed > 40:
    probs = [0.35, 0.45, 0.20]  # Intermediate
else:
    probs = [0.55, 0.35, 0.10]  # Beginners prefer Easy

GameDifficulty = np.random.choice(['Easy', 'Medium', 'Hard'], p=probs)
```

---

### LEVEL 5: Achievements & Engagement

#### Step 14: Generate AchievementsUnlocked
```
Formula:
Achievements = f(PlayTimeHours, PlayerLevel, max_achievements, expertise_seed, completionist_factor)

Completionist factor (random per player):
- 30% of players are completionists (80-100% achievements)
- 50% are moderate (30-60% achievements)
- 20% are casual (5-30% achievements)

Base achievement rate:
achievement_rate = (PlayerLevel / max_level) * completionist_factor * expertise_modifier

Implementation:
is_completionist = np.random.rand() < 0.30
if is_completionist:
    target_pct = np.random.uniform(0.7, 1.0)
elif np.random.rand() < 0.714:  # 50/70 of remainder
    target_pct = np.random.uniform(0.3, 0.7)
else:
    target_pct = np.random.uniform(0.05, 0.3)

progress_factor = min(1.0, PlayerLevel / max_level)
AchievementsUnlocked = round(max_achievements * target_pct * progress_factor)
```

#### Step 15: Generate EngagementLevel
```
Formula based on multiple factors:
engagement_score = f(PlayTimeHours, SessionsPerWeek, DaysPlayed, AchievementsUnlocked)

Scoring system:
1. PlayTime score (0-40 points):
   - Low: <50 hours = 0-15 points
   - Medium: 50-200 hours = 15-30 points
   - High: >200 hours = 30-40 points

2. Session frequency score (0-30 points):
   - Low: <3 sessions/week = 0-10 points
   - Medium: 3-8 sessions/week = 10-20 points
   - High: >8 sessions/week = 20-30 points

3. Tenure score (0-20 points):
   - Short: <60 days = 0-7 points
   - Medium: 60-365 days = 7-15 points
   - Long: >365 days = 15-20 points

4. Achievement score (0-10 points):
   - Based on achievement_percentage: achievements_unlocked / max_achievements * 10

Total engagement_score (0-100):
- Low: score < 40
- Medium: 40 ≤ score < 70
- High: score ≥ 70

Target distribution:
- Low: ~30%
- Medium: ~45%
- High: ~25%
```

---

### LEVEL 6: Spending Behavior

#### Step 16: Generate SpendingPropensity (EARLY - seed for spending)
```
This is a TARGET variable, but generated early to influence spending features.

Three-stage approach:
1. Generate "spending_seed" score (0-100)
2. Use to generate PurchaseCount and TotalSpend
3. Finalize SpendingPropensity category at end

Spending seed factors:
1. EngagementLevel (40% weight):
   - High engagement = +30 points
   - Medium engagement = +15 points
   - Low engagement = +0 points

2. Game/Genre monetization (25% weight):
   - RPG_002 (MMORPG): +25 points
   - ACT_001 (Battle Royale): +20 points
   - High monetization games: +15 points
   - Medium monetization: +10 points
   - Low monetization: +5 points

3. Location (15% weight):
   - Asia: +15 points (higher whale culture)
   - USA: +10 points
   - Europe: +5 points

4. Age (10% weight):
   - Age 25-45: +10 points (disposable income)
   - Age 18-24: +5 points
   - Age 45+: +8 points
   - Age <18: +0 points

5. PlayTimeHours (10% weight):
   - >500 hours: +10 points
   - 200-500 hours: +6 points
   - <200 hours: +2 points

6. Random variance (to add unpredictability):
   - ±15 points random

spending_seed = sum of all factors, clipped to 0-100

Distribution target:
- NonSpender: 55% (spending_seed < 25)
- Occasional: 35% (25 ≤ spending_seed < 65)
- Whale: 10% (spending_seed ≥ 65)
```

#### Step 17: Generate PurchaseCount
```
Based on spending_seed:

NonSpender (spending_seed < 25):
  PurchaseCount = 0

Occasional (25 ≤ spending_seed < 65):
  mean_purchases = 1 + (spending_seed - 25) / 4  # 1-11 range
  PurchaseCount = max(1, round(np.random.poisson(mean_purchases)))

Whale (spending_seed ≥ 65):
  mean_purchases = 10 + (spending_seed - 65) * 0.5  # 10-27.5 range
  PurchaseCount = max(10, round(np.random.poisson(mean_purchases)))

Influenced by:
- DaysPlayed (more days = more purchases)
- EngagementLevel (high = more purchases)
```

#### Step 18: Generate TotalSpend
```
Based on spending_seed and PurchaseCount:

NonSpender:
  TotalSpend = 0.0

Occasional:
  avg_purchase_value = uniform(2, 25) GBP
  TotalSpend = PurchaseCount * avg_purchase_value * random_variance
  TotalSpend = clip(1, 80) GBP

Whale:
  Base calculation:
  - If Asia + RPG_002: avg_purchase_value = uniform(30, 150) GBP
  - If ACT_001: avg_purchase_value = uniform(20, 80) GBP
  - Other high monetization: avg_purchase_value = uniform(15, 60) GBP

  TotalSpend = PurchaseCount * avg_purchase_value * random_variance
  TotalSpend = max(80, TotalSpend)  # Whales spend at least £80

Correlations to maintain:
- Strong positive correlation with EngagementLevel
- Strong positive correlation with PlayTimeHours
- Strong positive correlation with DaysPlayed
- Game-specific: RPG_002, ACT_001 have higher spending
```

---

### LEVEL 7: Spending Derived Features

#### Step 19: Calculate AvgPurchasesPerMonth
```
Formula:
AvgPurchasesPerMonth = PurchaseCount / (DaysPlayed / 30)

Handle edge cases:
- If DaysPlayed < 30: use DaysPlayed / 30 directly
- If PurchaseCount = 0: AvgPurchasesPerMonth = 0.0

Round to 2 decimal places
```

#### Step 20: Calculate AvgPurchaseValue
```
Formula:
AvgPurchaseValue = TotalSpend / PurchaseCount

Handle edge cases:
- If PurchaseCount = 0: AvgPurchaseValue = 0.0

Round to 2 decimal places

Expected patterns:
- Occasional spenders: £2-25 avg
- Whales: varies widely (£15-150 avg)
  - Small frequent purchases (£15-30 avg, many purchases)
  - Large occasional purchases (£80-150 avg, fewer purchases)
```

---

### LEVEL 8: Finalize Target Variables

#### Step 21: Finalize PlayerExpertise
```
Recalculate expertise_score with all available features:

Components (weighted scoring, 0-100):

1. GameDifficulty (20% weight):
   - Easy: 0 points
   - Medium: 10 points
   - Hard: 20 points

2. Efficiency - Achievements per 100 hours (25% weight):
   - efficiency = (AchievementsUnlocked / max_achievements) / (PlayTimeHours / 100)
   - Normalize to 0-25 points

3. Consistency - SessionsPerWeek (15% weight):
   - Low (<3): 0-5 points
   - Medium (3-8): 5-10 points
   - High (>8): 10-15 points

4. Skill Progression - Level per 100 hours (25% weight):
   - progression = (PlayerLevel / max_level) / (PlayTimeHours / 100)
   - Normalize to 0-25 points

5. Challenge-seeking - Difficulty + Achievements (15% weight):
   - Combination metric: (difficulty_score + achievement_percentage) / 2
   - Scale to 0-15 points

Total expertise_score (0-100)

Final categories:
- Beginner: expertise_score < 40 (target 35%)
- Intermediate: 40 ≤ expertise_score < 75 (target 45%)
- Expert: expertise_score ≥ 75 (target 20%)

Adjust thresholds slightly if distribution doesn't match target
```

#### Step 22: Finalize SpendingPropensity
```
Recalculate spending category based on actual spending metrics:

Primary rule (based on TotalSpend):
- TotalSpend = £0 → NonSpender
- £0 < TotalSpend ≤ £80 → Occasional
- TotalSpend > £80 → Whale

Secondary validation (check consistency):
- If NonSpender: PurchaseCount must = 0
- If Occasional: PurchaseCount typically 1-15
- If Whale: PurchaseCount typically 10+

Target distribution:
- NonSpender: 55%
- Occasional: 35%
- Whale: 10%

If distribution is off, adjust spending_seed thresholds in Step 16 and regenerate
```

---

## Parameter Summary

### Distribution Parameters

```python
CONFIG = {
    'num_rows': 10000,
    'num_players': 4500,  # Approximate, adjusted for multi-game

    # Demographics
    'gender_dist': [0.63, 0.35, 0.02],  # Male, Female, Other
    'location_dist': [0.40, 0.35, 0.25],  # USA, Europe, Asia
    'age_range': (13, 65),
    'age_beta_params': (2, 5),

    # Multi-game distribution
    'games_per_player': {1: 0.60, 2: 0.25, 3: 0.12, 4: 0.03},

    # Behavioral ranges
    'days_played': {'mu': 4.5, 'sigma': 1.0, 'range': (7, 1095)},
    'sessions_per_week': {
        'Action': {'mean': 8, 'std': 4, 'range': (3, 18)},
        'RPG': {'mean': 6, 'std': 3, 'range': (2, 15)},
        'Strategy': {'mean': 5, 'std': 2.5, 'range': (2, 12)},
    },
    'session_duration': {
        'Action': {'mean': 60, 'std': 25, 'range': (30, 120)},
        'RPG': {'mean': 120, 'std': 40, 'range': (60, 240)},
        'Strategy': {'mean': 150, 'std': 50, 'range': (60, 240)},
    },

    # Target distributions
    'player_expertise': {'Beginner': 0.35, 'Intermediate': 0.45, 'Expert': 0.20},
    'spending_propensity': {'NonSpender': 0.55, 'Occasional': 0.35, 'Whale': 0.10},
    'engagement_level': {'Low': 0.30, 'Medium': 0.45, 'High': 0.25},

    # Correlation strengths (Pearson r targets)
    'correlations': {
        'PlayTimeHours_PlayerLevel': 0.75,  # Strong positive
        'PlayTimeHours_Achievements': 0.60,  # Moderate-strong positive
        'DaysPlayed_PlayTimeHours': 0.70,  # Strong positive
        'PurchaseCount_TotalSpend': 0.85,  # Very strong positive
        'EngagementLevel_TotalSpend': 0.50,  # Moderate positive (categorical-continuous)
    },
}
```

---

## Validation Checks

After generation, validate:

1. **Feature Ranges**: All values within specified min/max
2. **Distributions**: Gender, Location, Genre match targets (±3%)
3. **Target Variables**: Expertise and Spending match distributions (±3%)
4. **Correlations**: Key correlations within ±0.10 of targets
5. **Logical Consistency**:
   - NonSpenders have PurchaseCount=0, TotalSpend=0
   - AvgPurchaseValue = TotalSpend / PurchaseCount (when PurchaseCount>0)
   - AvgPurchasesPerMonth = PurchaseCount / (DaysPlayed/30)
   - PlayTimeHours roughly consistent with Sessions × Duration × Days
6. **Multi-game Players**: Same PlayerID has same Age, Gender, Location
7. **No Missing Values**: All fields populated

---

## Output Format

CSV file with columns in order:
```
PlayerID, GameID, Age, Gender, Location, GameGenre, GameDifficulty,
PlayTimeHours, SessionsPerWeek, AvgSessionDurationMinutes, PlayerLevel,
AchievementsUnlocked, EngagementLevel, DaysPlayed, PurchaseCount,
TotalSpend, AvgPurchasesPerMonth, AvgPurchaseValue,
PlayerExpertise, SpendingPropensity
```

---

## Notes for Implementation

### Random Seed
Set a random seed for reproducibility:
```python
np.random.seed(42)  # Or allow as parameter
```

### Noise Addition
When adding noise to maintain realism:
- Use multiplicative noise for positive-only values: `value * uniform(0.9, 1.1)`
- Use additive noise for values that can be zero: `value + normal(0, std)`
- Clip to valid ranges after noise

### Correlation Management
Some correlations emerge naturally from generation order (e.g., PlayTimeHours → PlayerLevel).
Others may need post-generation adjustment using copulas or Cholesky decomposition if not achieving targets.

### Edge Case Handling
- Brand new players (7-14 days): Low levels, few achievements, likely NonSpenders
- Veteran players (700+ days): High variance in engagement/spending
- Inactive players: Not included (minimum 1 session/week average)

---

## Questions/Decisions Needed

1. **Expertise calculation weights**: Are the percentages (20%, 25%, 15%, 25%, 15%) reasonable?
2. **Spending seed formula**: Does the multi-factor approach make sense?
3. **Correlation targets**: Should we aim for stronger/weaker correlations anywhere?
4. **Edge cases**: Should we include any completely inactive players (0 sessions in recent period)?
5. **Validation thresholds**: Are ±3% for distributions and ±0.10 for correlations acceptable?

---

## Next Steps

1. Review and refine this algorithm specification
2. Implement `generate_dataset.py` based on this spec
3. Create `validate_dataset.py` for verification (Pandas + SciPy)
4. Generate test dataset and validate
5. Iterate on parameters as needed
