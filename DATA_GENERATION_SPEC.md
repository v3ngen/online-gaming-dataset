# Data Generation Specification

## Overview
This document captures the detailed requirements and specifications for generating the coherent online gaming behavior dataset.

**Target Size**: ~10,000 player records

---

## Target Variables (ML Prediction Tasks)

We have **TWO** target variables to provide students with different ML learning opportunities:

### 1. Player Expertise Level
**Variable Name**: `PlayerExpertise`
**Type**: Categorical
**Values**: `{Beginner, Intermediate, Expert}`
**Class Distribution**: 35% Beginner, 45% Intermediate, 20% Expert (imbalanced)

**Key Principles**:
- This is a NEW variable, not directly in the original dataset
- Should relate to `PlayerLevel` but NOT directly/deterministically
  - Rationale: Players can reach high/max level in games without being truly expert (e.g., through time investment alone)
- True expertise is a function of MULTIPLE factors
- **Design Goal**: Multi-factorial definition requiring feature engineering and discovery of non-obvious patterns

**Contributing Factors**:
1. **GameDifficulty** - Experts gravitate toward Hard, but not exclusively (non-deterministic)
2. **Efficiency Metrics** - Achievements per PlayTime ratio (experts achieve more per hour)
3. **Consistency** - High SessionsPerWeek + focused play patterns
4. **Skill Progression** - PlayerLevel relative to PlayTimeHours (experts level faster)
5. **Challenge-Seeking** - Combination of difficulty setting + achievements unlocked

### 2. Spending Propensity
**Variable Name**: `SpendingPropensity`
**Type**: Categorical
**Values**: `{NonSpender, Occasional, Whale}`
**Class Distribution**: 55% NonSpender, 35% Occasional, 10% Whale (highly imbalanced - realistic F2P distribution)

**Key Principles**:
- Represents player's in-game spending behavior
- Clear business application (monetization, player segmentation)
- Easier to learn/teach than PlayerExpertise (more obvious relationships)
- **Design Goal**: Realistic business problem with clear but non-trivial patterns

**Contributing Factors**:
1. **EngagementLevel** - High engagement → more spending likelihood
2. **PlayTimeHours** - Invested time → invested money (sunk cost fallacy)
3. **GameGenre** - Some genres monetize better (e.g., RPG/Strategy > Sports)
4. **Age** - Disposable income patterns (older players may spend more)
5. **AchievementsUnlocked** - Completionists spend more to unlock content
6. **InGamePurchases** (original feature) - Should be strongly correlated but with noise

---

## Feature Definitions & Relationships

### Original Features (from source dataset)
1. **PlayerID** - Unique identifier
2. **Age** - Player age
3. **Gender** - Player gender
4. **Location** - Geographic location
5. **GameGenre** - Type of game played
6. **PlayTimeHours** - Total play time
7. **InGamePurchases** - Whether player makes purchases
8. **GameDifficulty** - Difficulty setting {Easy, Medium, Hard}
9. **SessionsPerWeek** - Number of gaming sessions per week
10. **AvgSessionDurationMinutes** - Average session length
11. **PlayerLevel** - In-game level achieved
12. **AchievementsUnlocked** - Number of achievements
13. **EngagementLevel** - {Low, Medium, High}

### Relationships to Define

#### PlayerExpertise Dependencies
- [ ] **GameDifficulty**: Experts gravitate toward Hard (strong but not deterministic)
- [ ] **Efficiency (Achievements/PlayTime)**: Experts achieve more per hour
- [ ] **Consistency (SessionsPerWeek)**: Experts play regularly
- [ ] **Skill Progression (PlayerLevel/PlayTimeHours ratio)**: Experts level faster
- [ ] **Challenge-Seeking**: Combination of difficulty + achievements

#### SpendingPropensity Dependencies
- [ ] **EngagementLevel**: High engagement → more spending
- [ ] **PlayTimeHours**: More time invested → more spending
- [ ] **GameGenre**: RPG/Strategy players spend more than Sports players
- [ ] **Age**: Older players (disposable income) spend more
- [ ] **AchievementsUnlocked**: Completionists spend more
- [ ] **InGamePurchases (original)**: Should correlate strongly but with noise

#### Natural Correlations (for EDA discovery)
- [ ] **PlayTimeHours ↔ PlayerLevel**: Define correlation strength
- [ ] **PlayTimeHours ↔ AchievementsUnlocked**: Define relationship
- [ ] **Age ↔ GameGenre**: Define demographic preferences
- [ ] **Gender ↔ GameGenre**: Define demographic preferences
- [ ] **SessionsPerWeek + AvgSessionDurationMinutes ↔ PlayTimeHours**: Define consistency
- [ ] **EngagementLevel**: Define as function of which factors?
- [ ] Additional correlations: TBD

#### Domain Rules & Constraints
- [ ] **Age range**: Define realistic bounds
- [ ] **PlayerLevel range**: Define per game or overall?
- [ ] **SessionsPerWeek**: Define realistic bounds (0-?)
- [ ] **AvgSessionDurationMinutes**: Define realistic bounds
- [ ] **PlayTimeHours**: Define realistic bounds
- [ ] **AchievementsUnlocked**: Define realistic bounds
- [ ] **Location distribution**: Keep original? (USA, Europe, Asia, Other)
- [ ] **GameGenre distribution**: Keep original? (Action, RPG, Strategy, Sports, Simulation)

---

## Data Generation Approach

### Methodology
- [ ] Define generation order (which features depend on which)
- [ ] Define probability distributions for base features
- [ ] Define correlation matrices or conditional probability tables
- [ ] Define noise/variance levels for realism

### Implementation
- [ ] Programming language: Python (assumed)
- [ ] Libraries: pandas, numpy, scipy (TBD)
- [ ] Output format: CSV
- [ ] Output filename: TBD

---

## Educational Objectives

### For EDA
- Students should be able to discover:
  - [ ] Correlation patterns between continuous variables
  - [ ] Demographic preferences and patterns
  - [ ] Behavioral consistency patterns
  - [ ] Distribution characteristics

### For ML
- Students should be able to:
  - [ ] Predict `PlayerExpertise` from other features (harder task - feature engineering required)
  - [ ] Predict `SpendingPropensity` from other features (easier task - clearer business patterns)
  - [ ] Perform feature engineering to improve predictions
  - [ ] Discover non-linear relationships
  - [ ] Handle multi-factorial target variables
  - [ ] Deal with class imbalance (especially for SpendingPropensity)
  - [ ] Compare model performance between different target variables
  - [ ] Deal with realistic noise levels

---

## Open Questions
1. Should `PlayerLevel` have game-specific max levels, or be normalized?
2. What's the desired difficulty level for the ML task? (easy/medium/hard to predict)
3. Should there be any missing data, or keep it clean for educational purposes?
4. Any specific statistical distributions required (normal, uniform, power-law, etc.)?
5. Should temporal aspects be considered (e.g., newer players vs veterans)?

---

## Notes & Decisions
_This section will capture key decisions and rationale as we discuss_

- **Decision Log**:
  - **TWO target variables** for ML: `PlayerExpertise` and `SpendingPropensity`
  - `PlayerExpertise`: Harder task, multi-factorial, requires feature engineering
    - 3 classes: Beginner (35%), Intermediate (45%), Expert (20%)
    - Factors: GameDifficulty, efficiency metrics, consistency, skill progression, challenge-seeking
  - `SpendingPropensity`: Easier task, clear business link, good for teaching
    - 3 classes: NonSpender (55%), Occasional (35%), Whale (10%)
    - Factors: EngagementLevel, PlayTimeHours, GameGenre, Age, Achievements, InGamePurchases
  - Both variables provide class imbalance for students to handle
  - SpendingPropensity gives obvious business application for teaching
