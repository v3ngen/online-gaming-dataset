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
1. **PurchaseCount & TotalSpendUSD** - Core metrics (NonSpender: $0, Occasional: $1-100, Whale: $100+)
2. **EngagementLevel** - High engagement → more spending likelihood
3. **PlayTimeHours** - Invested time → invested money (sunk cost fallacy)
4. **DaysPlayed** - Longer-term commitment → more spending
5. **GameGenre** - Some genres monetize better (e.g., RPG/Strategy > Sports)
6. **Age** - Disposable income patterns (older players may spend more)
7. **AchievementsUnlocked** - Completionists spend more to unlock content

---

## Dataset Structure

### Data Granularity
**IMPORTANT**: Each row represents a **player-game combination**, not just a player.
- A single player can play multiple games
- PlayerID will repeat across rows for different games
- This enables students to perform feature engineering (e.g., aggregate by player, count games per player)

**Estimated Row Count**: ~10,000 rows representing player-game combinations

### Feature List

#### Identifiers
1. **PlayerID** - Unique player identifier (will repeat for players playing multiple games)
2. **GameID** - Unique game identifier (allows same player to play different games)

#### Demographics (Player-level attributes)
3. **Age** - Player age (integer, years)
4. **Gender** - Player gender {Male, Female, Other}
5. **Location** - Geographic region {USA, Europe, Asia, Other}

#### Game Context
6. **GameGenre** - Type of game {Action, RPG, Strategy, Sports, Simulation}
7. **GameDifficulty** - Difficulty setting chosen by player {Easy, Medium, Hard}

#### Behavioral Metrics (Player-Game specific)
8. **PlayTimeHours** - Total hours played in this specific game (float)
9. **SessionsPerWeek** - Average gaming sessions per week for this game (integer)
10. **AvgSessionDurationMinutes** - Average session length for this game (integer, minutes)
11. **PlayerLevel** - Level achieved in this game (integer)
12. **AchievementsUnlocked** - Number of achievements unlocked in this game (integer)

#### Engagement Indicators
13. **EngagementLevel** - Overall engagement with this game {Low, Medium, High}
14. **DaysPlayed** - Number of days player has been active in this game (integer)

#### Spending Behavior
15. **PurchaseCount** - Number of in-game purchases made in this game (integer, 0+)
16. **TotalSpend** - Total amount spent in this game (float, USD, 0+)
17. **AvgSessionSpend** - Average spend per session (float, derived or standalone)

#### Target Variables (ML Prediction)
18. **PlayerExpertise** - Expertise level {Beginner, Intermediate, Expert}
19. **SpendingPropensity** - Spending behavior category {NonSpender, Occasional, Whale}

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
- [ ] **DaysPlayed**: Longer-term players spend more (commitment)
- [ ] **GameGenre**: RPG/Strategy players spend more than Sports players
- [ ] **Age**: Older players (disposable income) spend more
- [ ] **AchievementsUnlocked**: Completionists spend more
- [ ] **PurchaseCount & TotalSpendUSD**: Core spending metrics used to derive SpendingPropensity

#### Natural Correlations (for EDA discovery)
- [ ] **PlayTimeHours ↔ PlayerLevel**: Strong positive correlation (more time = higher level)
- [ ] **PlayTimeHours ↔ AchievementsUnlocked**: Moderate positive correlation
- [ ] **DaysPlayed ↔ PlayTimeHours**: Strong positive correlation (longer tenure = more hours)
- [ ] **SessionsPerWeek × AvgSessionDurationMinutes ↔ PlayTimeHours**: Mathematical relationship with variance
- [ ] **Age ↔ GameGenre**: Demographic preferences (younger → Action/Sports, older → Strategy)
- [ ] **Gender ↔ GameGenre**: Demographic preferences (to define)
- [ ] **EngagementLevel**: Function of PlayTime, Sessions, Achievements, DaysPlayed
- [ ] **PurchaseCount ↔ TotalSpendUSD**: Strong positive correlation (more purchases = more spend)
- [ ] **PlayerLevel ↔ GameDifficulty**: Players at higher levels may choose harder difficulties
- [ ] **Multi-game players**: Players with multiple games may have different patterns (diversification)

#### Domain Rules & Constraints
- [ ] **Age range**: 13-65 years (realistic gaming demographic)
- [ ] **PlayerLevel range**: 1-100 (game-specific, may vary by genre)
- [ ] **SessionsPerWeek**: 0-21 (0 = inactive, 21 = 3/day max)
- [ ] **AvgSessionDurationMinutes**: 15-240 minutes (15 min = casual, 4 hrs = hardcore)
- [ ] **PlayTimeHours**: 0.5-2000+ hours (wide range, heavy-tailed distribution)
- [ ] **DaysPlayed**: 1-1095 (up to ~3 years of activity)
- [ ] **AchievementsUnlocked**: 0-100+ (varies by game)
- [ ] **PurchaseCount**: 0-200+ (most 0, some whales very high)
- [ ] **TotalSpendUSD**: $0-$10,000+ (heavy-tailed, most $0)
- [ ] **Location distribution**: USA 35%, Europe 30%, Asia 25%, Other 10%
- [ ] **GameGenre distribution**: Balanced or realistic (RPG/Action popular)
- [ ] **Number of games per player**: Most 1-3, some up to 5-10
- [ ] **Number of unique games**: 50-200 different GameIDs total

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
    - Factors: PurchaseCount, TotalSpendUSD, EngagementLevel, PlayTimeHours, DaysPlayed, GameGenre, Age, Achievements
  - Both variables provide class imbalance for students to handle
  - SpendingPropensity gives obvious business application for teaching
  - **Multi-game structure**: Each row is player-game combination, enabling feature engineering
    - Added GameID alongside PlayerID
    - Added spending features: PurchaseCount, TotalSpendUSD, AvgSessionSpendUSD
    - Added DaysPlayed for tenure/commitment tracking
    - Students can aggregate by player (count games, sum spending, identify genre preferences)
