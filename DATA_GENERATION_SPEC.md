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
1. **PurchaseCount & TotalSpend** - Core metrics (NonSpender: £0, Occasional: £1-80, Whale: £80+)
2. **EngagementLevel** - High engagement → more spending likelihood
3. **PlayTimeHours** - Invested time → invested money (sunk cost fallacy)
4. **DaysPlayed** - Longer-term commitment → more spending
5. **GameGenre** - RPG (esp. MMORPG) and competitive Action spend most; Strategy medium
6. **Specific Games** - RPG_002 (MMORPG) and ACT_001 (Battle Royale) have highest whale rates
7. **Location** - Asia has higher whale concentration (MMORPG culture)
8. **Age** - Disposable income patterns (older players may spend more)
9. **AchievementsUnlocked** - Completionists spend more to unlock content

---

## Dataset Structure

### Data Granularity
**IMPORTANT**: Each row represents a **player-game combination**, not just a player.
- A single player can play multiple games
- PlayerID will repeat across rows for different games
- This enables students to perform feature engineering (e.g., aggregate by player, count games per player)

**Estimated Row Count**: ~10,000 rows representing player-game combinations

### Feature List (20 features total)

**Note**: Features are numbered for reference but may appear in any order in the final dataset.

#### Identifiers
1. **PlayerID** - Unique player identifier (will repeat for players playing multiple games)
2. **GameID** - Unique game identifier (allows same player to play different games)

#### Demographics (Player-level attributes)
3. **Age** - Player age (integer, years)
4. **Gender** - Player gender {Male, Female, Other}
   - Overall distribution: 63% Male, 35% Female, 2% Other
5. **Location** - Geographic region {USA, Europe, Asia}

#### Game Context
6. **GameGenre** - Type of game {RPG, Action, Strategy}
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
15. **PurchaseCount** - Total count of in-game purchases made in this game (integer, 0+)
16. **TotalSpend** - Total amount spent in this game (float, GBP, 0+)
   - All spending normalized to GBP for consistency across regions
17. **AvgPurchasesPerMonth** - Average purchase frequency per month (float, PurchaseCount / (DaysPlayed / 30))
   - Captures spending cadence (e.g., 2 purchases/month vs 0.1 purchases/month)
18. **AvgPurchaseValue** - Average value per purchase (float, TotalSpend / PurchaseCount, £0 if no purchases)
   - Distinguishes spending patterns: high frequency/low value vs low frequency/high value

#### Target Variables (ML Prediction)
19. **PlayerExpertise** - Expertise level {Beginner, Intermediate, Expert}
20. **SpendingPropensity** - Spending behavior category {NonSpender, Occasional, Whale}

### Game Catalog

**Total Games**: 9 games across 3 genres (3 per genre)

#### RPG Genre (3 games)
- **RPG_001**: "Dragon's Quest" - High fantasy RPG
  - Max Level: 100
  - Max Achievements: 120
  - Typical playtime: 200-800 hours
  - Monetization: High (cosmetics, expansions, convenience items)

- **RPG_002**: "Mystic Realms Online" - MMORPG
  - Max Level: 80
  - Max Achievements: 150
  - Typical playtime: 300-2000+ hours
  - Monetization: Very High (subscriptions, cosmetics, pay-to-progress)
  - Popular in Asia

- **RPG_003**: "Dungeon Crawler Deluxe" - Action RPG
  - Max Level: 75
  - Max Achievements: 85
  - Typical playtime: 50-300 hours
  - Monetization: Medium (cosmetics, seasonal content)

**RPG Genre Patterns**:
- High playtime overall
- High achievement focus (completionists)
- Strong monetization (especially RPG_002 MMORPG)
- Older demographic bias (25-45 years)
- High engagement levels
- Players often play multiple RPG games
- **Gender distribution**: 45% Female, 53% Male, 2% Other (most balanced genre - women love RPGs)
- **Location bias**: Popular in Asia (especially MMORPGs)

#### Action Genre (3 games)
- **ACT_001**: "Battle Royale Extreme" - Competitive shooter
  - Max Level: 100
  - Max Achievements: 110
  - Typical playtime: 100-1000 hours
  - Monetization: High (battle passes, skins, weapons)
  - Popular in USA

- **ACT_002**: "Zombie Apocalypse" - Co-op survival
  - Max Level: 50
  - Max Achievements: 75
  - Typical playtime: 30-200 hours
  - Monetization: Low-Medium (DLC packs)

- **ACT_003**: "Street Fighter Ultimate" - Fighting game
  - Max Level: 60
  - Max Achievements: 90
  - Typical playtime: 50-500 hours
  - Monetization: Medium (character DLC, cosmetics)

**Action Genre Patterns**:
- Varied playtime (wide range)
- Younger demographic bias (16-35 years)
- Mixed monetization (ACT_001 high, ACT_002 low)
- Shorter session durations, higher frequency
- Difficulty varies widely
- **Gender distribution**: 25% Female, 73% Male, 2% Other (male-dominated, especially shooters)
- **Location bias**: Popular in USA (especially Battle Royale/shooters)

#### Strategy Genre (3 games)
- **STR_001**: "Empire Builder" - Grand strategy
  - Max Level: 50
  - Max Achievements: 130
  - Typical playtime: 100-800 hours
  - Monetization: Medium (DLC expansions, civilization packs)
  - Popular in Europe

- **STR_002**: "Tower Defense Masters" - Tower defense
  - Max Level: 75
  - Max Achievements: 80
  - Typical playtime: 40-200 hours
  - Monetization: Medium (tower packs, cosmetics)

- **STR_003**: "Chess Legends Online" - Turn-based strategy
  - Max Level: 30 (ranking system)
  - Max Achievements: 50
  - Typical playtime: 50-500 hours
  - Monetization: Low-Medium (premium features, cosmetics)

**Strategy Genre Patterns**:
- Long session durations (thoughtful gameplay)
- Older demographic bias (25-50 years)
- Medium monetization (expansion-focused)
- Lower session frequency, longer duration
- High difficulty tolerance
- **Gender distribution**: 40% Female, 58% Male, 2% Other (relatively balanced - strategy ranks 2nd for female gamers)
- **Location bias**: Popular in Europe (strong strategy gaming culture)

### Cross-Game Patterns
- **Genre-loyal players**: Play multiple games in same genre (e.g., RPG_001 + RPG_002)
- **Diversified players**: Play across genres (e.g., ACT_001 + STR_002)
- **Whale behavior**: More likely in RPG_002 (MMORPG) and ACT_001 (competitive shooter)
- **Multi-game players**: Typically 1-3 games, rarely 4+
- **Location-genre correlations**:
  - USA players slightly favor Action games
  - Europe players slightly favor Strategy games
  - Asia players slightly favor RPG games (especially MMORPGs)

### Relationships to Define

#### PlayerExpertise Dependencies
- [ ] **GameDifficulty**: Experts gravitate toward Hard (strong but not deterministic)
- [ ] **Efficiency (Achievements/PlayTime)**: Experts achieve more per hour
- [ ] **Consistency (SessionsPerWeek)**: Experts play regularly
- [ ] **Skill Progression (PlayerLevel/PlayTimeHours ratio)**: Experts level faster
- [ ] **Challenge-Seeking**: Combination of difficulty + achievements

#### SpendingPropensity Dependencies
- [ ] **PurchaseCount, TotalSpend, AvgPurchasesPerMonth**: Core spending metrics used to derive SpendingPropensity
- [ ] **EngagementLevel**: High engagement → more spending
- [ ] **PlayTimeHours**: More time invested → more spending
- [ ] **DaysPlayed**: Longer-term players spend more (commitment)
- [ ] **GameGenre**: RPG players spend most (especially MMORPG), Strategy medium, Action varied (high for competitive games)
- [ ] **Specific Games**: RPG_002 (MMORPG) and ACT_001 (Battle Royale) highest whale rates
- [ ] **Location**: Asia has higher whale concentration (MMORPG culture)
- [ ] **Age**: Older players (disposable income) spend more
- [ ] **AchievementsUnlocked**: Completionists spend more

#### Natural Correlations (for EDA discovery)
- [ ] **PlayTimeHours ↔ PlayerLevel**: Strong positive correlation (more time = higher level)
- [ ] **PlayTimeHours ↔ AchievementsUnlocked**: Moderate positive correlation
- [ ] **DaysPlayed ↔ PlayTimeHours**: Strong positive correlation (longer tenure = more hours)
- [ ] **SessionsPerWeek × AvgSessionDurationMinutes ↔ PlayTimeHours**: Mathematical relationship with variance
- [ ] **Age ↔ GameGenre**: Demographic preferences (younger 16-35 → Action, middle 25-45 → RPG, older 25-50 → Strategy)
- [ ] **Gender ↔ GameGenre**: Strong demographic preferences based on research
  - Action: 25% Female, 73% Male, 2% Other (male-dominated)
  - RPG: 45% Female, 53% Male, 2% Other (most balanced - women love RPGs)
  - Strategy: 40% Female, 58% Male, 2% Other (relatively balanced)
- [ ] **Location ↔ GameGenre**: Regional gaming culture differences
  - USA: 45% Action, 35% RPG, 20% Strategy (shooter culture)
  - Europe: 35% Action, 35% RPG, 30% Strategy (strategy gaming culture)
  - Asia: 30% Action, 50% RPG, 20% Strategy (MMORPG culture)
- [ ] **Location ↔ SpendingPropensity**: Asia higher whale concentration
- [ ] **EngagementLevel**: Function of PlayTime, Sessions, Achievements, DaysPlayed
- [ ] **PurchaseCount ↔ TotalSpend**: Strong positive correlation (more purchases = more spend)
- [ ] **PlayerLevel ↔ GameDifficulty**: Players at higher levels may choose harder difficulties
- [ ] **Multi-game players**: Players with multiple games may have different patterns (diversification)

#### Domain Rules & Constraints
- [ ] **Age range**: 13-65 years (realistic gaming demographic)
  - Action: 16-35 (younger bias)
  - RPG: 20-50 (middle-age bias)
  - Strategy: 22-60 (older bias)
- [ ] **PlayerLevel range**: Game-specific (see Game Catalog above)
  - RPG: 75-100 max
  - Action: 40-100 max
  - Strategy: 30-75 max
- [ ] **SessionsPerWeek**: 0-21 (0 = inactive, 21 = 3/day max)
  - Action: Higher frequency (5-15/week typical)
  - RPG: Medium frequency (3-10/week typical)
  - Strategy: Lower frequency (2-7/week typical)
- [ ] **AvgSessionDurationMinutes**: 15-240 minutes
  - Action: Shorter (30-90 min typical)
  - RPG: Medium-Long (60-180 min typical)
  - Strategy: Longer (60-240 min typical)
- [ ] **PlayTimeHours**: 0.5-2000+ hours (game-specific, see Game Catalog)
- [ ] **DaysPlayed**: 7-1095 (1 week to ~3 years of activity)
- [ ] **AchievementsUnlocked**: Game-specific (see Game Catalog)
- [ ] **PurchaseCount**: 0-200+ (total count, most 0, some whales very high)
- [ ] **TotalSpend**: £0-£8,000+ (heavy-tailed, most £0, all amounts in GBP)
  - RPG_002 (MMORPG): Highest spending
  - ACT_001 (Battle Royale): High spending
  - ACT_002, STR_003: Lower spending
  - Asia players higher whale concentration (especially in RPG_002)
- [ ] **AvgPurchasesPerMonth**: 0-10+ (most 0, occasional 0.5-2, whales 2-10+)
- [ ] **Gender distribution**:
  - Overall: 63% Male, 35% Female, 2% Other
  - By genre: Action (73% M, 25% F), RPG (53% M, 45% F), Strategy (58% M, 40% F)
- [ ] **Location distribution**: USA 40%, Europe 35%, Asia 25%
  - Location-genre biases:
    - USA: 45% Action, 35% RPG, 20% Strategy
    - Europe: 35% Action, 35% RPG, 30% Strategy
    - Asia: 30% Action, 50% RPG, 20% Strategy
- [ ] **GameGenre distribution**:
  - RPG: 40% of rows (popular + high retention)
  - Action: 40% of rows (popular + varied)
  - Strategy: 20% of rows (niche but dedicated)
- [ ] **Number of games per player**:
  - 1 game: 60%
  - 2 games: 25%
  - 3 games: 12%
  - 4+ games: 3%
- [ ] **GameID distribution**: 9 total games (3 RPG, 3 Action, 3 Strategy)
  - Popular games (RPG_002 MMORPG, ACT_001 Battle Royale): More players
  - Niche games (ACT_002, STR_003): Fewer players
  - Location-game correlations:
    - RPG_002 (MMORPG) popular in Asia
    - ACT_001 (Battle Royale) popular in USA
    - STR_001 (Grand Strategy) popular in Europe

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
    - Factors: PurchaseCount, TotalSpend, EngagementLevel, PlayTimeHours, DaysPlayed, GameGenre (RPG high), Specific Games (RPG_002, ACT_001), Location (Asia higher whales), Age, Achievements
  - Both variables provide class imbalance for students to handle
  - SpendingPropensity gives obvious business application for teaching
  - **Multi-game structure**: Each row is player-game combination, enabling feature engineering
    - Added GameID alongside PlayerID
    - **Simplified structure**: 3 genres, 3 games each (9 total)
      - RPG: 3 games (Dragon's Quest, Mystic Realms MMORPG, Dungeon Crawler)
      - Action: 3 games (Battle Royale, Zombie Apocalypse, Street Fighter)
      - Strategy: 3 games (Empire Builder, Tower Defense, Chess Legends)
      - Each game has distinct characteristics (max level, achievements, monetization)
      - Enables genre-loyal vs diversified player patterns
      - Realistic gender distributions based on 2024 research (RPG most balanced, Action male-dominated)
    - **3 locations only**: USA (40%), Europe (35%), Asia (25%)
      - Location-genre biases: USA→Action, Europe→Strategy, Asia→RPG
      - Asia has higher whale concentration (MMORPG culture)
    - Added spending features (all in GBP):
      - PurchaseCount: Total count of purchases
      - TotalSpend: Total monetary value
      - AvgPurchasesPerMonth: Purchase frequency (how often)
      - AvgPurchaseValue: Transaction size (large vs small purchases)
      - All spending normalized to GBP for consistency across global regions
    - Added DaysPlayed for tenure/commitment tracking
    - Students can aggregate by player (count games, sum spending, identify genre preferences)
