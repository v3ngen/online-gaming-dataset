"""
Online Gaming Behavior Dataset Generator

Generates a coherent synthetic dataset for educational purposes (EDA and ML).
Based on DATA_GENERATION_ALGORITHM.md specification.

Usage:
    python generate_dataset.py [--rows 10000] [--seed 42] [--output generated_gaming_dataset.csv]
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import argparse
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration parameters for dataset generation"""

    # Target dataset size
    num_rows: int = 10000
    num_players_estimate: int = 6500  # Will create multi-game combinations to reach target rows

    # Random seed for reproducibility
    random_seed: int = 42

    # Demographics distributions
    gender_dist: Dict[str, float] = None
    location_dist: Dict[str, float] = None

    # Age distribution (Beta distribution parameters)
    age_min: int = 13
    age_max: int = 65
    age_beta_a: float = 2.0
    age_beta_b: float = 5.0

    # Multi-game distribution
    games_per_player_dist: Dict[int, float] = None

    # Target variable distributions
    expertise_dist: Dict[str, float] = None
    spending_dist: Dict[str, float] = None
    engagement_dist: Dict[str, float] = None

    def __post_init__(self):
        """Initialize default dictionaries"""
        if self.gender_dist is None:
            self.gender_dist = {'Male': 0.63, 'Female': 0.35, 'Other': 0.02}

        if self.location_dist is None:
            self.location_dist = {'USA': 0.40, 'Europe': 0.35, 'Asia': 0.25}

        if self.games_per_player_dist is None:
            self.games_per_player_dist = {1: 0.60, 2: 0.25, 3: 0.12, 4: 0.03}

        if self.expertise_dist is None:
            self.expertise_dist = {'Beginner': 0.35, 'Intermediate': 0.45, 'Expert': 0.20}

        if self.spending_dist is None:
            self.spending_dist = {'NonSpender': 0.55, 'Occasional': 0.35, 'Whale': 0.10}

        if self.engagement_dist is None:
            self.engagement_dist = {'Low': 0.30, 'Medium': 0.45, 'High': 0.25}


# Game catalog with specifications
GAMES = {
    'RPG_001': {
        'name': "Dragon's Quest",
        'genre': 'RPG',
        'max_level': 100,
        'max_achievements': 120,
        'typical_hours_min': 200,
        'typical_hours_max': 800,
        'monetization': 'high',
    },
    'RPG_002': {
        'name': "Mystic Realms Online",
        'genre': 'RPG',
        'max_level': 80,
        'max_achievements': 150,
        'typical_hours_min': 300,
        'typical_hours_max': 2000,
        'monetization': 'very_high',
        'popular_in': 'Asia',  # MMORPG
    },
    'RPG_003': {
        'name': "Dungeon Crawler Deluxe",
        'genre': 'RPG',
        'max_level': 75,
        'max_achievements': 85,
        'typical_hours_min': 50,
        'typical_hours_max': 300,
        'monetization': 'medium',
    },
    'ACT_001': {
        'name': "Battle Royale Extreme",
        'genre': 'Action',
        'max_level': 100,
        'max_achievements': 110,
        'typical_hours_min': 100,
        'typical_hours_max': 1000,
        'monetization': 'high',
        'popular_in': 'USA',
    },
    'ACT_002': {
        'name': "Zombie Apocalypse",
        'genre': 'Action',
        'max_level': 50,
        'max_achievements': 75,
        'typical_hours_min': 30,
        'typical_hours_max': 200,
        'monetization': 'low_medium',
    },
    'ACT_003': {
        'name': "Street Fighter Ultimate",
        'genre': 'Action',
        'max_level': 60,
        'max_achievements': 90,
        'typical_hours_min': 50,
        'typical_hours_max': 500,
        'monetization': 'medium',
    },
    'STR_001': {
        'name': "Empire Builder",
        'genre': 'Strategy',
        'max_level': 50,
        'max_achievements': 130,
        'typical_hours_min': 100,
        'typical_hours_max': 800,
        'monetization': 'medium',
        'popular_in': 'Europe',
    },
    'STR_002': {
        'name': "Tower Defense Masters",
        'genre': 'Strategy',
        'max_level': 75,
        'max_achievements': 80,
        'typical_hours_min': 40,
        'typical_hours_max': 200,
        'monetization': 'medium',
    },
    'STR_003': {
        'name': "Chess Legends Online",
        'genre': 'Strategy',
        'max_level': 30,
        'max_achievements': 50,
        'typical_hours_min': 50,
        'typical_hours_max': 500,
        'monetization': 'low_medium',
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_gender_probabilities(genre: str) -> Dict[str, float]:
    """Get gender distribution probabilities based on game genre

    Based on 2024 gaming research:
    - RPG: Most balanced (45% Female)
    - Action: Male-dominated (25% Female)
    - Strategy: Relatively balanced (40% Female)
    """
    if genre == 'RPG':
        return {'Male': 0.53, 'Female': 0.45, 'Other': 0.02}
    elif genre == 'Action':
        return {'Male': 0.73, 'Female': 0.25, 'Other': 0.02}
    elif genre == 'Strategy':
        return {'Male': 0.58, 'Female': 0.40, 'Other': 0.02}
    else:
        # Fallback to overall distribution
        return {'Male': 0.63, 'Female': 0.35, 'Other': 0.02}


def get_genre_probabilities(location: str) -> Dict[str, float]:
    """Get genre selection probabilities based on location"""
    if location == 'USA':
        return {'RPG': 0.35, 'Action': 0.45, 'Strategy': 0.20}
    elif location == 'Europe':
        return {'RPG': 0.35, 'Action': 0.35, 'Strategy': 0.30}
    elif location == 'Asia':
        return {'RPG': 0.50, 'Action': 0.30, 'Strategy': 0.20}
    else:
        return {'RPG': 0.40, 'Action': 0.40, 'Strategy': 0.20}


def select_game(genre: str, location: str) -> str:
    """Select a specific game within a genre, with location bias"""
    genre_games = [gid for gid, g in GAMES.items() if g['genre'] == genre]

    # Create weights (popular games get higher weight)
    weights = []
    for gid in genre_games:
        game = GAMES[gid]
        weight = 1.0

        # Boost if game is popular in this location
        if game.get('popular_in') == location:
            weight = 2.0

        # Boost based on monetization (more popular games tend to have higher monetization)
        if game['monetization'] == 'very_high':
            weight *= 1.5
        elif game['monetization'] == 'high':
            weight *= 1.3

        weights.append(weight)

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    return np.random.choice(genre_games, p=weights)


def get_behavioral_params(genre: str, param_type: str) -> Dict:
    """Get genre-specific behavioral parameters"""
    if param_type == 'sessions_per_week':
        params = {
            'Action': {'mean': 8, 'std': 4, 'min': 3, 'max': 18},
            'RPG': {'mean': 6, 'std': 3, 'min': 2, 'max': 15},
            'Strategy': {'mean': 5, 'std': 2.5, 'min': 2, 'max': 12},
        }
    elif param_type == 'session_duration':
        params = {
            'Action': {'mean': 60, 'std': 25, 'min': 30, 'max': 120},
            'RPG': {'mean': 120, 'std': 40, 'min': 60, 'max': 240},
            'Strategy': {'mean': 150, 'std': 50, 'min': 60, 'max': 240},
        }
    else:
        raise ValueError(f"Unknown param_type: {param_type}")

    return params.get(genre, params['RPG'])  # Default to RPG if genre not found


# ============================================================================
# MAIN GENERATION CLASS
# ============================================================================

class DatasetGenerator:
    """Generates coherent gaming behavior dataset"""

    def __init__(self, config: Config):
        self.config = config
        np.random.seed(config.random_seed)
        self.players_df = None
        self.dataset_df = None

    def generate(self) -> pd.DataFrame:
        """Generate the complete dataset"""
        print("Starting dataset generation...")

        # Level 0: Player demographics
        print("Step 1-4: Generating player demographics...")
        self.generate_player_demographics()

        # Level 1: Game assignments
        print("Step 5-6: Assigning games to players...")
        self.assign_games_to_players()

        # Level 2-3: Behavioral metrics
        print("Step 7-10: Generating behavioral metrics...")
        self.generate_behavioral_metrics()

        # Level 4: Game progression
        print("Step 11-13: Generating game progression...")
        self.generate_game_progression()

        # Level 5: Achievements and engagement
        print("Step 14-15: Generating achievements and engagement...")
        self.generate_achievements_and_engagement()

        # Level 6-7: Spending
        print("Step 16-20: Generating spending behavior...")
        self.generate_spending()

        # Level 8: Target variables
        print("Step 21-22: Finalizing target variables...")
        self.finalize_target_variables()

        print(f"Dataset generation complete! {len(self.dataset_df)} rows created.")
        return self.dataset_df

    def generate_player_demographics(self):
        """Generate base player demographics (Steps 1-4)"""
        num_players = self.config.num_players_estimate

        # Step 1: PlayerID
        player_ids = np.arange(1, num_players + 1)

        # Step 2: Location (needed before games to determine genre preferences)
        locations = np.random.choice(
            list(self.config.location_dist.keys()),
            size=num_players,
            p=list(self.config.location_dist.values())
        )

        # Step 3: Age (Beta distribution)
        age_beta = np.random.beta(
            self.config.age_beta_a,
            self.config.age_beta_b,
            size=num_players
        )
        ages = (age_beta * (self.config.age_max - self.config.age_min) +
                self.config.age_min).astype(int)

        # NOTE: Gender is NOT assigned here - it will be assigned based on
        # the player's primary game genre to reflect realistic gender distributions
        self.players_df = pd.DataFrame({
            'PlayerID': player_ids,
            'Age': ages,
            'Location': locations,
        })

    def assign_games_to_players(self):
        """Assign games to players based on demographics (Steps 5-6)

        Gender is assigned per player based on their primary (first) game's genre.
        This ensures gender is consistent for each player across all their games,
        while achieving the desired genre-specific gender distributions.
        """
        player_game_rows = []
        player_genders = {}  # Cache gender per player

        for _, player in self.players_df.iterrows():
            # Step 5: Determine number of games for this player
            num_games = np.random.choice(
                list(self.config.games_per_player_dist.keys()),
                p=list(self.config.games_per_player_dist.values())
            )

            # Step 6: Select games based on location
            genre_probs = get_genre_probabilities(player['Location'])

            selected_games = []
            selected_game_ids = set()

            # Select all games for this player first
            for _ in range(num_games):
                # Select genre
                genre = np.random.choice(
                    list(genre_probs.keys()),
                    p=list(genre_probs.values())
                )

                # Select specific game
                game_id = select_game(genre, player['Location'])

                # Avoid duplicate games for same player
                attempts = 0
                while game_id in selected_game_ids and attempts < 10:
                    genre = np.random.choice(
                        list(genre_probs.keys()),
                        p=list(genre_probs.values())
                    )
                    game_id = select_game(genre, player['Location'])
                    attempts += 1

                selected_game_ids.add(game_id)
                selected_games.append(game_id)

            # Assign gender based on FIRST game's genre (primary genre)
            # This ensures gender consistency per player while achieving genre distributions
            primary_game_genre = GAMES[selected_games[0]]['genre']
            gender_probs = get_gender_probabilities(primary_game_genre)
            player_gender = np.random.choice(
                list(gender_probs.keys()),
                p=list(gender_probs.values())
            )
            player_genders[player['PlayerID']] = player_gender

            # Create player-game rows with consistent gender
            for game_id in selected_games:
                player_game_rows.append({
                    'PlayerID': player['PlayerID'],
                    'Age': player['Age'],
                    'Gender': player_gender,  # Consistent across all games
                    'Location': player['Location'],
                    'GameID': game_id,
                    'GameName': GAMES[game_id]['name'],
                    'GameGenre': GAMES[game_id]['genre'],
                })

        self.dataset_df = pd.DataFrame(player_game_rows)

        # Trim to target number of rows if exceeded
        if len(self.dataset_df) > self.config.num_rows:
            self.dataset_df = self.dataset_df.sample(n=self.config.num_rows, random_state=self.config.random_seed)
            self.dataset_df = self.dataset_df.reset_index(drop=True)

    def generate_behavioral_metrics(self):
        """Generate behavioral metrics (Steps 7-10)"""
        n = len(self.dataset_df)

        # Step 7: DaysPlayed (log-normal distribution)
        days_mu = 4.5
        days_sigma = 1.0
        days = np.random.lognormal(days_mu, days_sigma, size=n)
        self.dataset_df['DaysPlayed'] = np.clip(days, 7, 1095).astype(int)

        # Step 8: SessionsPerWeek (genre-specific)
        sessions = np.zeros(n)
        for i, row in self.dataset_df.iterrows():
            params = get_behavioral_params(row['GameGenre'], 'sessions_per_week')
            # Use Gamma distribution
            shape = (params['mean'] / params['std']) ** 2
            scale = params['std'] ** 2 / params['mean']
            sessions[i] = np.random.gamma(shape, scale)

        self.dataset_df['SessionsPerWeek'] = np.clip(
            sessions,
            self.dataset_df['GameGenre'].map(lambda g: get_behavioral_params(g, 'sessions_per_week')['min']),
            self.dataset_df['GameGenre'].map(lambda g: get_behavioral_params(g, 'sessions_per_week')['max'])
        ).round().astype(int)

        # Step 9: AvgSessionDurationMinutes (genre-specific)
        durations = np.zeros(n)
        for i, row in self.dataset_df.iterrows():
            params = get_behavioral_params(row['GameGenre'], 'session_duration')
            shape = (params['mean'] / params['std']) ** 2
            scale = params['std'] ** 2 / params['mean']
            durations[i] = np.random.gamma(shape, scale)

        self.dataset_df['AvgSessionDurationMinutes'] = np.clip(
            durations,
            self.dataset_df['GameGenre'].map(lambda g: get_behavioral_params(g, 'session_duration')['min']),
            self.dataset_df['GameGenre'].map(lambda g: get_behavioral_params(g, 'session_duration')['max'])
        ).round().astype(int)

        # Step 10: PlayTimeHours (derived from sessions, duration, days)
        estimated_hours = (
            self.dataset_df['SessionsPerWeek'] *
            self.dataset_df['AvgSessionDurationMinutes'] *
            self.dataset_df['DaysPlayed']
        ) / 420  # (7 days * 60 minutes)

        # Add variance (±20%)
        noise = np.random.uniform(0.8, 1.2, size=n)
        play_hours = estimated_hours * noise

        # Clamp to game-specific ranges
        for i, row in self.dataset_df.iterrows():
            game = GAMES[row['GameID']]
            play_hours[i] = np.clip(
                play_hours[i],
                game['typical_hours_min'] * 0.5,  # Allow some below minimum
                game['typical_hours_max'] * 1.2   # Allow some above maximum
            )

        self.dataset_df['PlayTimeHours'] = play_hours.round(2)

    def generate_game_progression(self):
        """Generate game progression (Steps 11-13)"""
        n = len(self.dataset_df)

        # Step 11: Generate expertise_seed (used later for PlayerExpertise)
        # Base random + age boost
        expertise_seed = np.random.uniform(0, 100, size=n)

        # Age influence (boost for 20-35 range)
        age_boost = np.where(
            (self.dataset_df['Age'] >= 20) & (self.dataset_df['Age'] <= 35),
            10,
            0
        )
        expertise_seed += age_boost
        expertise_seed = np.clip(expertise_seed, 0, 100)

        self.dataset_df['expertise_seed'] = expertise_seed

        # Step 12: PlayerLevel (based on PlayTimeHours and expertise_seed)
        player_levels = np.zeros(n)

        for i, row in self.dataset_df.iterrows():
            game = GAMES[row['GameID']]
            max_level = game['max_level']
            typical_hours_mean = (game['typical_hours_min'] + game['typical_hours_max']) / 2

            # Progression rate based on expertise
            if row['expertise_seed'] >= 75:
                progression_multiplier = 0.7  # Experts level faster
            elif row['expertise_seed'] >= 40:
                progression_multiplier = 1.0  # Intermediate
            else:
                progression_multiplier = 1.5  # Beginners level slower

            expected_max_hours = typical_hours_mean * progression_multiplier
            level_progress = row['PlayTimeHours'] / expected_max_hours

            # Add noise
            level_noise = np.random.uniform(-5, 5)
            level = min(max_level, level_progress * max_level + level_noise)
            player_levels[i] = max(1, level)

        self.dataset_df['PlayerLevel'] = player_levels.round().astype(int)

        # Step 13: GameDifficulty (based on expertise_seed and PlayerLevel)
        difficulties = []

        for _, row in self.dataset_df.iterrows():
            if row['expertise_seed'] >= 70:
                probs = [0.20, 0.40, 0.40]  # Experts prefer Hard
            elif row['expertise_seed'] >= 40:
                probs = [0.35, 0.45, 0.20]  # Intermediate
            else:
                probs = [0.55, 0.35, 0.10]  # Beginners prefer Easy

            difficulty = np.random.choice(['Easy', 'Medium', 'Hard'], p=probs)
            difficulties.append(difficulty)

        self.dataset_df['GameDifficulty'] = difficulties

    def generate_achievements_and_engagement(self):
        """Generate achievements and engagement (Steps 14-15)"""
        n = len(self.dataset_df)

        # Step 14: AchievementsUnlocked
        achievements = np.zeros(n)

        for i, row in self.dataset_df.iterrows():
            game = GAMES[row['GameID']]
            max_achievements = game['max_achievements']
            max_level = game['max_level']

            # Determine completionist factor
            rand = np.random.rand()
            if rand < 0.30:  # Completionist
                target_pct = np.random.uniform(0.7, 1.0)
            elif rand < 0.80:  # Moderate (50% of 70% remaining)
                target_pct = np.random.uniform(0.3, 0.7)
            else:  # Casual
                target_pct = np.random.uniform(0.05, 0.3)

            # Progress factor (can't unlock all achievements if low level)
            progress_factor = min(1.0, row['PlayerLevel'] / max_level)

            achievements[i] = max_achievements * target_pct * progress_factor

        self.dataset_df['AchievementsUnlocked'] = achievements.round().astype(int)

        # Step 15: EngagementLevel (score-based)
        engagement_scores = np.zeros(n)

        for i, row in self.dataset_df.iterrows():
            score = 0

            # PlayTime score (0-40 points)
            if row['PlayTimeHours'] < 50:
                score += np.random.uniform(0, 15)
            elif row['PlayTimeHours'] < 200:
                score += np.random.uniform(15, 30)
            else:
                score += np.random.uniform(30, 40)

            # Session frequency score (0-30 points)
            if row['SessionsPerWeek'] < 3:
                score += np.random.uniform(0, 10)
            elif row['SessionsPerWeek'] < 8:
                score += np.random.uniform(10, 20)
            else:
                score += np.random.uniform(20, 30)

            # Tenure score (0-20 points)
            if row['DaysPlayed'] < 60:
                score += np.random.uniform(0, 7)
            elif row['DaysPlayed'] < 365:
                score += np.random.uniform(7, 15)
            else:
                score += np.random.uniform(15, 20)

            # Achievement score (0-10 points)
            game = GAMES[row['GameID']]
            achievement_pct = row['AchievementsUnlocked'] / game['max_achievements']
            score += achievement_pct * 10

            engagement_scores[i] = score

        # Convert scores to categories (adjusted thresholds)
        self.dataset_df['engagement_score'] = engagement_scores
        self.dataset_df['EngagementLevel'] = pd.cut(
            engagement_scores,
            bins=[0, 45, 75, 100],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )

    def generate_spending(self):
        """Generate spending behavior (Steps 16-20)"""
        n = len(self.dataset_df)

        # Step 16: Generate spending_seed
        spending_seeds = np.zeros(n)

        for i, row in self.dataset_df.iterrows():
            seed = 0

            # EngagementLevel (40% weight)
            if row['EngagementLevel'] == 'High':
                seed += 30
            elif row['EngagementLevel'] == 'Medium':
                seed += 15

            # Game/Genre monetization (25% weight)
            game = GAMES[row['GameID']]
            if game['monetization'] == 'very_high':
                seed += 25
            elif game['monetization'] == 'high':
                seed += 20
            elif game['monetization'] == 'medium':
                seed += 10
            else:
                seed += 5

            # Location (15% weight)
            if row['Location'] == 'Asia':
                seed += 15
            elif row['Location'] == 'USA':
                seed += 10
            else:
                seed += 5

            # Age (10% weight)
            if 25 <= row['Age'] <= 45:
                seed += 10
            elif 18 <= row['Age'] <= 24:
                seed += 5
            elif row['Age'] >= 45:
                seed += 8

            # PlayTimeHours (10% weight)
            if row['PlayTimeHours'] > 500:
                seed += 10
            elif row['PlayTimeHours'] > 200:
                seed += 6
            else:
                seed += 2

            # Random variance
            seed += np.random.uniform(-15, 15)

            spending_seeds[i] = np.clip(seed, 0, 100)

        self.dataset_df['spending_seed'] = spending_seeds

        # Step 17: PurchaseCount
        purchase_counts = np.zeros(n)

        for i, seed in enumerate(spending_seeds):
            if seed < 45:  # NonSpender (increased threshold from 25)
                purchase_counts[i] = 0
            elif seed < 75:  # Occasional (increased threshold from 65)
                mean_purchases = 1 + (seed - 45) / 3
                purchase_counts[i] = max(1, np.random.poisson(mean_purchases))
            else:  # Whale
                mean_purchases = 10 + (seed - 75) * 0.8
                purchase_counts[i] = max(10, np.random.poisson(mean_purchases))

        self.dataset_df['PurchaseCount'] = purchase_counts.astype(int)

        # Step 18: TotalSpend
        total_spends = np.zeros(n)

        for i, row in self.dataset_df.iterrows():
            if row['PurchaseCount'] == 0:
                total_spends[i] = 0.0
            elif row['spending_seed'] < 75:  # Occasional (increased threshold)
                avg_value = np.random.uniform(2, 25)
                total = row['PurchaseCount'] * avg_value * np.random.uniform(0.8, 1.2)
                total_spends[i] = np.clip(total, 1, 80)
            else:  # Whale
                game = GAMES[row['GameID']]

                # Base on game and location
                if row['Location'] == 'Asia' and row['GameID'] == 'RPG_002':
                    avg_value = np.random.uniform(30, 150)
                elif row['GameID'] == 'ACT_001':
                    avg_value = np.random.uniform(20, 80)
                elif game['monetization'] in ['high', 'very_high']:
                    avg_value = np.random.uniform(15, 60)
                else:
                    avg_value = np.random.uniform(10, 40)

                total = row['PurchaseCount'] * avg_value * np.random.uniform(0.8, 1.2)
                total_spends[i] = max(80, total)

        self.dataset_df['TotalSpend'] = total_spends.round(2)

        # Step 19: AvgPurchasesPerMonth
        months_played = self.dataset_df['DaysPlayed'] / 30
        self.dataset_df['AvgPurchasesPerMonth'] = (
            self.dataset_df['PurchaseCount'] / months_played
        ).round(2)

        # Step 20: AvgPurchaseValue
        avg_purchase_values = np.zeros(n)
        for i, row in self.dataset_df.iterrows():
            if row['PurchaseCount'] > 0:
                avg_purchase_values[i] = row['TotalSpend'] / row['PurchaseCount']

        self.dataset_df['AvgPurchaseValue'] = avg_purchase_values.round(2)

    def finalize_target_variables(self):
        """Finalize target variables (Steps 21-22)"""
        n = len(self.dataset_df)

        # Step 21: PlayerExpertise
        expertise_scores = np.zeros(n)

        for i, row in self.dataset_df.iterrows():
            score = 0
            game = GAMES[row['GameID']]

            # Base score from expertise_seed (30 points)
            score += row['expertise_seed'] * 0.3

            # GameDifficulty (20 points)
            if row['GameDifficulty'] == 'Hard':
                score += 20
            elif row['GameDifficulty'] == 'Medium':
                score += 10
            else:
                score += 0

            # Achievement completion (20 points)
            achievement_pct = row['AchievementsUnlocked'] / game['max_achievements']
            score += achievement_pct * 20

            # Level progression (15 points)
            level_pct = row['PlayerLevel'] / game['max_level']
            score += level_pct * 15

            # Consistency - SessionsPerWeek (15 points)
            if row['SessionsPerWeek'] >= 8:
                score += 15
            elif row['SessionsPerWeek'] >= 3:
                score += 8
            else:
                score += 3

            expertise_scores[i] = np.clip(score, 0, 100)

        # Convert to categories (adjusted thresholds to match target distribution)
        self.dataset_df['expertise_score'] = expertise_scores
        self.dataset_df['PlayerExpertise'] = pd.cut(
            expertise_scores,
            bins=[0, 50, 70, 100],  # Adjusted for better distribution
            labels=['Beginner', 'Intermediate', 'Expert'],
            include_lowest=True
        )

        # Step 22: SpendingPropensity (based on TotalSpend)
        def categorize_spending(total_spend):
            if total_spend == 0:
                return 'NonSpender'
            elif total_spend <= 80:
                return 'Occasional'
            else:
                return 'Whale'

        self.dataset_df['SpendingPropensity'] = self.dataset_df['TotalSpend'].apply(categorize_spending)

        # Drop intermediate columns
        self.dataset_df = self.dataset_df.drop(columns=['expertise_seed', 'spending_seed',
                                                         'engagement_score', 'expertise_score'])

    def get_dataset(self) -> pd.DataFrame:
        """Return the generated dataset with columns in correct order, sorted by PlayerID"""
        column_order = [
            # Player-level attributes (fixed across all games)
            'PlayerID', 'Age', 'Gender', 'Location',
            # Game-specific attributes
            'GameID', 'GameName', 'GameGenre', 'GameDifficulty',
            # Behavioral metrics
            'PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes',
            'PlayerLevel', 'AchievementsUnlocked', 'EngagementLevel', 'DaysPlayed',
            # Spending metrics
            'PurchaseCount', 'TotalSpend', 'AvgPurchasesPerMonth', 'AvgPurchaseValue',
            # Target variables
            'SpendingPropensity', 'PlayerExpertise'
        ]

        # Sort by PlayerID for easy visual verification of player consistency
        # Multi-game players will have consecutive rows
        return self.dataset_df[column_order].sort_values('PlayerID').reset_index(drop=True)


# ============================================================================
# DATA QUALITY ISSUES (FOR TEACHING PURPOSES)
# ============================================================================

def introduce_data_quality_issues(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """
    Introduce realistic data quality issues for teaching data cleaning:
    1. Duplicate rows (~0.5%)
    2. Age anomalies - typo-style errors in user-entered field (~1%)
    3. Missing values - both feature-level and row-level

    Immune fields (no missing values): PlayerID, GameID, GameName,
                                       PlayerExpertise, SpendingPropensity
    """
    np.random.seed(random_seed)
    df_dirty = df.copy()

    print("\n=== Introducing Data Quality Issues (for teaching) ===")

    # 1. DUPLICATES (~0.37%, ~37 rows for 10k dataset)
    # Using a non-round percentage to make it less "clean"
    n_duplicates = int(len(df) * 0.0037)
    duplicate_indices = np.random.choice(df.index, size=n_duplicates, replace=False)
    duplicates = df.iloc[duplicate_indices].copy()
    df_dirty = pd.concat([df_dirty, duplicates], ignore_index=True)
    print(f"✓ Added {n_duplicates} duplicate rows")

    # 2. AGE ANOMALIES - Typo-style errors (~0.73%)
    # These simulate double-typed digits in user registration forms
    # Using non-round percentage and capping max age at 199
    n_age_anomalies = int(len(df_dirty) * 0.0073)
    age_anomaly_indices = np.random.choice(df_dirty.index, size=n_age_anomalies, replace=False)

    typo_patterns = [
        (16, 166), (25, 255), (34, 344), (28, 288),
        (19, 199), (22, 222), (33, 333), (45, 455),
        (18, 188), (27, 277), (29, 299), (31, 311),
        (24, 244), (26, 266), (32, 322), (35, 355),
        (38, 388), (41, 411), (44, 444), (48, 488)
    ]

    for idx in age_anomaly_indices:
        current_age = df_dirty.loc[idx, 'Age']
        # Find a matching pattern or create double-digit typo
        matching_patterns = [p for p in typo_patterns if abs(p[0] - current_age) <= 5 and p[1] <= 199]
        if matching_patterns:
            _, typo_value = matching_patterns[0]
        else:
            # Create a double-digit typo from last digit, capped at 199
            last_digit = current_age % 10
            typo_value = min(current_age * 10 + last_digit, 199)

        # Final safety check: don't exceed 199
        if typo_value > 199:
            # Use first two digits doubled instead
            first_digit = int(str(current_age)[0])
            typo_value = int(str(first_digit) + str(first_digit) + str(current_age)[-1])
            if typo_value > 199:
                typo_value = 199

        df_dirty.loc[idx, 'Age'] = typo_value

    print(f"✓ Added {n_age_anomalies} age anomalies (typo-style double-digit errors, max 199)")

    # 3. MISSING VALUES - Feature-level
    # AvgSessionDurationMinutes: 8% missing (session tracking failures)
    n_missing_session = int(len(df_dirty) * 0.08)
    session_missing_indices = np.random.choice(df_dirty.index, size=n_missing_session, replace=False)
    df_dirty.loc[session_missing_indices, 'AvgSessionDurationMinutes'] = np.nan
    print(f"✓ Added {n_missing_session} missing values to AvgSessionDurationMinutes (8%)")

    # AchievementsUnlocked: 6% missing (data sync issues)
    n_missing_achieve = int(len(df_dirty) * 0.06)
    achieve_missing_indices = np.random.choice(df_dirty.index, size=n_missing_achieve, replace=False)
    df_dirty.loc[achieve_missing_indices, 'AchievementsUnlocked'] = np.nan
    print(f"✓ Added {n_missing_achieve} missing values to AchievementsUnlocked (6%)")

    # 4. MISSING VALUES - Row-level (incomplete records)
    # 5-10 rows with multiple missing values each (4-6 fields per row)
    n_incomplete_rows = np.random.randint(5, 11)
    incomplete_row_indices = np.random.choice(df_dirty.index, size=n_incomplete_rows, replace=False)

    # Fields that can have missing values (exclude immune fields)
    nullable_fields = ['Age', 'Gender', 'Location', 'GameGenre', 'GameDifficulty',
                      'PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes',
                      'PlayerLevel', 'AchievementsUnlocked', 'EngagementLevel',
                      'DaysPlayed', 'PurchaseCount', 'TotalSpend',
                      'AvgPurchasesPerMonth', 'AvgPurchaseValue']

    for idx in incomplete_row_indices:
        n_fields_missing = np.random.randint(4, 7)
        fields_to_null = np.random.choice(nullable_fields, size=n_fields_missing, replace=False)
        df_dirty.loc[idx, fields_to_null] = np.nan

    print(f"✓ Added {n_incomplete_rows} incomplete rows (each with 4-6 missing values)")

    # Re-sort by PlayerID to maintain order
    df_dirty = df_dirty.sort_values('PlayerID').reset_index(drop=True)

    print(f"\n=== Data Quality Summary ===")
    print(f"Final dataset size: {len(df_dirty)} rows (original: {len(df)})")
    print(f"Total missing values: {df_dirty.isnull().sum().sum()}")
    print(f"Columns with missing values: {(df_dirty.isnull().sum() > 0).sum()}")

    return df_dirty


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate online gaming behavior dataset')
    parser.add_argument('--rows', type=int, default=10000, help='Target number of rows')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='generated_gaming_dataset.csv',
                       help='Output CSV filename')

    args = parser.parse_args()

    # Create configuration
    config = Config(
        num_rows=args.rows,
        random_seed=args.seed
    )

    # Generate dataset
    generator = DatasetGenerator(config)
    dataset = generator.generate()
    final_dataset = generator.get_dataset()

    # Introduce data quality issues for teaching purposes
    final_dataset = introduce_data_quality_issues(final_dataset, random_seed=config.random_seed)

    # Save to CSV
    final_dataset.to_csv(args.output, index=False)
    print(f"\nDataset saved to: {args.output}")
    print(f"Total rows: {len(final_dataset)}")
    print(f"Unique players: {final_dataset['PlayerID'].nunique()}")

    # Print quick summary
    print("\n=== Quick Summary ===")
    print(f"Gender distribution:\n{final_dataset['Gender'].value_counts(normalize=True)}")
    print(f"\nLocation distribution:\n{final_dataset['Location'].value_counts(normalize=True)}")
    print(f"\nGenre distribution:\n{final_dataset['GameGenre'].value_counts(normalize=True)}")
    print(f"\nPlayerExpertise distribution:\n{final_dataset['PlayerExpertise'].value_counts(normalize=True)}")
    print(f"\nSpendingPropensity distribution:\n{final_dataset['SpendingPropensity'].value_counts(normalize=True)}")


if __name__ == '__main__':
    main()
