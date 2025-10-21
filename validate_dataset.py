"""
Dataset Validation Script

Validates the generated gaming behavior dataset using Pandas and SciPy.
Checks distributions, correlations, logical consistency, and data quality.

Usage:
    python validate_dataset.py generated_gaming_dataset.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
import sys
from typing import Dict, Tuple


class DatasetValidator:
    """Validates generated dataset quality"""

    def __init__(self, filepath: str):
        """Load dataset from CSV"""
        print(f"Loading dataset from: {filepath}")
        self.df = pd.read_csv(filepath)
        print(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
        print(f"Unique players: {self.df['PlayerID'].nunique()}\n")

        self.issues = []
        self.warnings = []

    def validate_all(self):
        """Run all validation checks"""
        print("="*70)
        print("DATASET VALIDATION REPORT")
        print("="*70)

        self.check_missing_values()
        self.check_data_types()
        self.check_value_ranges()
        self.check_distributions()
        self.check_correlations()
        self.check_logical_consistency()
        self.check_multi_game_players()

        self.print_summary()

    def check_missing_values(self):
        """Check for missing values"""
        print("\n[1] MISSING VALUES CHECK")
        print("-" * 70)

        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("✓ No missing values found")
        else:
            print("✗ Missing values detected:")
            print(missing[missing > 0])
            self.issues.append("Missing values detected")

    def check_data_types(self):
        """Check data types are correct"""
        print("\n[2] DATA TYPE CHECK")
        print("-" * 70)

        expected_types = {
            'PlayerID': 'int',
            'Age': 'int',
            'Gender': 'object',
            'Location': 'object',
            'GameID': 'object',
            'GameName': 'object',
            'GameGenre': 'object',
            'GameDifficulty': 'object',
            'PlayTimeHours': 'float',
            'SessionsPerWeek': 'int',
            'AvgSessionDurationMinutes': 'int',
            'PlayerLevel': 'int',
            'AchievementsUnlocked': 'int',
            'EngagementLevel': 'object',
            'DaysPlayed': 'int',
            'PurchaseCount': 'int',
            'TotalSpend': 'float',
            'AvgPurchasesPerMonth': 'float',
            'AvgPurchaseValue': 'float',
            'SpendingPropensity': 'object',
            'PlayerExpertise': 'object',
        }

        type_issues = []
        for col, expected in expected_types.items():
            actual = self.df[col].dtype
            if expected == 'int' and not pd.api.types.is_integer_dtype(actual):
                type_issues.append(f"{col}: expected int, got {actual}")
            elif expected == 'float' and not pd.api.types.is_float_dtype(actual):
                type_issues.append(f"{col}: expected float, got {actual}")
            elif expected == 'object' and actual != 'object':
                type_issues.append(f"{col}: expected object, got {actual}")

        if not type_issues:
            print("✓ All data types correct")
        else:
            print("✗ Data type issues:")
            for issue in type_issues:
                print(f"  - {issue}")
            self.issues.extend(type_issues)

    def check_value_ranges(self):
        """Check values are within expected ranges"""
        print("\n[3] VALUE RANGE CHECK")
        print("-" * 70)

        range_checks = [
            ('Age', 13, 65),
            ('PlayerLevel', 1, 100),
            ('SessionsPerWeek', 1, 21),
            ('AvgSessionDurationMinutes', 15, 240),
            ('PlayTimeHours', 0.5, None),
            ('DaysPlayed', 7, 1095),
            ('AchievementsUnlocked', 0, 150),
            ('PurchaseCount', 0, None),
            ('TotalSpend', 0, None),
            ('AvgPurchasesPerMonth', 0, None),
            ('AvgPurchaseValue', 0, None),
        ]

        range_issues = []
        for col, min_val, max_val in range_checks:
            if min_val is not None and self.df[col].min() < min_val:
                range_issues.append(f"{col}: min {self.df[col].min()} < {min_val}")
            if max_val is not None and self.df[col].max() > max_val:
                range_issues.append(f"{col}: max {self.df[col].max()} > {max_val}")

        if not range_issues:
            print("✓ All values within expected ranges")
        else:
            print("⚠ Range warnings:")
            for issue in range_issues:
                print(f"  - {issue}")
            self.warnings.extend(range_issues)

    def check_distributions(self):
        """Check distributions match specifications"""
        print("\n[4] DISTRIBUTION CHECK")
        print("-" * 70)

        # Gender distribution (target: 63% M, 35% F, 2% Other)
        print("\nGender Distribution:")
        gender_dist = self.df['Gender'].value_counts(normalize=True).sort_index()
        print(gender_dist)
        expected_gender = {'Female': 0.35, 'Male': 0.63, 'Other': 0.02}
        for gender, expected_pct in expected_gender.items():
            actual_pct = gender_dist.get(gender, 0)
            diff = abs(actual_pct - expected_pct)
            status = "✓" if diff < 0.03 else "⚠"
            print(f"  {status} {gender}: {actual_pct:.2%} (target: {expected_pct:.0%}, diff: {diff:.2%})")

        # Location distribution (target: 40% USA, 35% Europe, 25% Asia)
        print("\nLocation Distribution:")
        location_dist = self.df['Location'].value_counts(normalize=True).sort_index()
        print(location_dist)
        expected_location = {'USA': 0.40, 'Europe': 0.35, 'Asia': 0.25}
        for location, expected_pct in expected_location.items():
            actual_pct = location_dist.get(location, 0)
            diff = abs(actual_pct - expected_pct)
            status = "✓" if diff < 0.03 else "⚠"
            print(f"  {status} {location}: {actual_pct:.2%} (target: {expected_pct:.0%}, diff: {diff:.2%})")

        # Genre distribution (target: 40% RPG, 40% Action, 20% Strategy)
        print("\nGenre Distribution:")
        genre_dist = self.df['GameGenre'].value_counts(normalize=True).sort_index()
        print(genre_dist)
        expected_genre = {'Action': 0.40, 'RPG': 0.40, 'Strategy': 0.20}
        for genre, expected_pct in expected_genre.items():
            actual_pct = genre_dist.get(genre, 0)
            diff = abs(actual_pct - expected_pct)
            status = "✓" if diff < 0.05 else "⚠"
            print(f"  {status} {genre}: {actual_pct:.2%} (target: {expected_pct:.0%}, diff: {diff:.2%})")

        # PlayerExpertise (target: 35% Beginner, 45% Intermediate, 20% Expert)
        print("\nPlayerExpertise Distribution:")
        expertise_dist = self.df['PlayerExpertise'].value_counts(normalize=True).sort_index()
        print(expertise_dist)
        expected_expertise = {'Beginner': 0.35, 'Intermediate': 0.45, 'Expert': 0.20}
        for level, expected_pct in expected_expertise.items():
            actual_pct = expertise_dist.get(level, 0)
            diff = abs(actual_pct - expected_pct)
            status = "✓" if diff < 0.05 else "⚠"
            print(f"  {status} {level}: {actual_pct:.2%} (target: {expected_pct:.0%}, diff: {diff:.2%})")

        # SpendingPropensity (target: 55% NonSpender, 35% Occasional, 10% Whale)
        print("\nSpendingPropensity Distribution:")
        spending_dist = self.df['SpendingPropensity'].value_counts(normalize=True).sort_index()
        print(spending_dist)
        expected_spending = {'NonSpender': 0.55, 'Occasional': 0.35, 'Whale': 0.10}
        for level, expected_pct in expected_spending.items():
            actual_pct = spending_dist.get(level, 0)
            diff = abs(actual_pct - expected_pct)
            status = "✓" if diff < 0.05 else "⚠"
            print(f"  {status} {level}: {actual_pct:.2%} (target: {expected_pct:.0%}, diff: {diff:.2%})")

        # EngagementLevel (target: 30% Low, 45% Medium, 25% High)
        print("\nEngagementLevel Distribution:")
        engagement_dist = self.df['EngagementLevel'].value_counts(normalize=True).sort_index()
        print(engagement_dist)
        expected_engagement = {'Low': 0.30, 'Medium': 0.45, 'High': 0.25}
        for level, expected_pct in expected_engagement.items():
            actual_pct = engagement_dist.get(level, 0)
            diff = abs(actual_pct - expected_pct)
            status = "✓" if diff < 0.05 else "⚠"
            print(f"  {status} {level}: {actual_pct:.2%} (target: {expected_pct:.0%}, diff: {diff:.2%})")

    def check_correlations(self):
        """Check key correlations"""
        print("\n[5] CORRELATION CHECK")
        print("-" * 70)

        # Expected correlations
        correlations_to_check = [
            ('PlayTimeHours', 'PlayerLevel', 0.75, 0.10),
            ('PlayTimeHours', 'AchievementsUnlocked', 0.60, 0.10),
            ('DaysPlayed', 'PlayTimeHours', 0.70, 0.10),
            ('PurchaseCount', 'TotalSpend', 0.85, 0.10),
        ]

        print("\nKey Correlations (Pearson r):")
        for var1, var2, target, tolerance in correlations_to_check:
            corr = self.df[var1].corr(self.df[var2])
            diff = abs(corr - target)
            status = "✓" if diff <= tolerance else "⚠"
            print(f"  {status} {var1} ↔ {var2}: {corr:.3f} (target: {target:.2f} ±{tolerance:.2f})")

        # Additional interesting correlations
        print("\nAdditional Correlations:")
        additional_corrs = [
            ('Age', 'TotalSpend'),
            ('SessionsPerWeek', 'EngagementLevel (numeric)'),
            ('PlayerLevel', 'AchievementsUnlocked'),
        ]

        # For EngagementLevel, convert to numeric
        engagement_map = {'Low': 0, 'Medium': 1, 'High': 2}
        df_temp = self.df.copy()
        df_temp['EngagementLevel_numeric'] = df_temp['EngagementLevel'].map(engagement_map)

        print(f"  Age ↔ TotalSpend: {self.df['Age'].corr(self.df['TotalSpend']):.3f}")
        print(f"  SessionsPerWeek ↔ EngagementLevel: {self.df['SessionsPerWeek'].corr(df_temp['EngagementLevel_numeric']):.3f}")
        print(f"  PlayerLevel ↔ AchievementsUnlocked: {self.df['PlayerLevel'].corr(self.df['AchievementsUnlocked']):.3f}")

    def check_logical_consistency(self):
        """Check logical consistency rules"""
        print("\n[6] LOGICAL CONSISTENCY CHECK")
        print("-" * 70)

        consistency_issues = []

        # Rule 1: NonSpenders must have PurchaseCount=0 and TotalSpend=0
        non_spenders = self.df[self.df['SpendingPropensity'] == 'NonSpender']
        invalid_non_spenders = non_spenders[
            (non_spenders['PurchaseCount'] != 0) | (non_spenders['TotalSpend'] != 0)
        ]
        if len(invalid_non_spenders) > 0:
            consistency_issues.append(f"NonSpenders with purchases: {len(invalid_non_spenders)}")

        # Rule 2: AvgPurchaseValue = TotalSpend / PurchaseCount (when PurchaseCount > 0)
        purchasers = self.df[self.df['PurchaseCount'] > 0].copy()
        purchasers['calculated_avg'] = purchasers['TotalSpend'] / purchasers['PurchaseCount']
        diff = (purchasers['AvgPurchaseValue'] - purchasers['calculated_avg']).abs()
        invalid_avg_value = diff[diff > 0.01]  # Allow small rounding errors
        if len(invalid_avg_value) > 0:
            consistency_issues.append(f"Incorrect AvgPurchaseValue: {len(invalid_avg_value)}")

        # Rule 3: AvgPurchasesPerMonth = PurchaseCount / (DaysPlayed/30)
        self.df['calculated_avg_per_month'] = self.df['PurchaseCount'] / (self.df['DaysPlayed'] / 30)
        diff = (self.df['AvgPurchasesPerMonth'] - self.df['calculated_avg_per_month']).abs()
        invalid_avg_per_month = diff[diff > 0.01]
        if len(invalid_avg_per_month) > 0:
            consistency_issues.append(f"Incorrect AvgPurchasesPerMonth: {len(invalid_avg_per_month)}")

        # Rule 4: Spending categories match TotalSpend
        spending_mismatch = 0
        for _, row in self.df.iterrows():
            expected_category = None
            if row['TotalSpend'] == 0:
                expected_category = 'NonSpender'
            elif row['TotalSpend'] <= 80:
                expected_category = 'Occasional'
            else:
                expected_category = 'Whale'

            if row['SpendingPropensity'] != expected_category:
                spending_mismatch += 1

        if spending_mismatch > 0:
            consistency_issues.append(f"Spending category mismatch: {spending_mismatch}")

        # Rule 5: PlayerLevel <= max_level for game
        # (Check a sample since we don't have game catalog here)

        if not consistency_issues:
            print("✓ All logical consistency checks passed")
        else:
            print("✗ Logical consistency issues found:")
            for issue in consistency_issues:
                print(f"  - {issue}")
            self.issues.extend(consistency_issues)

    def check_multi_game_players(self):
        """Check multi-game player consistency"""
        print("\n[7] MULTI-GAME PLAYER CONSISTENCY CHECK")
        print("-" * 70)

        # Group by PlayerID
        player_groups = self.df.groupby('PlayerID')

        # Check that Age, Gender, and Location are consistent for same player
        # These are player-level attributes that should not vary across games
        inconsistent_players = []

        for player_id, group in player_groups:
            if len(group) > 1:  # Multi-game player
                if group['Age'].nunique() > 1:
                    inconsistent_players.append(f"Player {player_id}: inconsistent Age")
                if group['Gender'].nunique() > 1:
                    inconsistent_players.append(f"Player {player_id}: inconsistent Gender")
                if group['Location'].nunique() > 1:
                    inconsistent_players.append(f"Player {player_id}: inconsistent Location")

        if not inconsistent_players:
            print("✓ Multi-game player demographics consistent")
        else:
            print(f"✗ Found {len(inconsistent_players)} inconsistencies:")
            for issue in inconsistent_players[:10]:  # Show first 10
                print(f"  - {issue}")
            if len(inconsistent_players) > 10:
                print(f"  ... and {len(inconsistent_players) - 10} more")
            self.issues.extend(inconsistent_players)

        # Show distribution of games per player
        games_per_player = player_groups.size().value_counts().sort_index()
        print("\nGames per player distribution:")
        for num_games, count in games_per_player.items():
            pct = count / len(player_groups) * 100
            print(f"  {num_games} game(s): {count} players ({pct:.1f}%)")

    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)

        if not self.issues and not self.warnings:
            print("✓ ALL CHECKS PASSED - Dataset is valid!")
        else:
            if self.issues:
                print(f"✗ Found {len(self.issues)} ISSUES:")
                for issue in self.issues[:10]:
                    print(f"  - {issue}")
                if len(self.issues) > 10:
                    print(f"  ... and {len(self.issues) - 10} more issues")

            if self.warnings:
                print(f"\n⚠ Found {len(self.warnings)} WARNINGS:")
                for warning in self.warnings[:10]:
                    print(f"  - {warning}")
                if len(self.warnings) > 10:
                    print(f"  ... and {len(self.warnings) - 10} more warnings")

        print("\n" + "="*70)

    def generate_eda_summary(self):
        """Generate additional EDA insights"""
        print("\n" + "="*70)
        print("EDA INSIGHTS")
        print("="*70)

        # Descriptive statistics for key numeric features
        print("\nDescriptive Statistics (Key Features):")
        print(self.df[['Age', 'PlayTimeHours', 'PlayerLevel', 'AchievementsUnlocked',
                      'TotalSpend', 'DaysPlayed']].describe())

        # Genre by Location
        print("\n\nGenre preferences by Location:")
        print(pd.crosstab(self.df['Location'], self.df['GameGenre'], normalize='index'))

        # Gender by Genre
        print("\n\nGender distribution by Genre:")
        print(pd.crosstab(self.df['Gender'], self.df['GameGenre'], normalize='columns'))

        # Spending by Engagement
        print("\n\nAverage spending by EngagementLevel:")
        print(self.df.groupby('EngagementLevel')['TotalSpend'].agg(['mean', 'median', 'count']))

        # Expertise by Difficulty
        print("\n\nExpertise by GameDifficulty:")
        print(pd.crosstab(self.df['PlayerExpertise'], self.df['GameDifficulty'], normalize='index'))


def main():
    parser = argparse.ArgumentParser(description='Validate gaming behavior dataset')
    parser.add_argument('filepath', type=str, help='Path to CSV file to validate')
    parser.add_argument('--eda', action='store_true', help='Show additional EDA insights')

    args = parser.parse_args()

    validator = DatasetValidator(args.filepath)
    validator.validate_all()

    if args.eda:
        validator.generate_eda_summary()

    # Return exit code based on validation
    if validator.issues:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
