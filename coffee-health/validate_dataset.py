"""
Validation script for the coffee & health dataset.

Checks the generated CSV against DATA_GENERATION_SPEC.md / DATA_GENERATION_ALGORITHM.md:
ranges, distributions, target-variable "good or better" rates per country,
key correlations, and data quality issue counts.

Usage:
    python3 validate_dataset.py generated_coffee_health_dataset.csv
"""

import sys
import pandas as pd
import numpy as np

COUNTRY_TARGETS_GOOD_PLUS = {'Norway': 0.80, 'Italy': 0.755, 'France': 0.685, 'UK': 0.65}
SRH_ORDER = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
ACTIVITY_ORDER = ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active']
STRESS_ORDER = ['Low', 'Medium', 'High']
SLEEP_Q_ORDER = ['Poor', 'Fair', 'Good', 'Excellent']
HEALTH_ORDER = ['No Issues', 'Mild', 'Moderate', 'Severe']


def ordinal(series, order):
    return series.map({v: i for i, v in enumerate(order)})


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'generated_coffee_health_dataset.csv'
    df = pd.read_csv(path)

    print('=' * 80)
    print(f'VALIDATING: {path}  ({len(df)} rows)')
    print('=' * 80)

    print('\n--- Shape & dtypes ---')
    print(f'Shape: {df.shape}')
    print(f'Duplicate rows: {df.duplicated().sum()} ({df.duplicated().mean() * 100:.2f}%)')

    print('\n--- Missing values ---')
    missing = df.isnull().sum()
    print(pd.DataFrame({'missing': missing, 'pct': (missing / len(df) * 100).round(2)}))

    print('\n--- Range checks ---')
    checks = [
        ('Age', 13, 199),  # includes typo anomalies up to 199
        ('BMI', 16, 45),
        ('Heart Rate', 45, 110),
        ('Sleep Hours', 3, 10.5),
        ('Daily Coffees', 0, 9),
        ('Caffeine Intake', 0, 750),
    ]
    for col, lo, hi in checks:
        s = df[col].dropna()
        ok = s.between(lo, hi).all()
        print(f'{col}: [{s.min():.1f}, {s.max():.1f}] within [{lo}, {hi}]? {"OK" if ok else "FAIL"}')
    n_age_anomalies = ((df['Age'] > 100) | (df['Age'] < 13)).sum()
    print(f'Age anomalies (>100 or <13): {n_age_anomalies} ({n_age_anomalies / len(df) * 100:.2f}%)')

    print('\n--- Categorical distributions ---')
    for col in ['Country', 'Gender', 'Smoking Status', 'Alcohol Level',
                'Physical Activity Level', 'Stress Level', 'Sleep Quality',
                'Health Issues', 'SelfRatedHealth']:
        print(f'\n{col}:')
        print((df[col].value_counts(normalize=True, dropna=False) * 100).round(1))

    print('\n' + '=' * 80)
    print('TARGET VALIDATION: SelfRatedHealth "Good or better" rate by Country')
    print('=' * 80)
    good_plus = df['SelfRatedHealth'].isin(['Good', 'Very Good', 'Excellent'])
    achieved = df.assign(_good_plus=good_plus).groupby('Country')['_good_plus'].mean()
    for country, target in COUNTRY_TARGETS_GOOD_PLUS.items():
        a = achieved.get(country, float('nan'))
        diff = (a - target) * 100
        flag = 'OK' if abs(diff) <= 5 else 'ADJUST'
        print(f'{country}: achieved {a*100:.1f}%  target {target*100:.1f}%  diff {diff:+.1f}pp  [{flag}]')

    print('\n' + '=' * 80)
    print('KEY CORRELATIONS (expected sign in brackets)')
    print('=' * 80)
    df['_activity_ord'] = ordinal(df['Physical Activity Level'], ACTIVITY_ORDER)
    df['_stress_ord'] = ordinal(df['Stress Level'], STRESS_ORDER)
    df['_sleepq_ord'] = ordinal(df['Sleep Quality'], SLEEP_Q_ORDER)
    df['_health_ord'] = ordinal(df['Health Issues'], HEALTH_ORDER)
    df['_srh_ord'] = ordinal(df['SelfRatedHealth'], SRH_ORDER)

    pairs = [
        ('BMI', '_activity_ord', 'negative'),
        ('Heart Rate', '_activity_ord', 'negative'),
        ('_sleepq_ord', '_stress_ord', 'negative'),
        ('_srh_ord', '_health_ord', 'negative'),
        ('Daily Coffees', 'Caffeine Intake', 'positive'),
    ]
    for a, b, expected in pairs:
        r = df[a].corr(df[b])
        sign_ok = (r < 0) if expected == 'negative' else (r > 0)
        print(f'corr({a}, {b}) = {r:.3f}  expected {expected}  [{"OK" if sign_ok else "FAIL"}]')

    print('\n' + '=' * 80)
    print('LOGICAL CONSISTENCY')
    print('=' * 80)
    never_or_former = df['Smoking Status'].isin(['Never', 'Former'])
    print(f'Never/Former smokers with no intensity label (expected, not applicable): {never_or_former.sum()} rows')
    n_immune_missing = df[['ID', 'Country', 'SelfRatedHealth']].isnull().sum().sum()
    print(f'Missing values in immune fields (ID, Country, SelfRatedHealth) — should be 0: {n_immune_missing}')
    print(f'Duplicate IDs (excluding intentional duplicate rows): {df["ID"].duplicated().sum() - df.duplicated().sum()}')


if __name__ == '__main__':
    main()
