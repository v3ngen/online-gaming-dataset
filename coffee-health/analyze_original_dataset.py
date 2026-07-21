"""
Quantitative limitations analysis of the original coffee-health dataset
(data/coffee_health_original.csv), a modified version of the "Global Coffee
Health Dataset" (Kaggle, uom190346a).

Purpose: confirm and quantify the issues identified qualitatively before
writing DATA_GENERATION_SPEC.md, per step (a) of the process in the repo
root README.md.

Usage:
    python3 analyze_original_dataset.py
"""

import pandas as pd
import numpy as np

pd.set_option('display.width', 120)
pd.set_option('display.max_columns', 20)

df = pd.read_csv('data/coffee_health_original.csv')

print('=' * 80)
print('SHAPE / DTYPES / MISSING VALUES')
print('=' * 80)
print(f'Shape: {df.shape}')
print(df.dtypes)
print('\nMissing values (count / %):')
missing = df.isnull().sum()
print(pd.DataFrame({'missing': missing, 'pct': (missing / len(df) * 100).round(2)}))
print(f'\nDuplicate rows: {df.duplicated().sum()}')

print('\n' + '=' * 80)
print('NUMERIC DESCRIPTIVE STATS')
print('=' * 80)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[numeric_cols].describe().T)

print('\n' + '=' * 80)
print('CATEGORICAL VALUE COUNTS')
print('=' * 80)
categorical_cols = ['Gender', 'Country', 'Sleep Quality', 'Physical Activity',
                     'Health Issues', 'Occupation', 'Smoker', 'Drinks Alcohol', 'Stress Level']
for col in categorical_cols:
    print(f'\n--- {col} ---')
    print(df[col].value_counts(dropna=False))

print('\n' + '=' * 80)
print('CORRELATION MATRIX (numeric features)')
print('=' * 80)
print(df[numeric_cols].corr().round(2))

print('\n' + '=' * 80)
print('CHECK 1: Daily Coffees vs Caffeine Intake')
print('=' * 80)
d = df.dropna(subset=['Daily Coffees', 'Caffeine Intake'])
print(f"Correlation: {d['Daily Coffees'].corr(d['Caffeine Intake']):.3f}")
d = d[d['Daily Coffees'] > 0].copy()
d['mg_per_cup'] = d['Caffeine Intake'] / d['Daily Coffees']
print('Implied mg-per-cup distribution (README says ~95mg/cup):')
print(d['mg_per_cup'].describe())

print('\n' + '=' * 80)
print('CHECK 2: Sleep Quality vs Sleep Hours (README: "based on sleep hours")')
print('=' * 80)
print(df.groupby('Sleep Quality')['Sleep Hours'].describe())
print('\nSleep Quality vs other lifestyle factors it SHOULD plausibly relate to')
print('(caffeine, stress, smoking, alcohol) -- expect weak/no relationship if')
print('quality really is sleep-hours-only:')
sq_order = ['Poor', 'Fair', 'Good', 'Excellent']
sq_map = {v: i for i, v in enumerate(sq_order)}
df['_sq_ord'] = df['Sleep Quality'].map(sq_map)
print(f"corr(Sleep Quality ordinal, Caffeine Intake) = {df['_sq_ord'].corr(df['Caffeine Intake']):.3f}")
print('\nMean Sleep Quality (ordinal, higher=better) by Stress Level:')
print(df.groupby('Stress Level')['_sq_ord'].mean())
print('\nMean Sleep Quality (ordinal) by Smoker:')
print(df.groupby('Smoker')['_sq_ord'].mean())
print('\nMean Sleep Quality (ordinal) by Drinks Alcohol:')
print(df.groupby('Drinks Alcohol')['_sq_ord'].mean())
print('\nCrosstab Stress Level x Sleep Quality (row %):')
print((pd.crosstab(df['Stress Level'], df['Sleep Quality'], normalize='index') * 100).round(1)[sq_order])

print('\n' + '=' * 80)
print('CHECK 3: Physical Activity vs Sleep Hours + lifestyle')
print('=' * 80)
print('NOTE: README describes Physical Activity as categorical Low/Medium/High,')
print('but the actual column is numeric (0-15) -- doc and data disagree.')
print(df['Physical Activity'].describe())
df['_pa_ord'] = df['Physical Activity']
print(f"\ncorr(Physical Activity, Sleep Hours) = {df['_pa_ord'].corr(df['Sleep Hours']):.3f}")
print(f"corr(Physical Activity, BMI) = {df['_pa_ord'].corr(df['BMI']):.3f}")
print(f"corr(Physical Activity, Heart Rate) = {df['_pa_ord'].corr(df['Heart Rate']):.3f}")
print('Mean Physical Activity by Occupation:')
print(df.groupby('Occupation')['_pa_ord'].mean())

print('\n' + '=' * 80)
print('CHECK 4: Health Issues vs Age / BMI / Sleep (README: "based on age, BMI and sleep")')
print('=' * 80)
hi_order = ['None', 'Mild', 'Moderate', 'Severe']
hi_map = {v: i for i, v in enumerate(hi_order)}
df['_hi_ord'] = df['Health Issues'].map(hi_map)
print(f"corr(Health Issues ordinal, Age) = {df['_hi_ord'].corr(df['Age']):.3f}")
print(f"corr(Health Issues ordinal, BMI) = {df['_hi_ord'].corr(df['BMI']):.3f}")
print(f"corr(Health Issues ordinal, Sleep Hours) = {df['_hi_ord'].corr(df['Sleep Hours']):.3f}")
print('\nHealth Issues vs lifestyle factors it currently ignores (smoking, alcohol, stress, activity):')
print(f"corr(Health Issues ordinal, Physical Activity ordinal) = {df['_hi_ord'].corr(df['_pa_ord']):.3f}")
print(f"corr(Health Issues ordinal, Stress Level ordinal) = ", end='')
stress_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['_stress_ord'] = df['Stress Level'].map(stress_map)
print(f"{df['_hi_ord'].corr(df['_stress_ord']):.3f}")

print('\n' + '=' * 80)
print('CHECK 5: Do Smoker / Drinks Alcohol relate to anything realistic?')
print('=' * 80)
print('Mean BMI, Heart Rate, Stress ordinal by Smoker:')
print(df.groupby('Smoker')[['BMI', 'Heart Rate', '_stress_ord']].mean())
print('\nMean BMI, Heart Rate, Stress ordinal by Drinks Alcohol:')
print(df.groupby('Drinks Alcohol')[['BMI', 'Heart Rate', '_stress_ord']].mean())

print('\n' + '=' * 80)
print('CHECK 6: Country distribution + any country-level signal')
print('=' * 80)
print(df['Country'].value_counts())
print('\nMean Daily Coffees, Caffeine Intake by Country:')
print(df.groupby('Country')[['Daily Coffees', 'Caffeine Intake']].mean().round(2))
