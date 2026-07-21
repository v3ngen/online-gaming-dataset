"""
Coffee & Health Dataset Generator

Generates a coherent synthetic dataset for educational purposes (EDA and ML).
Based on DATA_GENERATION_SPEC.md and DATA_GENERATION_ALGORITHM.md.

Usage:
    python3 generate_dataset.py [--rows 10000] [--seed 42] [--output generated_coffee_health_dataset.csv]
"""

import argparse
import numpy as np
import pandas as pd

COUNTRIES = ['Italy', 'France', 'UK', 'Norway']

SMOKING_DIST = {
    'Norway': {'Never': 0.658, 'Former': 0.20, 'Light Smoker': 0.092, 'Heavy Smoker': 0.050},
    'Italy':  {'Never': 0.586, 'Former': 0.18, 'Light Smoker': 0.152, 'Heavy Smoker': 0.082},
    'France': {'Never': 0.504, 'Former': 0.15, 'Light Smoker': 0.225, 'Heavy Smoker': 0.121},
    'UK':     {'Never': 0.595, 'Former': 0.28, 'Light Smoker': 0.081, 'Heavy Smoker': 0.044},
}

ALCOHOL_DIST = {
    'Norway': {'Non-Drinker': 0.25, 'Light': 0.40, 'Moderate': 0.25, 'Heavy': 0.10},
    'Italy':  {'Non-Drinker': 0.15, 'Light': 0.40, 'Moderate': 0.35, 'Heavy': 0.10},
    'UK':     {'Non-Drinker': 0.12, 'Light': 0.28, 'Moderate': 0.40, 'Heavy': 0.20},
    'France': {'Non-Drinker': 0.10, 'Light': 0.30, 'Moderate': 0.40, 'Heavy': 0.20},
}

COFFEE_PARAMS = {
    'Norway': {'cups_mean': 3.2, 'cups_std': 1.0, 'mg_per_cup': 110},
    'Italy':  {'cups_mean': 3.0, 'cups_std': 1.1, 'mg_per_cup': 65},
    'France': {'cups_mean': 2.4, 'cups_std': 0.9, 'mg_per_cup': 80},
    'UK':     {'cups_mean': 1.8, 'cups_std': 0.9, 'mg_per_cup': 70},
}

STRESS_DIST = {
    'Norway': {'Low': 0.60, 'Medium': 0.30, 'High': 0.10},
    'Italy':  {'Low': 0.50, 'Medium': 0.35, 'High': 0.15},
    'UK':     {'Low': 0.45, 'Medium': 0.35, 'High': 0.20},
    'France': {'Low': 0.40, 'Medium': 0.35, 'High': 0.25},
}

ACTIVITY_DIST = {
    'Norway': {'Sedentary': 0.15, 'Lightly Active': 0.25, 'Moderately Active': 0.35, 'Very Active': 0.25},
    'Italy':  {'Sedentary': 0.25, 'Lightly Active': 0.30, 'Moderately Active': 0.30, 'Very Active': 0.15},
    'France': {'Sedentary': 0.25, 'Lightly Active': 0.32, 'Moderately Active': 0.28, 'Very Active': 0.15},
    'UK':     {'Sedentary': 0.30, 'Lightly Active': 0.30, 'Moderately Active': 0.25, 'Very Active': 0.15},
}

BMI_PARAMS = {
    'Norway': {'mean': 25.5, 'std': 4.2},
    'Italy':  {'mean': 24.0, 'std': 3.5},
    'France': {'mean': 25.7, 'std': 4.3},
    'UK':     {'mean': 26.8, 'std': 4.8},
}

COUNTRY_OFFSET_SRH = {'Norway': -1.0, 'Italy': -1.75, 'France': 1.25, 'UK': -3.35}

STRESS_ORDER = ['Low', 'Medium', 'High']
ACTIVITY_ORDER = ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active']
SLEEP_Q_ORDER = ['Poor', 'Fair', 'Good', 'Excellent']
HEALTH_ISSUES_ORDER = ['No Issues', 'Mild', 'Moderate', 'Severe']
SRH_ORDER = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']

SMOKING_BMI_ADJ = {'Never': 0.0, 'Former': 0.0, 'Light Smoker': 0.0, 'Heavy Smoker': -0.5}
SMOKING_HR_ADJ = {'Never': 0.0, 'Former': 1.0, 'Light Smoker': 3.0, 'Heavy Smoker': 6.0}
SMOKING_SLEEP_ADJ = {'Never': 0.0, 'Former': -1.0, 'Light Smoker': -4.0, 'Heavy Smoker': -8.0}
SMOKING_HEALTH_ADJ = {'Never': 0.0, 'Former': 2.0, 'Light Smoker': 5.0, 'Heavy Smoker': 12.0}
SMOKING_SRH_ADJ = {'Never': 0.0, 'Former': -3.0, 'Light Smoker': -8.0, 'Heavy Smoker': -18.0}

ALCOHOL_SLEEP_ADJ = {'Non-Drinker': 0.0, 'Light': -1.0, 'Moderate': -4.0, 'Heavy': -10.0}

ACTIVITY_BMI_ADJ = {'Sedentary': 1.0, 'Lightly Active': 0.0, 'Moderately Active': -0.7, 'Very Active': -1.5}
ACTIVITY_HR_ADJ = {'Sedentary': 3.0, 'Lightly Active': 0.0, 'Moderately Active': -3.0, 'Very Active': -6.0}
ACTIVITY_SRH_ADJ = {'Sedentary': -8.0, 'Lightly Active': -2.0, 'Moderately Active': 5.0, 'Very Active': 12.0}

STRESS_HR_ADJ = {'Low': 0.0, 'Medium': 2.0, 'High': 4.0}
STRESS_SLEEP_ADJ = {'Low': 0.0, 'Medium': -6.0, 'High': -15.0}
STRESS_SRH_ADJ = {'Low': 0.0, 'Medium': -6.0, 'High': -14.0}

SLEEP_Q_SRH_ADJ = {'Poor': -12.0, 'Fair': -4.0, 'Good': 5.0, 'Excellent': 12.0}
HEALTH_SRH_ADJ = {'No Issues': 0.0, 'Mild': -10.0, 'Moderate': -25.0, 'Severe': -45.0}
GENDER_SRH_ADJ = {'Male': 0.0, 'Female': -3.0, 'Other': 0.0}


def sample_categorical_by_group(groups, dist_map, rng):
    result = np.empty(len(groups), dtype=object)
    groups = np.asarray(groups)
    for group_value, probs in dist_map.items():
        mask = groups == group_value
        n = int(mask.sum())
        if n == 0:
            continue
        cats = list(probs.keys())
        p = list(probs.values())
        result[mask] = rng.choice(cats, size=n, p=p)
    return result


def shift_ordinal(values, order, amount):
    """Shift categorical values along an ordered scale by `amount` steps (can be negative)."""
    idx = {v: i for i, v in enumerate(order)}
    codes = np.array([idx[v] for v in values])
    codes = np.clip(codes + amount, 0, len(order) - 1)
    return np.array(order)[codes]


def bucket(scores, thresholds, labels):
    """thresholds: ascending upper bounds for all but the last label."""
    idx = np.searchsorted(thresholds, scores, side='right')
    return np.array(labels)[idx]


def generate(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_per_country = n_rows // len(COUNTRIES)
    country = np.repeat(COUNTRIES, n_per_country)
    remainder = n_rows - len(country)
    if remainder > 0:
        country = np.concatenate([country, rng.choice(COUNTRIES, size=remainder)])
    rng.shuffle(country)

    n = len(country)
    df = pd.DataFrame({'ID': np.arange(1, n + 1), 'Country': country})

    # Level 0: demographics
    df['Age'] = np.round(rng.beta(2, 3, size=n) * 57 + 18).astype(int)
    df['Gender'] = rng.choice(['Male', 'Female', 'Other'], size=n, p=[0.48, 0.48, 0.04])

    # Level 1: country-driven lifestyle
    df['Smoking Status'] = sample_categorical_by_group(df['Country'], SMOKING_DIST, rng)
    df['Alcohol Level'] = sample_categorical_by_group(df['Country'], ALCOHOL_DIST, rng)

    cups_mean = df['Country'].map(lambda c: COFFEE_PARAMS[c]['cups_mean']).to_numpy()
    cups_std = df['Country'].map(lambda c: COFFEE_PARAMS[c]['cups_std']).to_numpy()
    mg_per_cup = df['Country'].map(lambda c: COFFEE_PARAMS[c]['mg_per_cup']).to_numpy()
    df['Daily Coffees'] = np.clip(rng.normal(cups_mean, cups_std), 0, 9).round(1)

    # ~10% of people lean decaf/half-caf (still order "coffees", but most cups are
    # low-caffeine) -- this is what keeps Daily Coffees <-> Caffeine Intake a strong
    # but imperfect correlation, with a real explanation rather than bare noise.
    decaf_leaning = rng.random(n) < 0.10
    decaf_multiplier = np.where(decaf_leaning, rng.uniform(0.05, 0.30, size=n), 1.0)
    effective_mg_per_cup = mg_per_cup * decaf_multiplier

    df['Caffeine Intake'] = np.clip(
        df['Daily Coffees'] * effective_mg_per_cup * rng.uniform(0.85, 1.15, size=n), 0, 750
    ).round(1)

    # Level 2: behavioral
    stress = sample_categorical_by_group(df['Country'], STRESS_DIST, rng)
    peak_age_mask = (df['Age'] >= 25) & (df['Age'] <= 45)
    escalate = peak_age_mask & (rng.random(n) < 0.15)
    stress = np.where(escalate, shift_ordinal(stress, STRESS_ORDER, 1), stress)
    df['Stress Level'] = stress

    activity = sample_categorical_by_group(df['Country'], ACTIVITY_DIST, rng)
    high_stress_mask = (df['Stress Level'] == 'High') & (rng.random(n) < 0.3)
    older_mask = (df['Age'] > 55) & (rng.random(n) < 0.3)
    activity = np.where(high_stress_mask, shift_ordinal(activity, ACTIVITY_ORDER, -1), activity)
    activity = np.where(older_mask, shift_ordinal(activity, ACTIVITY_ORDER, -1), activity)
    df['Physical Activity Level'] = activity

    # Level 3: physiology
    bmi_mean = df['Country'].map(lambda c: BMI_PARAMS[c]['mean']).to_numpy()
    bmi_std = df['Country'].map(lambda c: BMI_PARAMS[c]['std']).to_numpy()
    activity_bmi = df['Physical Activity Level'].map(ACTIVITY_BMI_ADJ).to_numpy()
    smoking_bmi = df['Smoking Status'].map(SMOKING_BMI_ADJ).to_numpy()
    age_bmi = 0.03 * np.minimum(df['Age'].to_numpy(), 60)
    df['BMI'] = np.clip(
        bmi_mean + activity_bmi + smoking_bmi + age_bmi + rng.normal(0, bmi_std),
        16, 45
    ).round(1)

    activity_hr = df['Physical Activity Level'].map(ACTIVITY_HR_ADJ).to_numpy()
    smoking_hr = df['Smoking Status'].map(SMOKING_HR_ADJ).to_numpy()
    stress_hr = df['Stress Level'].map(STRESS_HR_ADJ).to_numpy()
    age_hr = 0.05 * df['Age'].to_numpy()
    df['Avg Resting Heart Rate'] = np.clip(
        72 + activity_hr + smoking_hr + stress_hr + age_hr + rng.normal(0, 7, size=n),
        45, 110
    ).round(0)

    # Level 4: sleep
    stress_sleep_hours = df['Stress Level'].map({'Low': 0.0, 'Medium': -0.4, 'High': -0.9}).to_numpy()
    caffeine_sleep_hours = -0.0015 * df['Caffeine Intake'].to_numpy()
    df['Avg Sleep Hours Per Night'] = np.clip(
        7.0 + stress_sleep_hours + caffeine_sleep_hours + rng.normal(0, 1.0, size=n),
        3, 10.5
    ).round(1)

    stress_sleep_q = df['Stress Level'].map(STRESS_SLEEP_ADJ).to_numpy()
    smoking_sleep_q = df['Smoking Status'].map(SMOKING_SLEEP_ADJ).to_numpy()
    alcohol_sleep_q = df['Alcohol Level'].map(ALCOHOL_SLEEP_ADJ).to_numpy()
    sleep_q_score = (
        40 + 6 * (df['Avg Sleep Hours Per Night'].to_numpy() - 7)
        + stress_sleep_q + smoking_sleep_q + alcohol_sleep_q
        - 0.01 * df['Caffeine Intake'].to_numpy()
        + rng.normal(0, 8, size=n)
    )
    # Quantile-based thresholds (rather than hand-picked constants) so the marginal
    # distribution reliably hits the target shape regardless of how the weighted
    # sum's scale works out: Poor 10%, Fair 25%, Good 45%, Excellent 20%.
    sleep_q_thresholds = np.quantile(sleep_q_score, [0.10, 0.35, 0.80])
    df['Sleep Quality'] = bucket(sleep_q_score, sleep_q_thresholds, SLEEP_Q_ORDER)

    # Level 5: chronic health
    smoking_health = df['Smoking Status'].map(SMOKING_HEALTH_ADJ).to_numpy()
    age_health = 0.5 * np.maximum(0, df['Age'].to_numpy() - 30)
    bmi_health = 1.2 * np.maximum(0, np.abs(df['BMI'].to_numpy() - 22) - 3)
    health_score = 15 + age_health + bmi_health + smoking_health + rng.normal(0, 10, size=n)
    # Target: No Issues 55%, Mild 30%, Moderate 11%, Severe 4%.
    health_thresholds = np.quantile(health_score, [0.55, 0.85, 0.96])
    df['Health Issues'] = bucket(health_score, health_thresholds, HEALTH_ISSUES_ORDER)

    # Level 6: target
    country_offset = df['Country'].map(COUNTRY_OFFSET_SRH).to_numpy()
    gender_offset = df['Gender'].map(GENDER_SRH_ADJ).to_numpy()
    age_srh = -0.15 * np.maximum(0, df['Age'].to_numpy() - 30)
    activity_srh = df['Physical Activity Level'].map(ACTIVITY_SRH_ADJ).to_numpy()
    bmi_srh = -1.2 * np.maximum(0, np.abs(df['BMI'].to_numpy() - 22) - 3)
    sleep_q_srh = df['Sleep Quality'].map(SLEEP_Q_SRH_ADJ).to_numpy()
    smoking_srh = df['Smoking Status'].map(SMOKING_SRH_ADJ).to_numpy()
    stress_srh = df['Stress Level'].map(STRESS_SRH_ADJ).to_numpy()
    health_srh = df['Health Issues'].map(HEALTH_SRH_ADJ).to_numpy()

    srh_score = np.clip(
        60 + country_offset + gender_offset + age_srh + activity_srh + bmi_srh
        + sleep_q_srh + smoking_srh + stress_srh + health_srh
        + rng.normal(0, 8, size=n),
        0, 100
    )
    # Global quantile thresholds so the overall population hits a realistic skew
    # (Poor 8%, Fair 17%, Good 40%, Very Good 25%, Excellent 10% -- "good or better"
    # ~75% overall, close to the ~72% population-weighted average of the four
    # country targets). Country/Gender/etc. offsets are already baked into
    # srh_score, so per-country "good+" rates will still differ around this.
    srh_thresholds = np.quantile(srh_score, [0.08, 0.25, 0.65, 0.90])
    df['SelfRatedHealth'] = bucket(srh_score, srh_thresholds, SRH_ORDER)

    return df


def age_missingness_multiplier(age: np.ndarray) -> np.ndarray:
    """Older people are less likely to own/use a wearable or app that auto-logs
    sleep, resting heart rate, or stress -- so those device/self-report fields
    should go missing more often as age increases. Age bands chosen to match
    the bands the demo notebook slices missingness by."""
    return np.select(
        [age < 30, age < 45, age < 60],
        [0.5, 0.9, 1.4],
        default=2.1,
    )


def inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()

    # Device/self-report fields: missingness rate scales with age (see
    # age_missingness_multiplier). Age itself is still fully populated at this
    # point (anomalies/corruption below haven't run yet), so this is a clean signal.
    age_mult = age_missingness_multiplier(df['Age'].to_numpy())
    device_fields = [
        ('Avg Sleep Hours Per Night', 0.07),
        ('Avg Resting Heart Rate', 0.06),
        ('Stress Level', 0.05),
    ]
    for col, base_rate in device_fields:
        rate = np.clip(base_rate * age_mult, 0, 0.35)
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan

    # Health Issues: not device-sourced (more like a survey/clinical field), so
    # this stays a flat rate rather than age-differential.
    mask = rng.random(len(df)) < 0.10
    df.loc[mask, 'Health Issues'] = np.nan

    # Row-level incomplete records (5-10 rows, 4+ missing fields each)
    immune = {'ID', 'Country', 'SelfRatedHealth'}
    mutable_cols = [c for c in df.columns if c not in immune]
    n_incomplete_rows = rng.integers(5, 11)
    incomplete_row_idx = rng.choice(df.index, size=n_incomplete_rows, replace=False)
    for row_idx in incomplete_row_idx:
        n_missing = rng.integers(4, 7)
        cols_to_null = rng.choice(mutable_cols, size=n_missing, replace=False)
        df.loc[row_idx, cols_to_null] = np.nan

    # Age typo-style anomalies (~0.7%): a digit gets doubled during entry
    # (e.g. 34 -> 344). No collapsing cap here -- for our 18-75 age range the
    # doubled value never exceeds 755, so stripping the trailing digit always
    # recovers the exact original age. A student can reason "344 -> drop the
    # last digit -> 34" and actually be right, rather than everything collapsing
    # to one sentinel value.
    anomaly_mask = rng.random(len(df)) < 0.007
    ages = df.loc[anomaly_mask, 'Age']
    df.loc[anomaly_mask, 'Age'] = ages.apply(
        lambda a: min(999, int(f'{int(a)}{str(int(a))[-1]}')) if pd.notna(a) else a
    )

    # BMI typo-style anomalies (~0.6%): a missed/misplaced decimal point during
    # entry (e.g. 24.5 -> 245). Same "reasoned, not sentinel" property as Age:
    # dividing by 10 recovers the original value exactly.
    bmi_anomaly_mask = rng.random(len(df)) < 0.006
    df.loc[bmi_anomaly_mask, 'BMI'] = (df.loc[bmi_anomaly_mask, 'BMI'] * 10).round(1)

    # Duplicate rows (~0.4%)
    n_dupes = int(round(len(df) * 0.004))
    if n_dupes > 0:
        dupe_rows = df.sample(n=n_dupes, random_state=rng.integers(0, 2**31 - 1))
        df = pd.concat([df, dupe_rows], ignore_index=True)

    return df


def main():
    parser = argparse.ArgumentParser(description='Generate the coffee & health dataset')
    parser.add_argument('--rows', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='generated_coffee_health_dataset.csv')
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    df = inject_data_quality_issues(df, rng)

    df.to_csv(args.output, index=False)
    print(f'Generated {len(df)} rows -> {args.output}')


if __name__ == '__main__':
    main()
