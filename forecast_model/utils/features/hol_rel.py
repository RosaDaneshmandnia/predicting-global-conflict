import pandas as pd


RELIGION_TO_HOLIDAY = {
    # Islam
    'islmsunpct':  'islam_holiday_count',   # Sunni
    'islmshipct':  'shia_holiday_count',    # Shia — maps to shia specifically
    'islmibdpct':  'islam_holiday_count',   # Ibadi
    'islmnatpct':  'islam_holiday_count',   # Nation of Islam
    'islmalwpct':  'islam_holiday_count',   # Alawi
    'islmahmpct':  'islam_holiday_count',   # Ahmadiyya
    'islmothrpct': 'islam_holiday_count',   # Other Muslim
    'islmgenpct':  'islam_holiday_count',   # General Muslim

    # Christianity
    'chrstprotpct': 'christian_holiday_count',  # Protestant
    'chrstcatpct':  'christian_holiday_count',  # Catholic
    'chrstorthpct': 'christian_holiday_count',  # Orthodox
    'chrstangpct':  'christian_holiday_count',  # Anglican
    'chrstothrpct': 'christian_holiday_count',  # Other Christian
    'chrstgenpct':  'christian_holiday_count',  # General Christian

    # Judaism
    'judorthpct':  'jewish_holiday_count',
    'judconspct':  'jewish_holiday_count',
    'judrefpct':   'jewish_holiday_count',
    'judothrpct':  'jewish_holiday_count',
    'judgenpct':   'jewish_holiday_count',

    # Hinduism
    'hindgenpct':  'hindu_holiday_count',
    'jaingenpct':  'hindu_holiday_count', #shares indian calender

    # Buddhism
    'budmahpct':   'buddhist_holiday_count',  # Mahayana
    'budthrpct':   'buddhist_holiday_count',  # Theravada
    'budothrpct':  'buddhist_holiday_count',  # Other Buddhist
    'budgenpct':   'buddhist_holiday_count',  # General Buddhist


    # Others  map to cultural or nonreligious
    'syncgenpct':  'cultural_holiday_count',
    'anmgenpct':   'cultural_holiday_count',
    'zorogenpct':  'cultural_holiday_count',
    'bahgenpct':   'cultural_holiday_count',
    'taogenpct':   'cultural_holiday_count',
    'confgenpct':  'cultural_holiday_count',
    'shntgenpct':  'cultural_holiday_count',
    'othrgen':     'cultural_holiday_count',
    'othrgenpct':  'cultural_holiday_count',
    'nonreligpct': 'nonreligious_holiday_count',
}

def _get_holiday_col(religion_name, df_columns):
    if pd.isna(religion_name):
        return None
    key = str(religion_name).lower().strip()
    col = RELIGION_TO_HOLIDAY.get(key)
    if col and col in df_columns:
        return col
    return None


def add_holiday_religion_features(df):
    """
    Builds 3 layers of holiday x religion interaction features.

    Relevance-weighted holiday counts:
        Each holiday type weighted by that religion's share in the country.
       for example islam_holiday_count x majority_pct (if majority is islam)

    Minority tension signal:
        Minority religion has a holiday but is NOT the dominant group.
        Proxy for potential mobilization or inter-group friction.

    Total mobilization pressure:
        Single composite score, sum of all weighted holiday signals
        for majority + minority1 + minority2.

    """

    df = df.copy()
    cols = df.columns.tolist()



    def weighted_holidays_vec(religion_col, pct_col, suffix):
        religion_series = df[religion_col].str.lower().str.strip()
        holiday_vals = pd.Series(0.0, index=df.index)
        for religion_name, hol_col in RELIGION_TO_HOLIDAY.items():
            if hol_col not in df.columns:
                continue
            mask = religion_series == religion_name
            holiday_vals[mask] = df.loc[mask, hol_col] * df.loc[mask, pct_col]
        df[f'weighted_holidays_{suffix}'] = holiday_vals

    weighted_holidays_vec('majority_religion',  'majority_pct',  'majority')
    weighted_holidays_vec('minority1_religion', 'minority1_pct', 'minority1')
    weighted_holidays_vec('minority2_religion', 'minority2_pct', 'minority2')


    def minority_tension(minority_religion_col, minority_pct_col, suffix):
        minority_series  = df[minority_religion_col].str.lower().str.strip()
        majority_series  = df['majority_religion'].str.lower().str.strip()
        tension = pd.Series(0.0, index=df.index)

        for religion_name, hol_col in RELIGION_TO_HOLIDAY.items():
            if hol_col not in df.columns:
                continue
            is_minority      = minority_series == religion_name
            is_not_majority  = majority_series != religion_name
            mask             = is_minority & is_not_majority
            tension[mask]    = df.loc[mask, hol_col] * df.loc[mask, minority_pct_col]

        df[f'minority_tension_{suffix}'] = tension

    minority_tension('minority1_religion', 'minority1_pct', 'minority1')
    minority_tension('minority2_religion', 'minority2_pct', 'minority2')

    df['minority_tension_total'] = (
        df['minority_tension_minority1'] + df['minority_tension_minority2']
    )


    df['total_religious_mobilization'] = (
        df['weighted_holidays_majority'] +
        df['weighted_holidays_minority1'] +
        df['weighted_holidays_minority2']
    )

    new_cols = [
        'weighted_holidays_majority',
        'weighted_holidays_minority1',
        'weighted_holidays_minority2',
        'minority_tension_minority1',
        'minority_tension_minority2',
        'minority_tension_total',
        'total_religious_mobilization',
    ]

    lagged = df[new_cols].groupby(level='matched_admin1_id').shift(1)
    lagged.columns = [f'{c} (t-1)' for c in new_cols]
    df = pd.concat([df, lagged], axis=1)

    return df


def get_new_feature_names():

    base = [
        'weighted_holidays_majority',
        'weighted_holidays_minority1',
        'weighted_holidays_minority2',
        'minority_tension_minority1',
        'minority_tension_minority2',
        'minority_tension_total',
        'total_religious_mobilization',
    ]
    lagged = [f'{c} (t-1)' for c in base]
    return base + lagged