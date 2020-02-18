import pickle
from pathlib import Path

import pandas as pd

CATEGORICAL = ['food_id', 'meal_type', 'unit_id']
DATA_RESOLUTION_MIN = 15

_data_dir = Path(__file__).parent / 'data'

with (_data_dir / 'norm_stats.pickle').open('rb') as f:
    norm_stats = pickle.load(f)

with (_data_dir / 'categories.pickle').open('rb') as f:
    cat = pickle.load(f)
    cat = {k: pd.api.types.CategoricalDtype(categories=v) for k, v in cat.items()}


def normalize_column(df, col_name):
    with_mean = False
    mean, std = norm_stats[col_name]
    df[col_name] = df[col_name].fillna(mean)
    df[col_name] = ((df[col_name] - mean * with_mean) / std)


def normalize_glucose_meals(cgm, meals):
    normalize_column(cgm, 'GlucoseValue')
    for col_name in meals.columns:
        if col_name not in CATEGORICAL + ['id', 'date']:
            normalize_column(meals, col_name)


def to_cat(meals):
    for col_name in CATEGORICAL:
        meals[col_name] = meals[col_name].astype(cat[col_name])


def preprocess(cgm, meals):
    to_cat(meals)
    normalize_glucose_meals(cgm, meals)


def extract_y(df, n_future_time_points=8):
    """
    Extracting the m next time points (difference from time zero)
    :param n_future_time_points: number of future time points
    :return:
    """
    for g, i in zip(
            range(DATA_RESOLUTION_MIN, DATA_RESOLUTION_MIN * (n_future_time_points + 1), DATA_RESOLUTION_MIN),
            range(1, (n_future_time_points + 1), 1)):
        df['Glucose difference +%0.1dmin' % g] = df.GlucoseValue.shift(-i) - df.GlucoseValue
    return df.dropna(how='any', axis=0).drop('GlucoseValue', axis=1)


def create_shifts(df, n_previous_time_points=48):
    """
    Creating a data frame with columns corresponding to previous time points
    :param df: A pandas data frame
    :param n_previous_time_points: number of previous time points to shift
    :return:
    """
    for g, i in zip(
            range(DATA_RESOLUTION_MIN, DATA_RESOLUTION_MIN * (n_previous_time_points + 1), DATA_RESOLUTION_MIN),
            range(1, (n_previous_time_points + 1), 1)):
        df['GlucoseValue -%0.1dmin' % g] = df.GlucoseValue.shift(i)
    return df.dropna(how='any', axis=0)


def build_cgm(x_glucose):
    X = x_glucose.groupby(level='id').apply(create_shifts)
    y = x_glucose.groupby(level='id').apply(extract_y)
    X = X.loc[y.index].dropna(how='any', axis=0)
    y = y.loc[X.index].dropna(how='any', axis=0)
    return X, y


def get_cgm(path):
    x_glucose = pd.read_csv(path, index_col=[0, 1], parse_dates=['Date']).sort_index()
    normalize_column(x_glucose, 'GlucoseValue')
    X, y = build_cgm(x_glucose)
    return X, y
