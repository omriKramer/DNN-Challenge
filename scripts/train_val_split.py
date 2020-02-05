from pathlib import Path

import numpy as np
import pandas as pd


def filter_no_meals_data(cgm_df, meals_df):
    removal_patients = np.setdiff1d(cgm_df['id'].unique(), meals_df['id'].unique(), assume_unique=True)
    cgm_df = cgm_df[~cgm_df['id'].isin(removal_patients)]
    return cgm_df


def split_by_individuals(cgm_df, ratio):
    individuals = cgm_df['id'].unique()
    n_train = round(len(individuals) * ratio)
    train_indiv = np.random.choice(individuals, n_train, replace=False)
    val_indiv = np.setdiff1d(individuals, train_indiv)
    assert len(train_indiv) + len(val_indiv) == len(individuals)
    return train_indiv, val_indiv


def create_set(data_dir: Path, glucose_df, meals_df, indiv):
    data_dir.mkdir(exist_ok=True)
    glucose_df = glucose_df[glucose_df['id'].isin(indiv)]
    meals_df = meals_df[meals_df['id'].isin(indiv)]
    glucose_df.to_csv(data_dir / 'GlucoseValues.csv', index=False)
    meals_df.to_csv(data_dir / 'Meals.csv', index=False)


def main(data_dir, ratio):
    data_dir = Path(data_dir)
    glucose_df = pd.read_csv(data_dir / 'GlucoseValues.csv', parse_dates=['Date'])
    meals_df = pd.read_csv(data_dir / 'Meals.csv', parse_dates=['Date'])
    glucose_df = filter_no_meals_data(glucose_df, meals_df)
    train_indiv, val_indiv = split_by_individuals(glucose_df, ratio)
    create_set(data_dir / 'train', glucose_df, meals_df, train_indiv)
    create_set(data_dir / 'val', glucose_df, meals_df, val_indiv)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, default=0.8)
    args = parser.parse_args()
    main('../data/', args.ratio)
