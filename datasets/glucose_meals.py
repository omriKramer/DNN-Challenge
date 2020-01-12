import itertools
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


def split_by_individuals(cgm_df, ratio=0.8):
    individuals = cgm_df['id'].unique()
    num_indiv = round(len(individuals) * ratio)
    train_indiv = set(individuals[:num_indiv])
    val_indiv = set(individuals[num_indiv:])

    return train_indiv, val_indiv


def filter_no_meals_data(cgm_df, meals_df):
    removal_patients = np.setdiff1d(cgm_df['id'].unique(), meals_df['id'].unique(), assume_unique=True)
    cgm_df = cgm_df[~cgm_df['id'].isin(removal_patients)]
    return cgm_df


def cumsum_with_restarts(series, reset):
    reset = reset.cumsum()
    result = series.groupby(reset).cumsum()
    return result


def normalize_columns(df, columns):
    stats = df[columns].agg(['mean', 'std']).T
    df = df.fillna(stats['mean'])
    df.loc[:, columns] = df[columns].transform(lambda x: (x - x.mean()) / x.std())
    return df, stats


class GlucoseData(Dataset):
    CATEGORICAL = 'food_id', 'meal_type', 'unit_id'

    def _cleanup(self, cgm, meals_df):
        cgm = filter_no_meals_data(cgm, meals_df)

        cont_features = meals_df.columns.difference(('id', 'Date') + self.CATEGORICAL)
        self.meals_df, self.meals_stats = normalize_columns(meals_df, cont_features)
        self.cgm_df, self.cgm_stats = normalize_columns(cgm, ['GlucoseValue'])

    def __init__(self, cgm_df, meals_df, transform=None):
        self.transform = transform
        self._cleanup(cgm_df, meals_df)

        indices = []
        for i, individual_cgm in self.cgm_df.groupby('id'):
            past_diff = individual_cgm['Date'].diff().dt.seconds / 60
            gaps = past_diff > 15
            past_diff[gaps] = 0
            past_info = cumsum_with_restarts(past_diff, gaps)

            future_diff = (individual_cgm['Date'].diff(-1) * -1).dt.seconds / 60
            gaps = future_diff > 15
            future_diff[gaps] = 0
            future_info = cumsum_with_restarts(future_diff[::-1], gaps[::-1])[::-1]

            mask = (past_info >= 12 * 4 * 15) & (future_info >= 2 * 4 * 15)
            eligible = individual_cgm.index[mask]
            indices.append(eligible.to_list())

        self.indices = list(itertools.chain.from_iterable(indices))

    @classmethod
    def from_files(cls, cgm_file, meals_file, **kwargs):
        cgm_df = pd.read_csv(cgm_file, parse_dates=['Date'])
        dt = dict.fromkeys(cls.CATEGORICAL, 'category')
        meals_df = pd.read_csv(meals_file, parse_dates=['Date'], dtype=dt)
        glucose_data = cls(cgm_df, meals_df, **kwargs)
        return glucose_data

    @classmethod
    def train_val_split(cls, cgm_file, meals_file, **kwargs):
        glucose_data = cls.from_files(cgm_file, meals_file, **kwargs)
        train_indv, val_indv = split_by_individuals(glucose_data.cgm_df)
        train_idx, val_idx = [], []
        for i, indx in enumerate(glucose_data.indices):
            if glucose_data.cgm_df.at[indx, 'id'] in train_indv:
                train_idx.append(i)
            else:
                val_idx.append(i)

        train = torch.utils.data.Subset(glucose_data, train_idx)
        val = torch.utils.data.Subset(glucose_data, val_idx)
        return train, val

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        individual = self.cgm_df.at[index, 'id']
        time_point = self.cgm_df.at[index, 'Date']

        cgm_mask = (
                (self.cgm_df['id'] == individual)
                & (self.cgm_df['Date'] >= time_point - timedelta(hours=12))
                & (self.cgm_df['Date'] <= time_point + timedelta(hours=2))
        )
        cgm = self.cgm_df[cgm_mask]
        target = cgm.iloc[49:]
        target = target['GlucoseValue'].to_numpy() - self.cgm_df.at[index, 'GlucoseValue']
        cgm = cgm.iloc[:49]

        meals_mask = (
                (self.meals_df['id'] == individual)
                & (self.meals_df['Date'] >= time_point - timedelta(hours=12))
                & (self.meals_df['Date'] <= time_point)
        )
        meals = self.meals_df[meals_mask]

        sample = {'cgm': cgm, 'meals': meals, 'target': target}
        if self.transform:
            sample = self.transform(sample)

        return sample
