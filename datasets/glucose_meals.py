import itertools
from datetime import timedelta

import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset


def split_by_individuals(cgm_file, meals_file, ratio=0.8):
    cgm_df = pd.read_csv(cgm_file, parse_dates=['Date'])
    meals_df = pd.read_csv(meals_file, parse_dates=['Date'])
    cgm_df = filter_no_meals_data(cgm_df, meals_df)

    individuals = cgm_df['id'].unique()
    num_indiv = int(len(individuals) * ratio)
    train_indiv = individuals[:num_indiv]
    val_indiv = individuals[num_indiv:]

    cgm_train = cgm_df[cgm_df['id'].isin(train_indiv)]
    meals_train = meals_df[meals_df['id'].isin(train_indiv)]

    cgm_val = cgm_df[cgm_df['id'].isin(val_indiv)]
    meals_val = meals_df[meals_df['id'].isin(val_indiv)]

    return (cgm_train, meals_train), (cgm_val, meals_val)


def filter_no_meals_data(cgm_df, meals_df):
    removal_patients = np.setdiff1d(cgm_df['id'].unique(), meals_df['id'].unique(), assume_unique=True)
    cgm_df = cgm_df[~cgm_df['id'].isin(removal_patients)]
    return cgm_df


def cumsum_with_restarts(series, reset):
    reset = reset.cumsum()
    result = series.groupby(reset).cumsum()
    return result


def cleanup(cgm, meals):
    cgm = filter_no_meals_data(cgm, meals)
    meals['meal_type'] = meals['meal_type'].astype('category').cat.codes
    meals['unit_id'] = meals['meal_type'].astype('category').cat.codes
    meals = meals.fillna(value=0)
    return cgm, meals


class GlucoseData(Dataset):

    def __init__(self, cgm_df, meals_df, transform=None):
        self.transform = transform

        self.cgm_df, self.meals_df = cleanup(cgm_df, meals_df)

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
        meals_df = pd.read_csv(meals_file, parse_dates=['Date'])
        glucose_data = cls(cgm_df, meals_df, **kwargs)
        return glucose_data

    @classmethod
    def train_val_split(cls, cgm_file, meals_file, **kwargs):
        (cgm_train, meals_train), (cgm_val, meals_val) = split_by_individuals(cgm_file, meals_file)
        train = cls(cgm_train, meals_train, **kwargs)
        val = cls(cgm_val, meals_val, **kwargs)
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
