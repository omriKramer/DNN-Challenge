import itertools
from datetime import timedelta

import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset


def filter_no_meals_data(cgm_df, meals_df):
    removal_patients = np.setdiff1d(cgm_df['id'].unique(), meals_df['id'].unique(), assume_unique=True)
    cgm_df = cgm_df[~cgm_df['id'].isin(removal_patients)]
    return cgm_df


def cumsum_with_restarts(series, reset):
    reset = reset.cumsum()
    result = series.groupby(reset).cumsum()
    return result


class GlucoseData(Dataset):

    def __init__(self, cgm_file, meals_file, cgm_transform=None, meals_transform=None, target_transform=None):
        self.target_transform = target_transform
        self.meals_transform = meals_transform
        self.cgm_transform = cgm_transform

        self.cgm_df = pd.read_csv(cgm_file, parse_dates=['Date'])
        self.meals_df = pd.read_csv(meals_file, parse_dates=['Date'])

        self.cgm_df = filter_no_meals_data(self.cgm_df, self.meals_df)
        indices = []
        for i, individual_cgm in self.cgm_df.groupby('id'):
            individual_cgm = individual_cgm.copy()
            individual_cgm['past_diff'] = individual_cgm['Date'].diff().dt.seconds / 60
            gaps = individual_cgm['past_diff'] > 15
            individual_cgm['past_diff'][gaps] = 0
            individual_cgm['past_info'] = cumsum_with_restarts(individual_cgm['past_diff'], gaps)

            individual_cgm['future_diff'] = (individual_cgm['Date'].diff(-1) * -1).dt.seconds / 60
            gaps = individual_cgm['future_diff'] > 15
            individual_cgm['future_diff'][gaps] = 0
            individual_cgm['future_info'] = cumsum_with_restarts(individual_cgm['future_diff'][::-1], gaps[::-1])

            mask = (individual_cgm['past_info'] >= 12 * 4 * 15) & (individual_cgm['future_info'] >= 2 * 4 * 15)
            eligible = individual_cgm.index[mask]
            indices.append(eligible.to_list())

        self.indices = list(itertools.chain.from_iterable(indices))

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

        if self.cgm_transform:
            cgm = self.cgm_transform(cgm)
        if self.meals_transform:
            meals = self.meals_transform(meals)
        if self.target_transform:
            target = self.target_transform(target)

        sample = {'cgm': cgm, 'meals': meals, 'target': target}
        return sample
