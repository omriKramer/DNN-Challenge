import itertools
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class GlucoseData(Dataset):
    CATEGORICAL = ['food_id', 'meal_type', 'unit_id']

    def __init__(self, cgm_df, meals_df, y, resample_rule=15):
        self.cgm_df = cgm_df
        self.meals = meals_df.sort_index()
        self.y = y
        self.resample_rule = resample_rule

    def __len__(self):
        return len(self.cgm_df)

    def __getitem__(self, item):
        cgm = self.cgm_df.iloc[item]
        name, t = cgm.name[1:]
        meals = self.meals.loc[(name, t - timedelta(hours=12)): (name, t)]
        base = t.minute
        if self.resample_rule < 60:
            base = base % self.resample_rule
        rule = f'{self.resample_rule}T'
        index = pd.date_range(end=t, periods=720 // self.resample_rule, freq=rule)

        dummies = pd.get_dummies(meals[self.CATEGORICAL], dummy_na=True)
        cat = dummies.resample(rule, level='Date', base=base, label='right', closed='right').sum()
        cat = cat.reindex(index, fill_value=0) if not cat.empty else pd.DataFrame(0, index, cat.columns)

        sample = {'cgm': cgm}
        start = 0
        for var in self.CATEGORICAL:
            end = start + len(meals[var].cat.categories) + 1
            sample[var] = cat.iloc[:, start:end]
            start = end

        cont = (meals.drop(columns=self.CATEGORICAL)
                .resample(rule, level='Date', base=base, label='right', closed='right')
                .sum())
        sample['cont'] = cont.reindex(index, fill_value=0) if not cont.empty else pd.DataFrame(0, index, cont.columns)
        sample['target'] = self.y.iloc[item]
        sample = {k: torch.tensor(v.values, dtype=torch.float) for k, v in sample.items()}
        return sample
