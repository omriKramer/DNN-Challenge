import itertools
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class GlucoseData(Dataset):
    CATEGORICAL = ['food_id', 'meal_type', 'unit_id']

    def __init__(self, cgm_df, meals_df, y):
        self.cgm_df = cgm_df
        self.meals = meals_df.sort_index()
        self.y = y

    def __len__(self):
        return len(self.cgm_df)

    def __getitem__(self, item):
        cgm = self.cgm_df.iloc[item]
        name, t = cgm.name[1:]
        meals = self.meals.loc[(name, t-timedelta(hours=12)): (name, t)]
        base = t.minute % 15
        index = pd.date_range(end=t, periods=48, freq='15T')

        dummies = pd.get_dummies(meals[self.CATEGORICAL], dummy_na=True)
        cat = dummies.resample('15T', level='Date',  base=base, label='right', closed='right').sum()
        cat = cat.reindex(index, fill_value=0)

        cont = meals.resample('15T', level='Date',  base=base, label='right', closed='right').sum()
        cont = cont.reindex(index, fill_value=0)

        target = self.y.iloc[item]
        return (cgm, cont, cat), target
