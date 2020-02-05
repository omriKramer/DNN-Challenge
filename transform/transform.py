import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category='FutureWarning')


def combine_cgm_meals(sample):
    cgm = sample['cgm']
    meals = sample['meals']
    target = sample['target']

    df1 = cgm.set_index('Date')
    df2 = meals.set_index('Date')
    df1['Mark'] = 0
    df2['Mark'] = 1
    input_tens = pd.concat([df1, df2], axis=0, sort=True)
    input_tens = input_tens.fillna(0)
    input_tens = input_tens.sort_index()
    input_tens = input_tens.iloc[1:, ].values
    try:
        input_tens = torch.tensor(input_tens, dtype=torch.float64)
    except e:
        a = 1
    input_tens = F.pad(input_tens, pad=(0, 0, 0, 70 - input_tens.shape[0]))
    target = torch.tensor(target, dtype=torch.float64)
    sample = {'input_tens': input_tens, 'target': target}
    return sample



def cat2int(series):
    return series.cat.codes + 1


class ToTensor:
    def __init__(self, categorical):
        self.categorical = list(categorical)

    def __call__(self, sample):
        cgm = sample['cgm'].copy().drop(columns='id')
        meals = sample['meals'].copy().drop(columns='id')

        cgm['Date'] = normalize_time(cgm['Date'])
        meals['Date'] = normalize_time(meals['Date'])
        categorical = meals[self.categorical].transform(cat2int)
        if len(categorical):
            categorical = torch.tensor(categorical.values, dtype=torch.long)
        else:
            categorical = torch.empty(0, 3, dtype=torch.long)

        new_sample = {
            'cgm': torch.tensor(cgm.values.T, dtype=torch.float),
            'meals_cont': torch.tensor(meals.drop(columns=self.categorical).values, dtype=torch.float),
            'meals_cat': categorical,
            'target': torch.tensor(sample['target'], dtype=torch.float)
        }
        return new_sample
