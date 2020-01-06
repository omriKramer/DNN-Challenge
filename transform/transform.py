import pandas as pd
import torch


def combine_cgm_meals(sample):
    cgm = sample['cgm']
    meals = sample['meals']
    target = sample['target']

    meals.loc[meals.meal_type == 'meal', 'meal_type'] = 0
    meals.loc[meals.meal_type == 'snack', 'meal_type'] = 1
    meals.loc[meals.meal_type == 'drink', 'meal_type'] = 2

    df1 = cgm.set_index('Date')
    df2 = meals.set_index('Date')
    df1['Mark'] = 0
    df2['Mark'] = 1
    input_tens = pd.concat([df1, df2], axis=0, sort=True)
    input_tens = input_tens.fillna(0)
    input_tens = input_tens.iloc[1:, ].values
    input_tens = torch.tensor(input_tens, dtype=torch.float64)
    target = torch.tensor(target, dtype=torch.float64)
    sample = {'input_tens': input_tens, 'target': target}
    return sample


def normalize_time(series):
    # 1440 minutes in a day
    normalized = (series.dt.hour * 60 + series.dt.minute) / 1440
    return normalized


def to_tensor(sample):
    cgm = sample['cgm'].copy()
    meals = sample['meals'].copy()

    cgm['Date'] = normalize_time(cgm['Date'])
    meals['Date'] = normalize_time(meals['Date'])

    new_sample = {
        'cgm': cgm.drop(columns='id').values.T,
        'meals': meals.drop(columns='id').values,
        'target': sample['target']
    }

    new_sample = {k: torch.tensor(v, dtype=torch.float) for k, v in new_sample.items()}
    return new_sample
