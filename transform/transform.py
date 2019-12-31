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
