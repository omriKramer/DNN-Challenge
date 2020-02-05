import pickle
from pathlib import Path

import pandas as pd

CATEGORICAL = 'food_id', 'meal_type', 'unit_id'


def main(data_dir):
    data_dir = Path(data_dir)
    glucose_df = pd.read_csv(data_dir / 'GlucoseValues.csv', parse_dates=['Date'])
    meals_df = pd.read_csv(data_dir / 'Meals.csv', parse_dates=['Date'])

    norm_stats = {col_name: (meals_df[col_name].mean(), meals_df[col_name].std())
                  for col_name in meals_df.columns
                  if col_name not in CATEGORICAL + ('id', 'Date')}
    norm_stats['GlucoseValue'] = glucose_df['GlucoseValue'].mean(), glucose_df['GlucoseValue'].std()
    print(norm_stats)
    with (data_dir / 'norm_stats.pickle').open('wb') as f:
        pickle.dump(norm_stats, f)

    categories = {col_name: list(meals_df[col_name].astype('category').cat.categories) for col_name in CATEGORICAL}
    print(categories)
    with (data_dir / 'categories.pickle').open('wb') as f:
        pickle.dump(categories, f)


if __name__ == '__main__':
    main('../data/')
