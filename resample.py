from pathlib import Path

import pandas as pd

from pre import preprocess


def resample_meals(cgm: pd.DataFrame, meals: pd.DataFrame, freq: int) -> pd.DataFrame:
    resampled, ids = [], []
    for name, group in meals.groupby('id'):
        base = cgm.loc[name, 'GlucoseValue'].index.minute.min()
        group = group.resample(f'{freq}T', level='Date', base=base, closed='right', label='right').sum()
        resampled.append(group.reindex(cgm.loc[name].index, fill_value=0.))
        ids.append(name)
    resampled = pd.concat(resampled, keys=ids, names=['id'])
    return resampled


def main(args):
    p = Path(args.path)
    meals = pd.read_csv(p / 'Meals.csv', index_col=[0, 1], parse_dates=['Date']).sort_index()
    cgm = pd.read_csv(p / 'GlucoseValues.csv', index_col=[0, 1], parse_dates=['Date']).sort_index()
    preprocess(cgm, meals)
    resampled = resample_meals(cgm, meals, args.freq)
    resampled.to_csv(p / f'resampled_{args.freq}.csv')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--freq', '-f', default=15)
    main(parser.parse_args())