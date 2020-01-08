import argparse

import torch


def get_files():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cgm', default='../data/GlucoseValues.csv', help='location of cgm data file')
    parser.add_argument('--meals', default='../data/Meals.csv', help='location of meals data file')
    args = parser.parse_args()
    return args.cgm, args.meals


def collate(samples):
    batched = {
        'cgm': torch.stack([s['cgm'] for s in samples]),
        'meals': [s['meals'] for s in samples],
        'target': torch.stack([s['target'] for s in samples])
    }
    return batched
