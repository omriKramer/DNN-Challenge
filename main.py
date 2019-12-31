import argparse
from torch.utils.data import DataLoader
from datasets import GlucoseData
from transform import combine_cgm_meals


def main(cgm_file, meals_file):
    train, val = GlucoseData.train_val_split(cgm_file, meals_file, transform=combine_cgm_meals)
    train_loader = DataLoader(train, batch_size=8, shuffle=True)
    for sample in train_loader:
        print(sample)
        break


parser = argparse.ArgumentParser()
parser.add_argument('cgm', help='location of cgm data file')
parser.add_argument('meals', help='location of meals data file')
args = parser.parse_args()
main(args.cgm, args.meals)
