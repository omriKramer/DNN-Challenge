import argparse


def get_files():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cgm', default='../data/GlucoseValues.csv' ,help='location of cgm data file')
    parser.add_argument('--meals', default='../data/Meals.csv', help='location of meals data file')
    args = parser.parse_args()
    return args.cgm, args.meals
