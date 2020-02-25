import argparse

import fastai
import fastprogress
from fastprogress.fastprogress import force_console_behavior

master_bar, progress_bar = fastai.core.master_bar, fastai.core.progress_bar


def get_files():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cgm', default='data/GlucoseValues.csv', help='location of cgm data file')
    parser.add_argument('--meals', default='data/Meals.csv', help='location of meals data file')
    args = parser.parse_args()
    return args.cgm, args.meals


class ProgressBarCtx:
    """Context manager to disable the progress update bar."""

    def __init__(self, learn, show=True):
        self.learn = learn
        self.show = show

    def __enter__(self):
        if not self.show:
            # silence progress bar
            fastprogress.fastprogress.NO_BAR = True
            fastai.basic_train.master_bar, fastai.basic_train.progress_bar = force_console_behavior()
        return self.learn

    def __exit__(self, *args):
        fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar
