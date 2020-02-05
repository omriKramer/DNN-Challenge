import itertools
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class GlucoseData(Dataset):
    CATEGORICAL = 'food_id', 'meal_type', 'unit_id'

    def __init__(self, cgm_df, meals_df, y):
        self.cgm_df = cgm_df
        self.meals_df = meals_df
        self.y = y

    def __len__(self):
        return len(self.cgm_df)

    def __getitem__(self, item):
        return sample
