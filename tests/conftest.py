from pathlib import Path

import pandas as pd
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--train", default="data/train", help="path to train dir"
    )
    parser.addoption(
        "--val", default="data/val", help="path to val dir"
    )


def read_data(data_dir):
    glucose_df = pd.read_csv(data_dir / 'GlucoseValues.csv', parse_dates=['Date'])
    meals_df = pd.read_csv(data_dir / 'Meals.csv', parse_dates=['Date'])
    return glucose_df, meals_df


@pytest.fixture(scope='session')
def train_data(request):
    train_dir = Path(request.config.getoption('--train'))
    return read_data(train_dir)


@pytest.fixture(scope='session')
def val_data(request):
    val_dir = Path(request.config.getoption('--val'))
    return read_data(val_dir)
