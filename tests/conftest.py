import pytest
from datasets import GlucoseData


def pytest_addoption(parser):
    parser.addoption(
        "--cgm", default="../data/GlucoseValues.csv", help="path to glucose data file"
    )
    parser.addoption(
        "--meals", default="../data/Meals.csv", help="path to meals data file"
    )


@pytest.fixture(scope='session')
def cgm_file(request):
    return request.config.getoption('--cgm')


@pytest.fixture(scope='session')
def meals_file(request):
    return request.config.getoption('--meals')


@pytest.fixture(scope='session')
def glucose_data(cgm_file, meals_file):
    return GlucoseData.from_files(cgm_file, meals_file)
