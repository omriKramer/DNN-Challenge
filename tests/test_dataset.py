import pytest

from datasets import GlucoseData


@pytest.fixture(scope='module')
def glucose_data(request):
    glucose_file = request.config.getoption('--cgm')
    meals_file = request.config.getoption('--meals')
    return GlucoseData(glucose_file, meals_file)


def test_datapoints_length(glucose_data):
    past_instances = 4 * 12 + 1
    future_instances = 8
    for i in [0, 1404, -1]:
        sample = glucose_data[i]
        assert len(sample['cgm']) == past_instances
        assert len(sample['target']) == future_instances, f'length of target {i} was {len(sample["target"])}'
