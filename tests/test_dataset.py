from datasets import GlucoseData


def test_datapoints_length(glucose_data):
    past_instances = 4 * 12 + 1
    future_instances = 8
    for i in [0, 1404, -1]:
        sample = glucose_data[i]
        assert len(sample['cgm']) == past_instances
        assert len(sample['target']) == future_instances, f'length of target {i} was {len(sample["target"])}'


def test_split_by_individuals(cgm_file, meals_file):
    train, val = GlucoseData.train_val_split(cgm_file, meals_file)
    assert train
    assert val
