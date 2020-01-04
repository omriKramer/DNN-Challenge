import random

import torch

import transform


def test_to_tensor(glucose_data):
    inds = random.sample(range(len(glucose_data)), 3)
    for i in inds:
        sample = transform.to_tensor(glucose_data[i])
        assert sample['cgm'].shape == (49, 2)
        assert torch.is_tensor(sample['meals'])
        assert len(sample['target']) == 8
