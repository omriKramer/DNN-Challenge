import pytest


@pytest.mark.parametrize("cat", ['food_id', 'meal_type', 'unit_id'])
def test_split(train_data, val_data, cat):
    _, train_meals = train_data
    _, val_meals = val_data
    assert val_meals[cat].isin(train_meals[cat].unique()).all()
