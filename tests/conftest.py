def pytest_addoption(parser):
    parser.addoption(
        "--cgm", default="data/GlucoseValues.csv", help="path to glucose data file"
    )
    parser.addoption(
        "--meals", default="data/Meals.csv", help="path to meals data file"
    )
