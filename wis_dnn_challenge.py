####################################################################################################
# wis_dnn_challenge.py
# Description: This is a template file for the WIS DNN challenge submission.
# Important: The only thing you should not change is the signature of the class (Predictor) and its predict function.
#            Anything else is for you to decide how to implement.
#            We provide you with a very basic working version of this class.
#
# Author: <first name1>_<last name1> [<first name1>_<last name2>]
#
# Python 3.7
####################################################################################################
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import models
# The time series that you would get are such that the difference between two rows is 15 minutes.
# This is a global number that we used to prepare the data, so you would need it for different purposes.
from datasets import GlucoseData

DATA_RESOLUTION_MIN = 15


def normalize_time(series):
    # 1440 minutes in a day
    normalized = (series.hour * 60 + series.minute) / 1440
    return normalized


class Predictor(object):
    """
    This is where you should implement your predictor.
    The testing script calls the 'predict' function with the glucose and meals test data which you will need in order to
    build your features for prediction.
    You should implement this function as you wish, just do not change the function's signature (name, parameters).
    The other functions are here just as an example for you to have something to start with, you may implement whatever
    you wish however you see fit.
    """
    CATEGORICAL = 'food_id', 'meal_type', 'unit_id'

    def __init__(self, path2data):
        """
        This constructor only gets the path to a folder where the training data frames are.
        :param path2data: a folder with your training data.
        """
        self.path2data = path2data
        self.train_glucose = None
        self.train_meals = None
        self.nn = models.Linear()
        data_dir = Path(__file__).parent / 'data'
        with (data_dir / 'norm_stats.pickle').open('rb') as f:
            self.norm_stats = pickle.load(f)
        with (data_dir / 'categories.pickle').open('rb') as f:
            cat = pickle.load(f)
            self.cat = {k: pd.api.types.CategoricalDtype(categories=v) for k, v in cat.items()}

    def predict(self, X_glucose, X_meals):
        """
        You must not change the signature of this function!!!
        You are given two data frames: glucose values and meals.
        For every timestamp (t) in X_glucose for which you have at least 12 hours (48 points) of past glucose and two
        hours (8 points) of future glucose, predict the difference in glucose values for the next 8 time stamps
        (t+15, t+30, ..., t+120).

        :param X_glucose: A pandas data frame holding the glucose values in the format you trained on.
        :param X_meals: A pandas data frame holding the meals data in the format you trained on.
        :return: A numpy ndarray, sized (M x 8) holding your predictions for every valid row in X_glucose.
                 M is the number of valid rows in X_glucose (number of time stamps for which you have at least 12 hours
                 of past glucose values and 2 hours of future glucose values.
                 Every row in your final ndarray should correspond to:
                 (glucose[t+15min]-glucose[t], glucose[t+30min]-glucose[t], ..., glucose[t+120min]-glucose[t])
        """

        # build features for set of (ID, timestamp)
        X = self.build_features(X_glucose, X_meals)

        # feed the network you trained (this for example would be a horrible prediction)
        y = pd.concat([X.mean(1) for i in range(8)], axis=1)

        # unnormalize the prediction
        y *= self.norm_stats['GlucoseValue'][1] if self.norm_param else 93.19270255474244
        return y

    def define_nn(self):
        """
        Define your neural network.
        :return: None
        """
        self.nn = None
        return

    def train_nn(self, X_train, y_train, X_val, y_val, epochs=5, lr=1e-2, wd=1e-2):
        """
        Train your network using the training data.
        :param X_train: A pandas data frame holding the features
        :param y_train: A numpy ndarray, sized (M x 8) holding the values you need to predict.
        :return:
        """
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.nn.to(device)
        loss_fn = torch.nn.MSELoss()
        mae_fn = torch.nn.L1Loss(reduction='sum')
        glucose_std = self.norm_stats['GlucoseValue'][1]

        ds_kwargs = {'resample_rule': 60}
        train = GlucoseData(*X_train, y_train, **ds_kwargs)
        train_dl = DataLoader(train, batch_size=128, shuffle=True, num_workers=8)
        val = GlucoseData(*X_val, y_val, **ds_kwargs)
        val_dl = DataLoader(val, batch_size=128, num_workers=8)
        opt = torch.optim.Adam(self.nn.parameters(), lr=lr, weight_decay=wd)
        total_loss = 0.
        print('starting training...')
        for i in range(epochs):
            for sample, target in train_dl:
                sample = {k: v.to(device) for k, v in sample.items()}
                target = target.to(device)
                out = self.nn(sample)
                loss = loss_fn(out, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(target)

            train_loss = total_loss / len(train)

            total_loss, mae = 0., 0.
            with torch.no_grad():
                for sample, target in val_dl:
                    sample = {k: v.to(device) for k, v in sample.items()}
                    target = target.to(device)
                    out = self.nn(sample)
                    loss = loss_fn(out, target)
                    total_loss += loss.item() * len(target)
                    mae += mae_fn(out * glucose_std, target * glucose_std)

            val_loss = total_loss / len(val)
            mae /= len(val)
            msg = (f'{i} train loss: {train_loss:.2}, val loss {val_loss:.2}, '
                   f'MAE: {mae:.2}, Mae/STD: {mae / glucose_std:.2}')
            print(msg)
            self.save_nn_model(i, opt)

    def save_nn_model(self, epoch, opt):
        """
        Save your neural network after training.
        :return:
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.nn.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }, f'checkpoint{epoch:02}.tar')

    def load_nn_model(self):
        """
        Load your trained neural network.
        :return:
        """
        pass

    @staticmethod
    def load_data_frame(path):
        """
        Load a pandas data frame in the relevant format.
        :param path: path to csv.
        :return: the loaded data frame.
        """
        return pd.read_csv(path, index_col=[0, 1], parse_dates=['Date'])

    def load_raw_data(self):
        """
        Loads raw data frames from csv files, and do some basic cleaning
        :return:
        """
        self.train_glucose = Predictor.load_data_frame(os.path.join(self.path2data, 'GlucoseValues.csv'))
        self.train_meals = Predictor.load_data_frame(os.path.join(self.path2data, 'Meals.csv'))

        # suggested procedure
        # 1. handle outliers: trimming, clipping...
        # 2. feature normalizations
        # 3. resample meals data to match glucose values intervals
        return

    def build_features(self, X_glucose, X_meals: pd.DataFrame, build_y=False, n_previous_time_points=48):
        """
        Given glucose and meals data, build the features needed for prediction.
        :param X_glucose: A pandas data frame holding the glucose values.
        :param X_meals: A pandas data frame holding the meals data.
        :param build_y: Whether to also extract the values needed for prediction.
        :param n_previous_time_points:
        :return: The features needed for your prediction, and optionally also the relevant y arrays for training.
        """

        # self.normalize_column(X_glucose, 'GlucoseValue')
        for col_name in X_meals.columns:
            if col_name not in self.CATEGORICAL + ('id', 'date'):
                self.normalize_column(X_meals, col_name)

        # using X_glucose and X_meals to build the features
        # for example just taking the last 2 hours of glucose values
        X = X_glucose.groupby('id').apply(Predictor.create_shifts, n_previous_time_points=n_previous_time_points)

        X['time'] = normalize_time(X.index.get_level_values(2))
        # X_meals['time'] = normalize_time(X_meals.index.get_level_values(1))
        for col_name in self.CATEGORICAL:
            X_meals[col_name] = X_meals[col_name].astype(self.cat[col_name])

        # this implementation of extracting y is a valid one.
        if build_y:
            y = X_glucose.groupby('id').apply(Predictor.extract_y, n_future_time_points=8)
            X = X.loc[y.index].dropna(how='any', axis=0)
            y = y.loc[X.index].dropna(how='any', axis=0)
            return (X, X_meals), y
        return X, X_meals

    def normalize_column(self, df, col_name):
        with_mean = False
        mean, std = self.norm_stats[col_name]
        df[col_name] = df[col_name].fillna(mean)
        df[col_name] = ((df[col_name] - mean * with_mean) / std)

    @staticmethod
    def create_shifts(df, n_previous_time_points=48):
        """
        Creating a data frame with columns corresponding to previous time points
        :param df: A pandas data frame
        :param n_previous_time_points: number of previous time points to shift
        :return:
        """
        for g, i in zip(
                range(DATA_RESOLUTION_MIN, DATA_RESOLUTION_MIN * (n_previous_time_points + 1), DATA_RESOLUTION_MIN),
                range(1, (n_previous_time_points + 1), 1)):
            df['GlucoseValue -%0.1dmin' % g] = df.GlucoseValue.shift(i)
        return df.dropna(how='any', axis=0)

    @staticmethod
    def extract_y(df, n_future_time_points=8):
        """
        Extracting the m next time points (difference from time zero)
        :param n_future_time_points: number of future time points
        :return:
        """
        for g, i in zip(
                range(DATA_RESOLUTION_MIN, DATA_RESOLUTION_MIN * (n_future_time_points + 1), DATA_RESOLUTION_MIN),
                range(1, (n_future_time_points + 1), 1)):
            df['Glucose difference +%0.1dmin' % g] = df.GlucoseValue.shift(-i) - df.GlucoseValue
        return df.dropna(how='any', axis=0).drop('GlucoseValue', axis=1)


def main():
    # example of predict() usage

    # create Predictor instance
    path2data = "data/train"
    path2val = "data/val"
    predictor = Predictor(path2data)

    # load the raw data
    predictor.load_raw_data()

    # build X and y for training
    X, y = predictor.build_features(X_glucose=predictor.train_glucose, X_meals=predictor.train_meals, build_y=True)

    # load validation data
    glucose_val = Predictor.load_data_frame(os.path.join(path2val, 'GlucoseValues.csv'))
    meals_val = Predictor.load_data_frame(os.path.join(path2val, 'Meals.csv'))
    X_val, y_val = predictor.build_features(X_glucose=glucose_val, X_meals=meals_val, build_y=True)

    # train your model (this you need to implement)
    predictor.train_nn(X, y, X_val, y_val)


if __name__ == "__main__":
    main()

    # _, y_gt = predictor.build_features(X_glucose=glucose_val, X_meals=meals_val, build_y=True)
    # y_gt *= predictor.norm_param['glucose'][1]
    # # predict y
    # y_pred = predictor.predict(X_glucose=glucose_val, X_meals=meals_val)
    #
    # # test the prediction (this is the mean absolute error for example)
    # score = np.mean(np.abs(y_pred - y_gt))
    #
    # print("Your score is: {}".format(score))
