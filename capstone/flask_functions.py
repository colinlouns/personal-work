import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import svm, linear_model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt
import random

def squaring(x):
    return x*x

def data_to_model(player_df, stat_requested, model):

    ss = StandardScaler()

    numeric_columns = ['Age', f'{stat_requested}']

    for column in list(numeric_columns):
        player_df[column] = pd.to_numeric(player_df[column], errors = 'coerce')

    data = player_df[['player_id', 'Age', f'{stat_requested}']]
    data['Age^2'] = squaring(data['Age'])
    data[f'{stat_requested}_1'] = data.groupby(['player_id'])[f'{stat_requested}'].shift(1)
    data[f'{stat_requested}_2'] = data.groupby(['player_id'])[f'{stat_requested}'].shift(2)
    data[f'{stat_requested}_3'] = data.groupby(['player_id'])[f'{stat_requested}'].shift(3)

    data_numbers = data.drop('player_id', axis = 1)
    data_numbers = data_numbers.dropna()

    ss.fit(data_numbers)
    data_numbers_scaled = ss.transform(data_numbers)

    predictors = data_numbers_scaled[-1]
    prediction = model.predict(predictors.reshape(1, -1))

    return prediction
