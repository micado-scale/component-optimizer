import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


def evaluateNeuralNetwork(y_test, y_predicted):
    mae = mean_absolute_error(y_test, y_predicted)
    r2  = r2_score(y_test, y_predicted)
    mse = mean_squared_error(y_test, y_predicted)

    return mae, r2, mse


def evaluateGoodnessOfPrediction(y_normalized, y_predicted):
    """Measure the correlation between the rean and the predicted values"""
    
    correlation = np.corrcoef(y_normalized, y_predicted)[0,1]
    explained_variance = explained_variance_score(y_normalized, y_predicted)
    mae = mean_absolute_error(y_normalized, y_predicted)
    mse = mean_squared_error(y_normalized, y_predicted)
    r2  = r2_score(y_normalized, y_predicted)

    print('Correlation           = {v:0.3f}'.format(v = correlation))
    print('Explained variance    = {v:0.3f}'.format(v = explained_variance))
    print('Mean Absolute Error   = {v:0.3f}'.format(v = mae))
    print('Mean Squared Error    = {v:0.3f}'.format(v = mse))
    print('R2 Score              = {v:0.3f}'.format(v = r2))
    