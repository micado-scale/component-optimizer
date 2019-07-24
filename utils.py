from sklearn.externals import joblib
import pandas as pd
import numpy as np

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

pandas_dataframe_styles = {
    'font-family': 'monospace',
    'white-space': 'pre'
}

def readCSV(filename):
    df = pd.read_csv(filename, sep=";", header="infer", skiprows=0, na_values="null" )

    # Return DataFrame
    return df

def removeMissingData(df):
    cleanDF = df.dropna(axis=0)
    return cleanDF


def dropVariable(df, column):
    del df[column]
    return df


def preProcessing(df):
    df = df.copy()
    
    # Drop Time
    df = dropVariable(df, 'Time')
    df = dropVariable(df, 'avg latency (quantile 0.9)')

    # Debug
    # printDF(df)

    # Remove cases with missing values
    df = removeMissingData(df)
    return df


def renameVariable(df, old_var_name, new_var_name):
    new_df = df.copy()
    new_df.rename(columns={old_var_name: new_var_name}, inplace=True)
    return new_df


def setMetricNames(names):
    new_metricNames = names.copy()
    return new_metricNames


def setExtendedMetricNames(names):
    new_extendedMetricNames = names.copy()
    return new_extendedMetricNames


def dropFirstCases(df, n):
    new_df = df.copy()
    filteredDF = new_df[new_df.index > n]
    return filteredDF


def loadMinMaxScalerXFull():
    X_normalized_MinMaxScaler = joblib.load('models/scaler_normalizeX.save')
    
    return X_normalized_MinMaxScaler


def loadMinMaxScalerYFull():
    y_normalized_MinMaxScaler = joblib.load('models/scaler_normalizeY.save')
    
    return y_normalized_MinMaxScaler

def loadNeuralNetworkModel():
    modelNeuralNet = joblib.load('models/saved_mlp_model.pkl')
    
    return modelNeuralNet


def compareTwoVariables(df1, df2, variableName):
    '''Compare Two Pandas DataFrame by Descripted Statistics. I use this to compare train test data, where columns are the same'''
    describeX1 = df1[variableName].describe(percentiles=[]).to_frame()
    describeX2  = df2[variableName].describe(percentiles=[]).to_frame()
    columnNamesX2 = [x + 'T' for x in describeX2.columns]
    describeX2.columns = columnNamesX2
    # compareX1_X2 = pd.concat([describeX1, describeX2], axis=1, sort=False)
    compareX1_X2 = pd.concat([describeX1, describeX2], axis=1)
    compareX1_X2.sort_index(axis=1, inplace=True)

    compareX1_X2['Difference in percent'] = (compareX1_X2[variableName] - compareX1_X2[variableName + 'T'])/(compareX1_X2[variableName])*100

    return compareX1_X2.style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


def printInfoTrainTestSet(y_train, y_test):
    '''Expected pandas.dataframes'''
    
    print('y_train.min()             = ', y_train.min())
    print('y_test.min()              = ', y_test.min())
    print('------------------------------------------')
    print('type(y_train)             = ', type(y_train))
    print('type(y_test)              = ', type(y_test))
    print('------------------------------------------')
    print('type(y_train.values)      = ', type(y_train.values))
    print('type(y_test.values)       = ', type(y_test.values))
    print('------------------------------------------')
    print('y_train.values.min()      = ', y_train.values.min())
    print('y_test.values.min()       = ', y_test.values.min())
    print('------------------------------------------')
    print('y_train.values.max()      = ', y_train.values.max())
    print('y_test.values.max()       = ', y_test.values.max())
    print('------------------------------------------')
    print('y_train.values.argmin()   = ', y_train.values.argmin())
    print('y_test.values.argmin()    = ', y_test.values.argmin())
    print('------------------------------------------')
    print('y_train.values.argmax()   = ', y_train.values.argmax())
    print('y_test.values.argmax()    = ', y_test.values.argmax())
    print('------------------------------------------')
    print('y_train.idxmin()          = ', y_train.idxmin())
    print('y_test.idxmin()           = ', y_test.idxmin())
    print('------------------------------------------')
    print('y_train.size              = ', y_train.size)
    print('y_test.size               = ', y_test.size)
    print('------------------------------------------')
    print('y_train.first_valid_index()   = ', y_train.first_valid_index())
    print('y_test.first_valid_index()    = ', y_test.first_valid_index())
    print('------------------------------------------')
    print('y_train.last_valid_index()    = ', y_train.last_valid_index())
    print('y_test.last_valid_index()     = ', y_test.last_valid_index())
    print('------------------------------------------')
    print('y_train.nsmallest(2)          =\r', y_train.nsmallest(2))
    print('y_test.nsmallest(2)           =\r', y_test.nsmallest(2))
    print('------------------------------------------')
    print('y_train.name            = ', y_train.name)
    print('y_test.name             = ', y_test.name)
    print('------------------------------------------')
    print('y_train.keys()          =\r', y_train.keys())
    print('y_test.keys()           =\r', y_test.keys())
    print('------------------------------------------')

    pass

def printInfoNumpyArrays(y_train, y_test):
    '''
    Print out infromation about two numpy.array
    
    :param numpy.array y_train: one numpy.array
    :param numpy.array y_test: othoer numpy.array
    :return: pass
    :rtype: none
    :raises TypeError: if the y_train or y_test is not numpy.array
    '''
    
    print('type(y_train)             = ', type(y_train))
    print('type(y_test)              = ', type(y_test))
    print('------------------------------------------')
    print('len(y_train)              = ', len(y_train))
    print('len(y_test)               = ', len(y_test))
    print('------------------------------------------')
    print('y_train.size              = ', y_train.size)
    print('y_test.size               = ', y_test.size)
    print('------------------------------------------')
    print('y_train.shape             = ', y_train.shape)
    print('y_test.shape              = ', y_test.shape)
    print('------------------------------------------')
    print('y_train.min               = ', y_train.min())
    print('y_test.min                = ', y_test.min())
    print('------------------------------------------')
    print('y_train.max               = ', y_train.max())
    print('y_test.max                = ', y_test.max())

    pass


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