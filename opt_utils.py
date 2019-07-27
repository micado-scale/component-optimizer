import os

from ruamel import yaml
import csv
import zipfile
import pandas as pd
import numpy as np

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.externals import joblib

import logging

logger = logging.getLogger('optimizer')

pandas_dataframe_styles = {
    'font-family': 'monospace',
    'white-space': 'pre'
}


def readCSV(filename):
    df = pd.read_csv(filename, sep=",", header="infer", skiprows=0, na_values="null")

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
    if( df.columns.contains('Time') ):
        df = dropVariable(df, 'Time')
        logger.info('Time column dropped from data frame')
        
    if( df.columns.contains('timestamp') ):
        df = dropVariable(df, 'timestamp')
        logger.info('timestamp column dropped from data frame')
        
    if( df.columns.contains('avg latency (quantile 0.9)') ):
        df = dropVariable(df, 'avg latency (quantile 0.9)')
        logger.info('avg latency (quantile 0.9) column dropped from data frame')
    # Remove cases with missing values
    df = removeMissingData(df)
    return df


def renameVariable(df, old_var_name, new_var_name):
    new_df = df.copy()
    if( df.columns.contains(old_var_name) ):
        new_df.rename(columns={old_var_name: new_var_name}, inplace=True)
    else:
        logger.info('--------------------- Wrong Column Name ---------------------')
    return new_df


def dropFirstCases(df, n):
    new_df = df.copy()
    filteredDF = new_df[new_df.index > n]
    return filteredDF





def read_yaml(yaml_file):
    global logger
    with open(yaml_file, 'r') as stream:
        try:
            yaml_data = yaml.safe_load(stream)
        except (FileNotFoundError, IOError, yaml.YAMLError) as e:
            logger.error(e)
        else:
            return yaml_data

        
def create_dirs(directories):
    for dir in directories: 
        if not os.path.exists(dir):
            try:
                os.makedirs(dir)
            except IOError as e:
                logger.error(e)
        else:
            logger.info(f'Directory {dir} exists')


def write_yaml(yaml_file, data):
    global logger
    with open(yaml_file, 'w') as stream:
        try:
            yaml.dump(data, stream, default_flow_style=False)
        except (IOError, yaml.YAMLError) as e:
            logger.error(e)

#TODO refactor
def read_data(filename, skip_header=False):
    global logger
    with open(filename, 'r') as csv_file:
        try:
            reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')
        except (FileNotFoundError, IOError, csv.Error) as e:
            logger.error(e)
        else:
            if skip_header:
                next(reader, None)
            return list(reader)


def persist_data(filename, data, mode):
    global logger
    try:
        with open(filename, mode) as stream:
            # wr = csv.writer(stream, quoting=csv.QUOTE_NONNUMERIC)
            wr = csv.writer(stream)
            if isinstance(data[0], list):
                for line in data: 
                    wr.writerow(line)
            else:
                wr.writerow(data)
    except (FileNotFoundError, IOError) as e:
        logger.error(e)

def zip_files(files, zip_filename):
    global logger
    try:
        zipf = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)
    except IOError as e:
        logger.error(e)
    else:
        try:
            for file in files:
                zipf.write(file)
        except (FileNotFoundError, IOError) as e:
            logger.error(e)
        finally:
            zipf.close()