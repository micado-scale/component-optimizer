import logging
import logging.config

import opt_config

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib

np.set_printoptions(precision=3, suppress=True)

pandas_dataframe_styles = {
    'font-family': 'monospace',
    'white-space': 'pre'
}

_target_variable = None
_input_metrics = None
_worker_count = None

def init(target_variable, input_metrics, worker_count):
    global _target_variable
    _target_variable = target_variable[0]
    
    global _input_metrics
    _input_metrics = input_metrics
    
    global _worker_count
    _worker_count = worker_count[0]


def run(nn_file_name, visualize = False):

    logger = logging.getLogger('optimizer')
    
    logger.info('-----opt_trainer.run()-----')
    
    logger.info(f'_target_variable = {_target_variable}')
    logger.info(f'_input_metrics = {_input_metrics}')

    logger.info('---------------------------')
    logger.info(nn_file_name)
    logger.info('---------------------------')
    

    # Declare variables
    inputCSVFile   = 'data/grafana_data_export_long_running_test.csv'
    neuralCSVFile = nn_file_name
    
    # targetVariable = 'avg latency (quantile 0.5)'
    targetVariable = _target_variable
    
    logger.info('---------------------------------------------------------------------------------')
    logger.info(f'targetVariable = {targetVariable}')
    logger.info(f'_target_variable = {_target_variable}')
    logger.info('---------------------------------------------------------------------------------')

    inputMetrics = _input_metrics
    
    logger.info('---------------------------------------------------------------------------------')
    logger.info(f'inputMetrics = {inputMetrics}')
    logger.info(f'_input_metrics = {_input_metrics}')
    logger.info('---------------------------------------------------------------------------------')

    workerCount = _worker_count
    
    logger.info('---------------------------------------------------------------------------------')
    logger.info(f'workerCount = {workerCount}')
    logger.info(f'_worker_count = {_worker_count}')
    logger.info('---------------------------------------------------------------------------------')

    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Don't touch it if you don't know what you do
    # ## ------------------------------------------------------------------------------------------------------
    
    scaler_min = -1                     # 0
    scaler_max = 1                      # 1
    train_test_ratio = 0.3              # 0.3
    activation_function = 'tanh'        # tanh, relu, logistic
    neuronsWhole = 10                   # 10
    neuronsTrainTest = 4                # 4
    cutFirstCases = 10                  # 10

    lead = 1                            # 1 default

    showPlots = False                   # True
    showPlots = visualize               # This value comes as a parameter
    explore = False                     # False
    
    error_msg = 'No error'              # None


    # In[3]: Declare some functions

    def readCSV(filename):
        df = pd.read_csv(filename, sep=";", header="infer", skiprows=1, na_values="null")
        # Return DataFrame
        return df

    # Read DataFrame
    df = readCSV(inputCSVFile)
    
    def readNeuralCSV(filename):
        df = pd.read_csv(filename, sep=",", header="infer", skiprows=0, na_values="null")
        # Return DataFrame
        return df
        
    
    # Read nn_train_data
    nf = readNeuralCSV(neuralCSVFile)
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Here you can switch between the staitc csv and the live csv
    # ## ------------------------------------------------------------------------------------------------------
    
    # Swap df to nf depends on which file will be used
    df = nf

    # ## ------------------------------------------------------------------------------------------------------
    # ## comment out above line if you want ot use static csv file
    # ## ------------------------------------------------------------------------------------------------------

    
    # Declare some functions
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

    
    def dataFrameInfo(df):
        logger.info('-------------DataFrame----------------------')
        logger.info(f'df.columns  = {df.columns}')
        logger.info(f'df.shape    = {df.shape}')
        logger.info(f'df.head()   = {df.head()}')
        logger.info('-------------DataFrame----------------------')


    # ## ------------------------------------------------------------------------------------------------------
    # ## Pre-processing
    # ## ------------------------------------------------------------------------------------------------------
    
    # Preprecess DataFrame
    preProcessedDF = preProcessing(df)   

    # Print DataFrame Info
    dataFrameInfo(preProcessedDF)
    
    # Set targetVariable
    targetVariable = targetVariable
    logger.info(f'target variable set = {targetVariable}')


    # Declare some functions
    def renameVariable(df, old_var_name, new_var_name):
        new_df = df.copy()
        if( df.columns.contains(old_var_name) ):
            new_df.rename(columns={old_var_name: new_var_name}, inplace=True)
        else:
            logger.info('--------------------- Wrong Column Name ---------------------')
        return new_df


    WorkerCountName = None
    if( df.columns.contains('Worker count') ):
        WorkerCountName = 'Worker count'
    elif( df.columns.contains('vm_number') ):
        WorkerCountName = 'vm_number'
    else:
        WorkerCountName = 'Worker count'
        
    logger.info(f'(WorkerCountName = {WorkerCountName}')
    

    # Rename Worker count or vm_number to WorkerCount
    preProcessedDF = renameVariable(preProcessedDF, WorkerCountName, 'WorkerCount')


    # In[10]: Set Metrics Names

    def setMetricNames(names):
        new_metricNames = names.copy()
        return new_metricNames

    metricNames = setMetricNames(['CPU', 'Inter', 'CTXSW', 'KBIn', 'PktIn', 'KBOut', 'PktOut'])
    
    # Ezeeket az értékeket az init-ben adom át neki, annyi a különbség, hogy az első két változóra nincs szükségünk
    metricNames = inputMetrics[2:]
    
    logger.info(f'metricNames = {metricNames}')


    # In[12]: Set Extended Metrics Names

    def setExtendedMetricNames(names):
        new_extendedMetricNames = names.copy()
        return new_extendedMetricNames

    # extendedMetricNames = setExtendedMetricNames(['CPU', 'Inter', 'CTXSW', 'KBIn', 'PktIn', 'KBOut', 'PktOut', 'WorkerCount'])    
    extendedMetricNames = setExtendedMetricNames(metricNames + ['WorkerCount'])
    
    logger.info(f'extendedMetricNames = {extendedMetricNames}')



    # In[14]: Drop First Cases

    def dropFirstCases(df, n):
        new_df = df.copy()
        filteredDF = new_df[new_df.index > n]
        return filteredDF

    # If in the begining of the samples have a lot of outliers
    filteredDF = dropFirstCases(preProcessedDF, cutFirstCases)


    # In[16]: preProcessedDF let filteredDF

    preProcessedDF = filteredDF

    
    # In[17]: Correlation matrix

    from visualizerlinux import CorrelationMatrixSave
    if showPlots : CorrelationMatrixSave(preProcessedDF)


    # In[19]: Visualize the relationship between the target and independent variables

    from visualizerlinux import ScatterPlots
    if showPlots : ScatterPlots(preProcessedDF, preProcessedDF[targetVariable], extendedMetricNames, targetVariable)


    # In[21]: Visualize the relationship between the time and variables

    from visualizerlinux import TimeLinePlot
    if showPlots : TimeLinePlot(preProcessedDF, targetVariable)


    # In[23]: Visualize the relationship between the time and variables

    from visualizerlinux import TimeLinePlots
    if showPlots : TimeLinePlots(preProcessedDF, extendedMetricNames)


    # In[25]: Autoregression

    if( explore ):
        n = 1
        for i in preProcessedDF.columns:
            print('AC(1)      ', i, '\t= ', np.round(preProcessedDF[i].autocorr(lag=1), 2))
            n = n+1
            if( n == 10 ):
                break


    # ## ------------------------------------------------------------------------------------------------------            
    # ## Create a whole new DataFrame for Before After Data
    # ## ------------------------------------------------------------------------------------------------------

    # In[26]:
    
    logger.info('CreateBeforeAfter method')
    logger.debug(preProcessedDF.columns)
    logger.debug(len(inputMetrics))
    logger.debug(inputMetrics)

    def createBeforeafterDF(df, lag, inputMetrics):
        beforeafterDF = df.copy()
        length = len(inputMetrics)
        inputVariables = np.flip(beforeafterDF.columns[0:length].ravel(), axis=-1)
        print('Input Variablels : ', inputVariables)

        index = length
        for i in inputVariables:
            new_column = beforeafterDF[i].shift(lag)
            new_column_name = (i + str(1)) # Todo: rename str(lag)
            beforeafterDF.insert(loc=index, column=new_column_name, value=new_column)

        beforeafterDF = beforeafterDF[lag:]

        print('Before After DF columns: ', beforeafterDF.columns)

        return beforeafterDF


    # In[27]: Create new dataframe with lags

    beforeafterDF = createBeforeafterDF(preProcessedDF, 1, inputMetrics)

    logger.info('CreateBeforeAfter method done')

    # ### Set Features for Neural Network - these are the input variables

    # In[28]: Declare some functions

    def setFeaturesAndTheirLags(df):
        X = df.iloc[:,0:9]
        return X


    # In[29]: Set Features in other words this set will be the Input Variables

    X = setFeaturesAndTheirLags(beforeafterDF)

    logger.info('SetFeaturesAndTheirLags method done')

    # ### Set Target Variable for Neural Network - this is the target variable

    # In[30]: Declare some functions

    def setTarget(df, targetVariable):
        y = df[targetVariable]
        return y


    # In[31]: Set target variable

    y = setTarget(beforeafterDF, targetVariable)

    
    # In[33]:

    if( explore ):
        logger.info(f'(y values from 0 to 10 = {y.values[0:10]}')
        logger.info(f'(y head = {y.head()}')
        logger.info(f'(y describe = {y.describe()}')

    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Normalization
    # ## ------------------------------------------------------------------------------------------------------
    
    # ### Normalize the whole X

    # In[35]: Declare some functions

    def normalizeX(df):
        """Return a normalized value of df.
        Save MinMaxScaler normalizer for X variable"""

        scaler = MinMaxScaler(feature_range=(scaler_min, scaler_max))
        # scaler.fit(df)
        scaler.fit(df.astype(np.float64))
        # normalized = scaler.transform(df)
        normalized = scaler.transform(df.astype(np.float64))

        # store MinMaxScaler for X
        joblib.dump(scaler, 'models/scaler_normalizeX.save') 

        return normalized, scaler


    # In[36]: Normalize Features and Save Normalized values, Normalize input variables set

    X_normalized, X_normalized_MinMaxScaler = normalizeX(X)
    
    logger.info('X_normalized done')


    # ### Load MinMaxScalerXFull

    # In[37]: Declare some functions

    def loadMinMaxScalerXFull():
        X_normalized_MinMaxScaler = joblib.load('models/scaler_normalizeX.save')

        return X_normalized_MinMaxScaler


    # In[38]: Load Saved Normalized Data (Normalizer)

    X_normalized_MinMaxScaler = loadMinMaxScalerXFull()
    
    logger.info('X_normalized_MinMaxScaler load done')


    # In[39]: Declare some functions

    def printNormalizedX(X_normalized):
        print("X_normalized type        = ", type(X_normalized))
        print("X_normalizde dtype       = ", X_normalized.dtype)
        print("X_normalized shape       = ", X_normalized.shape)
        print("X_normalized ndim        = ", X_normalized.ndim)
        print("X_normalized[:,0].max()  = ", X_normalized[:,0].max())
        print("X_normalized[:,0].min()  = ", X_normalized[:,0].min())


    # In[40]:

    if( explore ):
        printNormalizedX(X_normalized)
        X_normalized[1]


    # In[42]: De-normalize Features set

    X_denormalized = X_normalized_MinMaxScaler.inverse_transform(X_normalized)


    # In[43]:

    if( explore ):
        X_denormalized[1]
        X_denormalized[-1]


    # ### Normalize the whole y

    # In[45]: Declare some functions

    def normalizeY(df):
        """Return a normalized value of df.
        Save MinMaxScaler normalizer for Y variable"""

        new_df = df.copy()
        new_df_reshaped = new_df.values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(scaler_min, scaler_max))
        scaler.fit(new_df_reshaped.astype(np.float64))
        normalizedY = scaler.transform(new_df_reshaped.astype(np.float64))
        normalizedY = normalizedY.flatten()

        # store MinMaxScaler for Y
        joblib.dump(scaler, 'models/scaler_normalizeY.save') 

        return normalizedY, scaler


    # In[46]: Normalize Target and Save Normalized values, Normalize target variable set

    y_normalized, y_normalized_MinMaxScaler = normalizeY(y)


    # In[47]: Declare some functions

    def printNormalizedY(y_normalized):
        """Void. Print normalizeY(df) values"""

        print("y_normalized type        = ", type(y_normalized))
        print("y_normalized dtype       = ", y_normalized.dtype)
        print("y_normalized shape       = ", y_normalized.shape)
        print("y_normalized ndim        = ", y_normalized.ndim)
        print("y_normalized[:].max()    = ", y_normalized[:].max())
        print("y_normalized[:].min()    = ", y_normalized[:].min())


    # In[48]:
    
    if( explore ):
        printNormalizedY(y_normalized)
        y_normalized[0:3]


    # ### Load MinMaxScalerYFull

    # In[50]: Declare some functions

    def loadMinMaxScalerYFull():
        y_normalized_MinMaxScaler = joblib.load('models/scaler_normalizeY.save')

        return y_normalized_MinMaxScaler


    # In[51]: Load Saved Normalized Data (Normalizer)

    y_normalized_MinMaxScaler = loadMinMaxScalerYFull()


    # In[52]: De-normalize Features set

    y_denormalized = y_normalized_MinMaxScaler.inverse_transform(y_normalized.reshape(y_normalized.shape[0],1))


    # In[53]:
    
    if( explore ):
        y_denormalized[0:3]
        y_denormalized[-3:]

    logger.info('Normalization done')

    # ## ------------------------------------------------------------------------------------------------------
    # ## Train Neural Network with Optimizer Class, trainMultiLayerRegressor method
    # ## ------------------------------------------------------------------------------------------------------
    
    logger.info('MLP start')

    # In[55]: Declare some functions

    def trainMultiLayerRegressor(X_normalized, y_normalized, activation, neuronsWhole):

        # Train Neural Network
        mlp = MLPRegressor(hidden_layer_sizes=neuronsWhole,
                           max_iter=250,
                           activation=activation,
                           solver="lbfgs",
                           learning_rate="constant",
                           learning_rate_init=0.01,
                           alpha=0.01,
                           verbose=False,
                           momentum=0.9,
                           early_stopping=False,
                           tol=0.00000001,
                           shuffle=False,
                           # n_iter_no_change=20, \
                           random_state=1234)

        mlp.fit(X_normalized, y_normalized)

        # Save model on file system
        joblib.dump(mlp, 'models/saved_mlp_model.pkl')

        return mlp


    # In[56]: Train Neural Network
    
    mlp = trainMultiLayerRegressor(X_normalized, y_normalized, activation_function, neuronsWhole)


    # In[57]: Declare some funcitons

    def predictMultiLayerRegressor(mlp, X_normalized):
        y_predicted = mlp.predict(X_normalized)

        return y_predicted


    # In[58]: Create prediction

    y_predicted = predictMultiLayerRegressor(mlp, X_normalized)


    # In[59]: Evaluete the model

    from utils import evaluateGoodnessOfPrediction
    
    evaluateGoodnessOfPrediction(y_normalized, y_predicted)


    # In[61]: Visualize Data
    
    from visualizerlinux import VisualizePredictedYScatter
    
    if showPlots : VisualizePredictedYScatter(y_normalized, y_predicted, targetVariable)

        
    # In[63]: Visualize Data

    from visualizerlinux import VisualizePredictedYLine, VisualizePredictedYLineWithValues

    if showPlots : VisualizePredictedYLineWithValues(y_normalized, y_predicted, targetVariable, 'Normalized')



    # ### De-normlaize
    # 
    # I want to see the result in original scale. I don't care about the X but the y_normalized and y_predcited.
    # 
    # 

    # In[65]: De-normalize target variable and predicted target variable

    y_denormalized = y_normalized_MinMaxScaler.inverse_transform(y_normalized.reshape(y_normalized.shape[0],1))

    y_predicted_denormalized = y_normalized_MinMaxScaler.inverse_transform(y_predicted.reshape(y_predicted.shape[0],1))


    # In[66]: Visualize the de-normalized data as well

    if showPlots : VisualizePredictedYLineWithValues(y_denormalized, y_predicted_denormalized, targetVariable, 'Denormalized')


    # In[67]: Compare the Original Target Variable and the mean of its Predicted Values

    meanOfOriginalPandasDataframe = y.values.mean()
    meanOfOriginalTargetVariable  = y_denormalized.mean()
    meanOfPredictedTargetVariable = y_predicted_denormalized.mean()

    print('mean original pandas dataframe = ', meanOfOriginalPandasDataframe)
    print('mean original target variable  = ', meanOfOriginalTargetVariable)
    print('mean predicted target variable = ', meanOfPredictedTargetVariable)


    # In[68]: Declare De-normalizer functions

    def denormalizeX(X_normalized, X_normalized_MinMaxScaler):
        X_denormalized = X_normalized_MinMaxScaler.inverse_transform(X_normalized)
        return X_denormalized


    # In[69]: De-normalize Features

    X_denormalized = denormalizeX(X_normalized, X_normalized_MinMaxScaler)


    # In[70]:
    
    if( explore ):
        X_denormalized[1]
        X_normalized[1]
        X_denormalized[-1]
        X_normalized[-1]


    # In[74]: Declare De-normalizer functions

    def denormalizeY(y_normalized, y_normalized_MinMaxScaler):
        y_denormalized = y_normalized_MinMaxScaler.inverse_transform(y_normalized.reshape(y_normalized.shape[0],1))
        return y_denormalized


    # In[75]: De-normalize Target

    y_denormalized = denormalizeY(y_normalized, y_normalized_MinMaxScaler)

    y_predicted_denormalized = denormalizeY(y_predicted, y_normalized_MinMaxScaler)


    # In[76]:

    if showPlots : VisualizePredictedYLineWithValues(y_denormalized, y_predicted_denormalized, targetVariable, 'Denormalized')


    # ## ------------------------------------------------------------------------------------------------------
    # ## Create Train-Test-Validation set
    # ## ------------------------------------------------------------------------------------------------------

    # ## ------------------------------------------------------------------------------------------------------
    # ## Begining of Train Test Experiment
    # ## ------------------------------------------------------------------------------------------------------

    
    # na úgy vagyok vele, hogy ezt az egészet most beteszem egy if-be és kizárom az egészet, mert elvileg ez nem
    # fog kelleni a tanuláshoz csak az exploration és a backtest használja
    
    # if( 0 > 1 ):
        
    # ### Split Data

    # In[77]: Declare splitDataFrame funcition

    def splitDataFrame(X, y, testSize):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=testSize,
                                                            random_state=12345,
                                                            shuffle=False,
                                                            stratify=None)

        return X_train, X_test, y_train, y_test


    # In[78]: Split DataFrame
    X_train, X_test, y_train, y_test = splitDataFrame(X, y, train_test_ratio)


    # In[79]:
    print(X_train.count())
    print(X_test.count())
    print("y_train.count() = ", y_train.count())
    print("y_test.count()  = ", y_test.count())


    # In[82]: Compare Train Test Set by Variables

    from utils import compareTwoVariables

    # TODO:
    # Nem biztos, hogy ezek lesznek a változónevek
    
    print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
    print( inputMetrics[0] )
    print( inputMetrics[1] )
    # compareTwoVariables(X_train, X_test, 'CPU')
    # compareTwoVariables(X_train, X_test, 'CTXSW')
    # compareTwoVariables(X_train, X_test, 'AVG RR')


    # In[86]: Visualize Train Test Set
    if showPlots : VisualizePredictedYLineWithValues(y.values, y_train.values, targetVariable, 'Denormalized')

    if showPlots : VisualizePredictedYLineWithValues(y, y_train, targetVariable, 'Denormalized')

    if showPlots : VisualizePredictedYLineWithValues(0, y_test, targetVariable, 'Denormalized')

    if showPlots : VisualizePredictedYLineWithValues(0, y_test.values, targetVariable, 'Denormalized')

    if showPlots : VisualizePredictedYLineWithValues(y_train, y_train.values, targetVariable, 'Denormalized')


    # In[91]: Compare Train Test Set

    from utils import printInfoTrainTestSet

    printInfoTrainTestSet(y_train, y_test)


    # In[93]:
    if( explore ):
        y_train.values[0:10]
        y_test.values[10:20]


    # ### Normalize Data - based on Train set - It is forbiden to use Test set

    # In[95]: Declare some functions

    def normalizeXTrainTest(X_train, X_test):
        scaler = MinMaxScaler(feature_range=(scaler_min, scaler_max))
        scaler.fit(X_train)
        X_train_normalized = scaler.transform(X_train)
        X_test_normalized = scaler.transform(X_test)

        return X_train_normalized, X_test_normalized, scaler


    # In[96]: Normalize Features - both Train and Test - based on Train set

    X_train_normalized, X_test_normalized, X_normalized_MinMaxScalerTrainTest = normalizeXTrainTest(X_train, X_test)


    # In[97]:
    if( explore ):
        print("X_train_normalized max = ", X_train_normalized.max())
        print("X_train_normalized min = ", X_train_normalized.min())
        print("X_test_normalized max  = ", X_test_normalized.max())
        print("X_test_normalized min  = ", X_test_normalized.min())


    # In[98]: Declare some functions

    def normalizeYTrainTest(y_train, y_test):
        # Create numpy.array from pandas.series then reshape numpy array
        y_train_input = y_train.values.reshape(-1, 1)
        y_test_input = y_test.values.reshape(-1, 1)
        # Scaler
        scaler = MinMaxScaler(feature_range=(scaler_min, scaler_max))
        scaler.fit(y_train_input)
        # Scale
        y_train_normalized_tmp = scaler.transform(y_train_input)
        y_test_normalized_tmp = scaler.transform(y_test_input)
        # Flat numpy.array
        y_train_normalized = y_train_normalized_tmp.flatten()
        y_test_normalized = y_test_normalized_tmp.flatten()

        return y_train_normalized, y_test_normalized, scaler


    # In[99]: Normalize Target set - both Train and Test - based on Train set

    y_train_normalized, y_test_normalized, y_normalized_MinMaxScalerTrainTest = normalizeYTrainTest(y_train, y_test)


    # In[100]:
    if( explore ):
        print("y_train_normalized max = ", y_train_normalized.max())
        print("y_train_normalized min = ", y_train_normalized.min())
        print("y_test_normalized max  = ", y_test_normalized.max())
        print("y_test_normalized min  = ", y_test_normalized.min())


    # In[101]:

    from utils import printInfoNumpyArrays

    printInfoNumpyArrays(y_train_normalized, y_test_normalized)
    printInfoNumpyArrays(y_train, y_test)


    # In[104]: Compare Train and Test Variables one-by-one

    for i in range(0, len(inputMetrics)):
        print(X_test_normalized[:,i].min())
        print(X_test_normalized[:,i].max())
        print('--------------------------')


    # In[105]:

    X_train_denormalized = denormalizeX(X_train_normalized, X_normalized_MinMaxScalerTrainTest)

    X_test_denormalized = denormalizeX(X_test_normalized, X_normalized_MinMaxScalerTrainTest)

    
    y_train_denormalized = denormalizeY(y_train_normalized, y_normalized_MinMaxScalerTrainTest)

    y_test_denormalized = denormalizeY(y_test_normalized, y_normalized_MinMaxScalerTrainTest)


    # In[109]: Visualize 

    if showPlots :

        VisualizePredictedYLineWithValues(y.values, y_train_denormalized, targetVariable, 'Denormalized')

        VisualizePredictedYLineWithValues(y.values[len(y_train_denormalized):], y_test_denormalized, targetVariable, 'Denormalized')


    # In[111]: this is the same Neural Network configuration as I did it before, when whole dataset was trained
    
    logger.info('Split MLP start')

    def trainMultiLayerRegressor(X_train_normalized, y_train_normalized, activation, neuronsTrainTest):

        # Train Neural Network
        mlp = MLPRegressor(hidden_layer_sizes=neuronsTrainTest,
                           max_iter=250,
                           activation=activation,
                           solver="lbfgs",
                           learning_rate="constant",
                           learning_rate_init=0.01,
                           alpha=0.01,
                           verbose=False,
                           momentum=0.9,
                           early_stopping=False,
                           tol=0.00000001,
                           shuffle=False,
                           # n_iter_no_change=200,
                           random_state=1234)

        mlp.fit(X_train_normalized, y_train_normalized)

        # Save model on file system
        joblib.dump(mlp, 'models/saved_mlp_model_train_test.pkl')

        return mlp


    # In[112]: Train Neural Network
    mlp = trainMultiLayerRegressor(X_train_normalized, y_train_normalized, activation_function, neuronsTrainTest)


    # In[113]: Declare some fucntions

    def predictMultiLayerRegressor(mlp, X_normalized):
        y_predicted = mlp.predict(X_normalized)

        return y_predicted


    # In[114]: Create prediction
    y_train_predicted = predictMultiLayerRegressor(mlp, X_train_normalized)

    # Create prediction
    y_test_predicted = predictMultiLayerRegressor(mlp, X_test_normalized)


    # In[115]: Evaluate the model
    
    evaluateGoodnessOfPrediction(y_train_normalized, y_train_predicted)
    print('---------------------')
    evaluateGoodnessOfPrediction(y_test_normalized, y_test_predicted)

    logger.info('Split Evaluation done')

    # In[116]: De-normlaize target variable and predicted target variable

    y_train_denormalized = denormalizeY(y_train_normalized, y_normalized_MinMaxScalerTrainTest)

    y_test_denormalized = denormalizeY(y_test_normalized, y_normalized_MinMaxScalerTrainTest)

    y_train_predicted_denormalized = denormalizeY(y_train_predicted, y_normalized_MinMaxScalerTrainTest)

    y_test_predicted_denormalized = denormalizeY(y_test_predicted, y_normalized_MinMaxScalerTrainTest)


    # In[117]:


    from visualizerlinux import ScatterPlotsTrainTest


    # In[118]:
    if showPlots :
        ScatterPlotsTrainTest(y_train_denormalized, y_train_predicted_denormalized,
                              y_test_denormalized, y_test_predicted_denormalized,
                              targetVariable)


    # In[119]:
    if showPlots :
        ScatterPlotsTrainTest(y_train_normalized, y_train_predicted,
                              y_test_normalized, y_test_predicted,
                              targetVariable)


    # In[120]:
    if showPlots :
        VisualizePredictedYLine(y_train_denormalized, y_train_predicted_denormalized, targetVariable, lines = True)


    # In[121]:
    if showPlots :
        VisualizePredictedYLine(y_train_normalized, y_train_predicted, targetVariable, lines = True)


    # In[122]:
    if showPlots :
        VisualizePredictedYLine(y_test_denormalized, y_test_predicted_denormalized, targetVariable, lines = True)


    # In[123]:
    if showPlots :
        VisualizePredictedYLine(y_test_normalized, y_test_predicted, targetVariable, lines = True)


        
    # ## ------------------------------------------------------------------------------------------------------
    # ## End of Train Test Experiment
    # ## ------------------------------------------------------------------------------------------------------

    # ## ------------------------------------------------------------------------------------------------------
    # ## Linear Regression Learn
    # ## ------------------------------------------------------------------------------------------------------
    
    logger.info('Linear Regression start')
        
    # In[124]: Import dependencies

    from sklearn.linear_model import LinearRegression
    from sklearn import metrics


    # In[125]: Declare some functions

    # TODO:
    # Átvezetni valahogy, hogy a bemeneti változók fényében kezelje hogy hány változó van a dataframeben
    def createBeforeafterDFLags(df, lag):
        beforeafterDFLags = df.copy()
        inputVariables = np.flip(beforeafterDFLags.columns[0:10].ravel(), axis=-1)
        print('Input Variablels : ', inputVariables)

        index = 10
        for i in inputVariables:
            new_column = beforeafterDFLags[i].shift(lag)
            new_column_name = (str('prev') + str(1) + i)
            beforeafterDFLags.insert(loc=index, column=new_column_name, value=new_column)

        beforeafterDFLags = beforeafterDFLags[lag:]             # remove first row as we haven't got data in lag var

        return beforeafterDFLags


    # In[126]: Create lag variables (see above -> 'prev1CPU', 'prev1Inter', etc)

    beforeafterDFLags = createBeforeafterDFLags(preProcessedDF, 1)


    # In[127]:
    if( explore ):
        beforeafterDFLags.columns


    # In[128]: Declare some functions

    def createBeforeafterDFLeads(df, lead = 1):
        beforeafterDFLeads = df.copy()
        inputVariables = np.flip(beforeafterDFLeads.columns[0:10].ravel(), axis=-1)
        print('Input Variablels : ', inputVariables)

        index = 10
        for i in inputVariables:
            new_column = beforeafterDFLeads[i].shift(-lead)
            new_column_name = (str('next') + str(1) + i) # Todo: rename str(lead)
            beforeafterDFLeads.insert(loc=index, column=new_column_name, value=new_column)

        beforeafterDFLeads = beforeafterDFLeads[:-lead]         # remove last row as we haven't got data in lead var

        beforeafterDFLeads = beforeafterDFLeads.iloc[:,:-1]     # remove last column - Latency

        return beforeafterDFLeads


    # In[129]: Create lead variables (see above -> 'next1CPU', 'next1Inter', etc)

    beforeafterDF = createBeforeafterDFLeads(beforeafterDFLags, lead = lead)


    # In[130]:
    if( explore ):
        beforeafterDF.columns


    # In[131]: Assert

    # TODO:
    # Ez is csak akkor stimmel ha a cikkben leírt bemeneti változókat és azok számát használjuk
    a_colName = beforeafterDF.columns[-1]
    a_cols = beforeafterDF.shape[1]

    assert a_colName == 'prev1WorkerCount', "This column name is: {0} insted of prev1WorkerCount".format(a_colName)
    assert a_cols == 30, "This column number is: {0} insted of 17".format(a_colName)

    logger.info('Assert variable number before-after done')

    # In[132]: Declare some functions

    def calculateWorkerCountDifferences(beforeafterDF):
        new_beforeafterDF = beforeafterDF.copy()
        new_beforeafterDF['addedWorkerCount'] = new_beforeafterDF['next1WorkerCount'].values - new_beforeafterDF['WorkerCount']

        return new_beforeafterDF


    # In[133]: Explore data

    theBeforeAfterDF = calculateWorkerCountDifferences(beforeafterDF)


    # In[134]: Declare some functions

    def createScalingDF(theBeforeAfterDF):
        new_beforeafterDF = theBeforeAfterDF.copy()
        scalingDF = new_beforeafterDF[new_beforeafterDF.WorkerCount != new_beforeafterDF.next1WorkerCount]

        return scalingDF


    # In[135]: Collect rows where 'WorkerCount != next1WorkerCount' -> If this condition is True that means scalling happened

    scalingDF = createScalingDF(theBeforeAfterDF)


    # In[136]: Calculate and store values of each metrics after scalling happened in new collumns

    beforeafterMetricsDF = scalingDF.copy()

    for i in metricNames:
        # print(i)
        changeInMetricAfterScale = beforeafterMetricsDF['next1'+i]-beforeafterMetricsDF[i]
        beforeafterMetricsDF['changed1'+i] = changeInMetricAfterScale


    # In[137]: Explore
    if( explore ):
        beforeafterMetricsDF[['prev1CPU','CPU','next1CPU','changed1CPU','prev1WorkerCount','WorkerCount','next1WorkerCount']]. head(10).style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


    # In[138]: Explore
    if( explore ):
        beforeafterMetricsDF[['prev1CPU','CPU','next1CPU','changed1CPU','prev1WorkerCount','WorkerCount','next1WorkerCount']]. head(10).style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


    # In[139]: Explore
    if( explore ):
        beforeafterMetricsDF[['changed1CPU','changed1Inter', 'changed1CTXSW', 'changed1KBIn',
                              'changed1KBOut', 'changed1PktIn', 'changed1PktOut',
                              'addedWorkerCount']]. groupby(['addedWorkerCount'], as_index=False).mean().style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


    # In[140]: Explore
    if( explore ):
        beforeafterMetricsDF[['changed1CPU', 'changed1Inter', 'changed1CTXSW', 'changed1KBIn',
                              'changed1KBOut', 'changed1PktIn', 'changed1PktOut',
                              'addedWorkerCount']].groupby(['addedWorkerCount'], as_index=False).count().style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


    # In[141]: Explore
    if( explore ):
        print(theBeforeAfterDF.shape)
        print(scalingDF.shape)


    # In[143]:Declare some functions
    
    # TODO:
    # Mi az hogy üres a termDF
    
    def calculateLinearRegressionTerms(metric, dataFrame):
        termDF = dataFrame.copy()
        termDF['metric'] = termDF[metric]
        termDF['term1']  = termDF[metric] * termDF['WorkerCount'] / (termDF['WorkerCount'] + termDF['addedWorkerCount'])
        termDF['term2']  = termDF[metric] * termDF['addedWorkerCount'] / (termDF['WorkerCount'] + termDF['addedWorkerCount'])
        return termDF


    def createInputAndTargetToLinearRegression(currentMetric, dataFrameB):
        newDataFrameB = dataFrameB.copy()
        yb = newDataFrameB['next1' + str(currentMetric)]
        featuresDF = newDataFrameB[[str(currentMetric), 'WorkerCount', 'next1WorkerCount', 'addedWorkerCount']]

        tmpDF = calculateLinearRegressionTerms(currentMetric, featuresDF)

        Xb = tmpDF.iloc[:, [-3, -2, -1]]     # keep last three column - given metric, term1, term2

        # These are only for check everything is in order
        # print(y.head(1))
        # print(featuresDF.head(1))
        # print(X.head(2))
        # scalingDF[['CPU', 'next1CPU', 'WorkerCount', 'next1WorkerCount', 'addedWorkerCount']][0:3]
        return Xb, yb


    def calculateLinearRegressionModel(currentMetric, dataFrameA):
        newDataFrameA = dataFrameA.copy()
        Xa, ya = createInputAndTargetToLinearRegression(currentMetric, newDataFrameA)

        # ToDo : Return and store particular model

        lr = LinearRegression(fit_intercept=True, normalize=False)
        lr.fit(Xa, ya)
        # prediction = lr.predict(X)

        return lr


    def calculateLinearRegressionPrediction(metric, dataFrame, model):
        X, y = createInputAndTargetToLinearRegression(metric, dataFrame)
        model.fit(X, y)
        y_predicted = model.predict(X)

        evaluateGoodnessOfPrediction(y, y_predicted)
        print('-----------------------------------')

        return y_predicted


    # In[144]: Store scalingDF in a new DF ->

    temporaryScalingDF = scalingDF.copy()


    # In[145]: The format as I will store Linear Regression models for each metric

    d={}
    for i in metricNames:
        d["model{0}".format(i)]="Hello " + i
        # print(d)

    d.get('modelCPU')


    # In[146]: Declare some functions, Calculate and store each Linear Regression model of each metrics

    def learningLinearRegression(scalingDF, temporaryScalingDF, metricNames):
        # linearRegressionModels = {}
        # temporaryScalingDF = scalingDF.copy()
        d={}

        for i in metricNames:

            # d["model{0}".format(i)]="Hello " + i

            model = calculateLinearRegressionModel(i, scalingDF)
            prediction = calculateLinearRegressionPrediction(i, scalingDF, model)

            # save model to the file system
            joblib.dump(model, 'models/saved_linearregression_model_' + i + '.pkl')

            # nos van egy ilyen modellünk mit tegyünk vele, tároljuk el mindegyiket különböző néven
            d["model{0}".format(i)] = model

            # el kéne tárolni
            temporaryScalingDF['predictedNext1'+i] = prediction


        return temporaryScalingDF, d


    # In[147]: Calculate and Store each Linear Regression model of each metrics. Predicted and Real values are stored in a new DF.

    temporaryScalingDF, linearRegressionModels = learningLinearRegression(scalingDF, temporaryScalingDF, metricNames)


    # In[148]: Explore
    if( explore ):
        linearRegressionModelNames = linearRegressionModels.keys()
        print(linearRegressionModelNames)

        modelCPU = linearRegressionModels.get('modelCPU')
        print(type(modelCPU))

        print(temporaryScalingDF.columns)
        print(temporaryScalingDF.shape)



    # In[152]: Visualize

    from visualizerlinux import ipythonPlotMetricsRealAgainstPredicted

    if showPlots :
        ipythonPlotMetricsRealAgainstPredicted(temporaryScalingDF, metricNames)


    from visualizerlinux import ipythonPlotMetricsRealAgainstPredictedRegression

    if showPlots :
        ipythonPlotMetricsRealAgainstPredictedRegression(temporaryScalingDF, metricNames)

    
    training_result = [error_msg]
    return training_result

    # ## ------------------------------------------------------------------------------------------------------
    # ## End of Learning Phase
    # ## ------------------------------------------------------------------------------------------------------
