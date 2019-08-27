import logging
import logging.config

import opt_config

from utils import printNormalizedX, printNormalizedY

from visualizerlinux import TimeLinePlot
from visualizerlinux import ScatterPlots
from visualizerlinux import TimeLinePlots
from visualizerlinux import CorrelationMatrixSave
from visualizerlinux import VisualizePredictedYScatter
from visualizerlinux import VisualizePredictedYLine, VisualizePredictedYLineWithValues
from visualizerlinux import ipythonPlotMetricsRealAgainstPredictedRegression
from visualizerlinux import ipythonPlotMetricsRealAgainstPredicted

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn import metrics

np.set_printoptions(precision=3, suppress=True)

pandas_dataframe_styles = {
    'font-family': 'monospace',
    'white-space': 'pre'
}

_target_variable = None
_input_metrics = None
_worker_count = None
_training_samples_required = None
_outsource_metrics = None
_prev_temporary_scaling_df_row = 0


# ## ------------------------------------------------------------------------------------------------------
# ## Define init method
# ## ------------------------------------------------------------------------------------------------------

def init(target_variable, input_metrics, worker_count, training_samples_required, outsource_metrics):
    
    logger = logging.getLogger('optimizer')
    
    logger.info('')
    logger.info('---------------------------  trainer init  -------------------------')
    logger.info('')
    
    global _target_variable
    _target_variable = target_variable[0]
    
    global _input_metrics
    _input_metrics = input_metrics
    
    global _worker_count
    _worker_count = worker_count[0]
    
    global _training_samples_required
    _training_samples_required = training_samples_required
    
    global _outsource_metrics
    _outsource_metrics = outsource_metrics
    
    logger.info('     ----------------------------------------------')
    logger.info('     ----------- TRAINER INIT DIAGNOSIS -----------')
    logger.info('     ----------------------------------------------')
    
    logger.info(f'     target_variable = {target_variable}')
    logger.info(f'     _target_variable = {_target_variable}')
    logger.info(f'     _worker_count = {_worker_count}')
    logger.info(f'     worker_count = {worker_count}')
    logger.info(f'     _input_metrics = {_input_metrics}')
    logger.info(f'     input_metrics = {input_metrics}')
    logger.info(f'     _training_samples_required = {_training_samples_required}')
    logger.info(f'     training_samples_required = {training_samples_required}')
    logger.info(f'     _outsource_metrics = {_outsource_metrics}')
    logger.info(f'     outsource_metrics = {outsource_metrics}')
    
    logger.info('---------------------------  trainer init end  -------------------------')
    logger.info('')

    
# ## ------------------------------------------------------------------------------------------------------
# ## Define run
# ## ------------------------------------------------------------------------------------------------------

def run(nn_file_name, visualize = False):

    logger = logging.getLogger('optimizer')
    
    logger.info('---------------------------     opt_trainer.run()     ---------------------------')
    

    logger.info('---------------------------------------------------------------------------------')
    logger.info(f'      nn_file_name     = {nn_file_name}')
    logger.info('---------------------------------------------------------------------------------')
    

    # Declare variables
    inputCSVFile   = 'data/grafana_data_export_long_running_test.csv'
    neuralCSVFile = nn_file_name
    
    # targetVariable = 'avg latency (quantile 0.5)'
    targetVariable = _target_variable
    
    logger.info('---------------------------------------------------------------------------------')
    logger.info(f'      targetVariable   = {targetVariable}')
    logger.info(f'      _target_variable = {_target_variable}')
    logger.info('---------------------------------------------------------------------------------')

    inputMetrics = _input_metrics
    
    logger.info('---------------------------------------------------------------------------------')
    logger.info(f'      inputMetrics     = {inputMetrics}')
    logger.info(f'      _input_metrics   = {_input_metrics}')
    logger.info('---------------------------------------------------------------------------------')

    workerCount = _worker_count
    
    logger.info('---------------------------------------------------------------------------------')
    logger.info(f'      workerCount      = {workerCount}')
    logger.info(f'      _worker_count    = {_worker_count}')
    logger.info('---------------------------------------------------------------------------------')

    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Don't touch it if you don't know what you do
    # ## ------------------------------------------------------------------------------------------------------
    
    scaler_min = -1                     # 0
    scaler_max = 1                      # 1
    train_test_ratio = 0.3              # 0.3
    activation_function = 'tanh'        # tanh, relu, logistic
    neuronsWhole = 4                    # 10
    neuronsTrainTest = 4                # 4
    cutFirstCases = 0                   # 0
    
    lead = 1                            # 1 default

    showPlots = False                   # True
    showPlots = visualize               # This value comes as a parameter
    explore = False                     # False
    
    error_msg = 'No error'              # None


    # In[3]: Declare some functions

    def readCSV(filename):
        df = pd.read_csv(filename, sep=";", header="infer", skiprows=1, na_values="null")
        return df

    # Read DataFrame
    df = readCSV(inputCSVFile)
    
    def readNeuralCSV(filename):
        df = pd.read_csv(filename, sep=",", header="infer", skiprows=0, na_values="null")
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
        logger.info('---------------------------------------------------------------------------------')
        logger.info('      --------------   pre-processed DataFrame from csv   --------------         ')
        logger.info(f'      df.shape    = {df.shape}')
        logger.info(f'      df.columns  = {df.columns}')
        # logger.info(f'      df.head()   = {df.head()}')
        logger.info('     -------------------------------------------------------------------         ')
        logger.info('---------------------------------------------------------------------------------')


    # ## ------------------------------------------------------------------------------------------------------
    # ## Pre-processing
    # ## ------------------------------------------------------------------------------------------------------
    
    # Preprecess DataFrame
    preProcessedDF = preProcessing(df)   

    # Print DataFrame Info
    dataFrameInfo(preProcessedDF)

    # Declare some functions
    def renameVariable(df, old_var_name, new_var_name):
        new_df = df.copy()
        if( df.columns.contains(old_var_name) ):
            new_df.rename(columns={old_var_name: new_var_name}, inplace=True)
        else:
            logger.info('--------------------- Wrong Column Name ---------------------')
        return new_df


    WorkerCountName = _worker_count
        
    logger.info(f'WorkerCountName = {WorkerCountName}')
    
    
    # Rename Worker count or vm_number to WorkerCount
    preProcessedDF = renameVariable(preProcessedDF, WorkerCountName, 'WorkerCount')


    # In[10]: Set Metrics Names

    def setMetricNames(names):
        new_metricNames = names.copy()
        return new_metricNames

    # metricNames = setMetricNames(['CPU', 'Inter', 'CTXSW', 'KBIn', 'PktIn', 'KBOut', 'PktOut'])
    
    # ## ------------------------------------------------------------------------------------------------------            
    # ## redefine metricNames
    # ##
    # ## Here can occure a potential problem cased by the difference between the input_metrics list
    # ## and metricNames list.
    # ## Because we don't know which are outsource or external metrics and which are so called inside
    # ## metrics
    # ##
    # ## I solved this problem in a following way:
    # ## I remove every element from metricNames list which occur in _outsorce_metrics list
    # ##
    # ## In my sample application these are the 'AVR_RR' and 'SUM_RR'
    # ##
    # ## ------------------------------------------------------------------------------------------------------
    logger.info('----------------------------------------------------------')
    logger.info('------------------- set metricNames ----------------------')
    logger.info('----------------------------------------------------------')
    
    metricNames = _input_metrics.copy()

    for i in _outsource_metrics:
        logger.info(f'removed elements of the metricNames list = {i}')
        metricNames.remove(i)
    
    logger.info(f'metricNames = {metricNames}')
    logger.info(f'_input_metrics = {_input_metrics}')
    
    logger.info('----------------------------------------------------------')

    
    
    # ## ------------------------------------------------------------------------------------------------------            
    # ## Drop first n row if needed
    # ## ------------------------------------------------------------------------------------------------------
    
    # In[14]: Drop First Cases
    # def dropFirstCases(df, n):
    #     new_df = df.copy()
    #     filteredDF = new_df[new_df.index > n]
    #     return filteredDF

    # If in the begining of the samples have a lot of outliers
    # filteredDF = dropFirstCases(preProcessedDF, cutFirstCases)


    # In[16]:
    # ## ------------------------------------------------------------------------------------------------------            
    # ## preProcessedDF let filteredDF
    # ## ------------------------------------------------------------------------------------------------------
    # preProcessedDF = filteredDF
    
    logger.info('------------------- preProcessedDF -----------------------')
    logger.info('----------------------------------------------------------')
    logger.info(f'preProcessedDF.shape = {preProcessedDF.shape}')
    logger.info(f'preProcessedDF.columns = {preProcessedDF.columns}')
    # logger.info(f'preProcessedDF.head(2) = {preProcessedDF.head(2)}')
    logger.info('----------------------------------------------------------')
    

    
    # ## ------------------------------------------------------------------------------------------------------            
    # ## Report
    # ## ------------------------------------------------------------------------------------------------------
    
    rows = preProcessedDF.shape[0]
    
    # CorrelationMatrixSave(preProcessedDF)
    if rows % 3 == 0: ScatterPlots(preProcessedDF, preProcessedDF[targetVariable], _input_metrics, targetVariable)
    if rows % 3 == 1: TimeLinePlot(preProcessedDF, targetVariable)
    if rows % 3 == 2: TimeLinePlots(preProcessedDF, _input_metrics)
    
    
    # In[26]:
    # ## ------------------------------------------------------------------------------------------------------            
    # ## Create a whole new DataFrame for Before After Data
    # ## ------------------------------------------------------------------------------------------------------
    
    # ## This is only for further development - consider the lags as inputs for neural network

    logger.info('---------------------------------------------------------------------------------')
    logger.info('---------------   createBeforeafterDF(df, lag, inputMetrics)    -----------------')
    logger.info(f'preProcessedDF.columns = {preProcessedDF.columns}')
    logger.info(f'len(inputMetrics) = {len(inputMetrics)}')
    logger.info(f'inputMetrics = {inputMetrics}')

    def createBeforeafterDF(df, lag, inputMetrics):
        beforeafterDF = df.copy()
        length = len(inputMetrics)
        inputVariables = np.flip(beforeafterDF.columns[0:length].ravel(), axis=-1)
        # print('Input Variablels : ', inputVariables)

        index = length
        for i in inputVariables:
            new_column = beforeafterDF[i].shift(lag)
            new_column_name = (i + str(1)) # Todo: rename str(lag)
            beforeafterDF.insert(loc=index, column=new_column_name, value=new_column)

        beforeafterDF = beforeafterDF[lag:]

        logger.info('---------------------------------------------------------------------------------')
        logger.info(f'Before After DF columns = {beforeafterDF.columns}')

        return beforeafterDF


    
    # In[27]:
    # ## ------------------------------------------------------------------------------------------------------
    # ## Create new dataframe with lags
    # ## ------------------------------------------------------------------------------------------------------

    beforeafterDF = createBeforeafterDF(preProcessedDF, 1, inputMetrics)

    logger.info('---------------------------------------------------------------------------------')
    logger.info('                        CreateBeforeAfter method done                            ')
    logger.info('---------------------------------------------------------------------------------')


    
    
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Set Features for Neural Network - these are the input variables
    # ## ------------------------------------------------------------------------------------------------------

    # In[28]: Declare some functions

    def setFeatures(df, columnNames):
        # X = df.iloc[:,0:9]
        X = df[columnNames]
        return X


    # In[29]:
    # ## ------------------------------------------------------------------------------------------------------
    # ## Set Features in other words this set will be the Input Variables
    # ## ------------------------------------------------------------------------------------------------------

    # X = setFeaturesAndTheirLags(beforeafterDF, inputMetrics)
    X = setFeatures(beforeafterDF, inputMetrics)

    logger.info('---------      Set Features method done       ------------')
    logger.info('-------------------------- X -----------------------------')
    logger.info('----------------------------------------------------------')

    logger.info('----------------------------------------------------------')
    logger.info(f'      type(X) = {type(X)}')
    logger.info(f'      X.shape = {X.shape}')
    logger.info(f'      X.columns = {X.columns}')
    logger.info(X.head(3))
    logger.info('----------------------------------------------------------')
    
    
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Set Target Variable for Neural Network - this is the target variable
    # ## ------------------------------------------------------------------------------------------------------

    # In[30]: Declare some functions
    def setTarget(df, targetVariable):
        y = df[targetVariable]
        return y


    # In[31]: Set target variable
    y = setTarget(beforeafterDF, targetVariable)
    
    logger.info('   Y target variable as a pandas.series -> numpy.array    ')
    logger.info('----------------------------------------------------------')
    logger.info(y.head())
    logger.info(f'(y describe = {y.describe()}')
    logger.info('----------------------------------------------------------')

    
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
    
    logger.info('')
    logger.info('---------          X_normalized done          ------------')
    logger.info('-------------------------- X -----------------------------')


    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Load MinMaxScalerXFull
    # ## ------------------------------------------------------------------------------------------------------
    
    # In[37]: Declare some functions
    def loadMinMaxScalerXFull():
        X_normalized_MinMaxScaler = joblib.load('models/scaler_normalizeX.save')

        return X_normalized_MinMaxScaler


    # In[38]: Load Saved Normalized Data (Normalizer)
    X_normalized_MinMaxScaler = loadMinMaxScalerXFull()
    
    logger.info('')
    logger.info('--------- X_normalized_MinMaxScaler load done ------------')
    logger.info('-------------------------- X -----------------------------')



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
    
    logger.info('')
    logger.info('---------          y_normalized done          ------------')
    logger.info('-------------------------- y -----------------------------')


    # In[48]:
    if( explore ):
        printNormalizedY(y_normalized)
        y_normalized[0:3]


    # ## ------------------------------------------------------------------------------------------------------
    # ## Load MinMaxScalerYFull
    # ## ------------------------------------------------------------------------------------------------------

    # In[50]: Declare some functions
    def loadMinMaxScalerYFull():
        y_normalized_MinMaxScaler = joblib.load('models/scaler_normalizeY.save')

        return y_normalized_MinMaxScaler


    # In[51]: Load Saved Normalized Data (Normalizer)
    y_normalized_MinMaxScaler = loadMinMaxScalerYFull()
    
    logger.info('')
    logger.info('--------- y_normalized_MinMaxScaler load done ------------')
    logger.info('-------------------------- y -----------------------------')


    # In[52]: De-normalize Features set
    y_denormalized = y_normalized_MinMaxScaler.inverse_transform(y_normalized.reshape(y_normalized.shape[0],1))


    # In[53]:
    if( explore ):
        y_denormalized[0:3]
        y_denormalized[-3:]

    logger.info('')
    logger.info('')
    logger.info('---------          Normalization done         ------------')
    logger.info('----------------------------------------------------------')

    
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Train Neural Network with Optimizer Class, trainMultiLayerRegressor method
    # ## ------------------------------------------------------------------------------------------------------
    
    logger.info('----------------------------------------------------------')
    logger.info('----------------------- MLP start ------------------------')
    logger.info('----------------------------------------------------------')

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
    goodness_of_fitt = evaluateGoodnessOfPrediction(y_normalized, y_predicted)
    
    logger.info('------------- Neural Network Goodness of Fitt ------------')
    logger.info('----------------------------------------------------------')
    logger.info('   evaluateGoodnessOfPrediction(y_normalized, y_predicted)')
    logger.info('   This dictionary is also the part of the return of Train')
    logger.info(f'(  goodness_of_fitt = \n {goodness_of_fitt}')
    logger.info('----------------------------------------------------------')
    
    
    # TODO
    # visszatérni az értékekkel és eltárolni őket valamilyen változóban
    
    
    # ## ------------------------------------------------------------------------------------------------------            
    # ## Report
    # ## ------------------------------------------------------------------------------------------------------
    
    VisualizePredictedYScatter(y_normalized, y_predicted, targetVariable)
    VisualizePredictedYLineWithValues(y_normalized, y_predicted, targetVariable, 'Normalized')
    




    # ### De-normlaize

    # ## ------------------------------------------------------------------------------------------------------
    # ## I want to see the result in original scale. I don't care about the X but the y_normalized and y_predcited.
    # ## ------------------------------------------------------------------------------------------------------
    
    # In[65]: De-normalize target variable and predicted target variable
    y_denormalized = y_normalized_MinMaxScaler.inverse_transform(y_normalized.reshape(y_normalized.shape[0],1))
    y_predicted_denormalized = y_normalized_MinMaxScaler.inverse_transform(y_predicted.reshape(y_predicted.shape[0],1))



    # In[68]: Declare De-normalizer functions
    def denormalizeX(X_normalized, X_normalized_MinMaxScaler):
        X_denormalized = X_normalized_MinMaxScaler.inverse_transform(X_normalized)
        return X_denormalized


    # In[69]: De-normalize Features
    X_denormalized = denormalizeX(X_normalized, X_normalized_MinMaxScaler)


    # In[74]: Declare De-normalizer functions
    def denormalizeY(y_normalized, y_normalized_MinMaxScaler):
        y_denormalized = y_normalized_MinMaxScaler.inverse_transform(y_normalized.reshape(y_normalized.shape[0],1))
        return y_denormalized


    # In[75]: De-normalize Target
    y_denormalized = denormalizeY(y_normalized, y_normalized_MinMaxScaler)
    y_predicted_denormalized = denormalizeY(y_predicted, y_normalized_MinMaxScaler)
    
    
    # ## ------------------------------------------------------------------------------------------------------            
    # ## Report
    # ## ------------------------------------------------------------------------------------------------------
    
    VisualizePredictedYLineWithValues(y_denormalized, y_predicted_denormalized, targetVariable, 'Denormalized')
    
    
    
    

    # ## ------------------------------------------------------------------------------------------------------
    # ## Linear Regression Learn
    # ## ------------------------------------------------------------------------------------------------------
    
    logger.info('----------------------------------------------------------')
    logger.info('-------------- Linear Regression start -------------------')
    logger.info('----------------------------------------------------------')
        
    # In[125]: Declare some functions

    # TODO:
    # Átvezetni valahogy, hogy a bemeneti változók fényében kezelje hogy hány változó van a dataframeben
    
    # Ugy látom, hogy az advice-ban sehol nem szerepel a trainer aminek az az oka
    # hogy az Advice-nak semmi szüksége nincs a lag-ok-ra se a lead-ek-re
    # Ugyanis miután meg van tanulva egy model, az már csak a model beolvasásával törödik
    # és abban egyáltalán nem szerepelnek a lagok meg a lagek,

    
    # When pandas.DataFrame (preProcessedDF) was constructed, the order of the variables
    # was determined by the program. The target variable is the latest.
    # As far as we do not care about the previous or following target variable,
    # the program hasn't got to count till the last column.
    # As soon as we get the pandas.DataFrame, the program will know the number of
    # the columns.
    # So 'index' is a temporal variable what contains the number of the columns -1
    
    # In other words, the lag will be computed for every column, except the last one.
 

    logger.info('----------------------------------------------------------')
    logger.info('-------------- Create Before After Diagnosis -------------')
    logger.info('----------------------------------------------------------')
    
    logger.info('     preProcessedDF the input of createBeforeafterDFLags()')
    logger.info('')
    logger.info(f'     preProcessedDF.shape = {preProcessedDF.shape}')
    logger.info(f'     preProcessedDF.columns = {preProcessedDF.columns}')
    
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Linear Regression Calculate N'th previous values
    # ## ------------------------------------------------------------------------------------------------------
    
    def createBeforeafterDFLags(df, lag):
        beforeafterDFLags = df.copy()
        dfColumnsNumber = beforeafterDFLags.shape[1]
        logger.info(f'     createBeforeafterDFLags(df, lag) df col number    = {dfColumnsNumber}')
        # index = 10
        index = dfColumnsNumber - 1
        logger.info(f'     createBeforeafterDFLags(df, lag) df col number -1 = {index} \n')
        
        inputVariables = np.flip(beforeafterDFLags.columns[0:index].ravel(), axis=-1)
        logger.info(f'     Input Variables in createBeforeafterDFLags = {inputVariables} \n')

        for i in inputVariables:
            new_column = beforeafterDFLags[i].shift(lag)
            new_column_name = (str('prev') + str(1) + i)
            beforeafterDFLags.insert(loc=index, column=new_column_name, value=new_column)

        beforeafterDFLags = beforeafterDFLags[lag:]             # remove first row as we haven't got data in lag var

        return beforeafterDFLags, index                         # return not just the df but an int as well

    
    # In[126]: Create lag variables (see above -> 'prev1CPU', 'prev1Inter', etc)
    beforeafterDFLags, index1 = createBeforeafterDFLags(preProcessedDF, 1)
    
    logger.info('----------------------------------------------------------')
    logger.info('-------------- Create Before After Diagnosis -------------')
    logger.info('----------------------------------------------------------')
    
    logger.info('     after createBeforeafterDFLags(preProcessedDF, 1)')
    logger.info('     beforeafterDFLags, index1 = createBeforeafterDFLags(preProcessedDF, 1)')
    logger.info(f'     beforeafterDFLags.shape = {beforeafterDFLags.shape} \n')

    logger.info(f'     beforeafterDFLags.columns = {beforeafterDFLags.columns}')
    logger.info('---------------------------------------------------------- \n')
    logger.debug(f"\n {beforeafterDFLags[['prev1CPU', 'CPU']].head(10)}")
    logger.debug('----------------------------------------------------------')
    logger.debug(f"\n {beforeafterDFLags[['WorkerCount', 'prev1WorkerCount']].head(10)}")
    logger.debug('----------------------------------------------------------')
    logger.debug(f"\n {beforeafterDFLags[['WorkerCount', 'prev1WorkerCount']].tail(10)}")


    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Linear Regression Calculate N'th next values
    # ## ------------------------------------------------------------------------------------------------------

    # Na itt viszont már para van itt viszont már tudnia kell, hogy mi is tulajdonképen
    # változók hossza
    def createBeforeafterDFLeads(df, index, lead = 1):
        beforeafterDFLeads = df.copy()
        inputVariables = np.flip(beforeafterDFLeads.columns[0:index].ravel(), axis=-1)
        logger.info(f'Input Variables in createBeforeafterDFLeads: {inputVariables} \n')

        # In the case of WorkerCount column we take account the next value.
        # Every other case we take account the parameter what was given by the user.
        for i in inputVariables:
            if( i == 'WorkerCount'):
                lead_value = 1
            else:
                lead_value = lead
                
            new_column = beforeafterDFLeads[i].shift(-lead_value)
            new_column_name = (str('next') + str(1) + i)
            beforeafterDFLeads.insert(loc=index, column=new_column_name, value=new_column)

        beforeafterDFLeads = beforeafterDFLeads[:-lead]         # remove last row as we haven't got data in lead (next) variables
        beforeafterDFLeads = beforeafterDFLeads.iloc[:,:-1]     # remove last column - Latency

        return beforeafterDFLeads


    # In[129]: Create lead variables (see above -> 'next1CPU', 'next1Inter', etc)
    beforeafterDF = createBeforeafterDFLeads(beforeafterDFLags, index1, lead = lead)
    
    logger.info('----------------------------------------------------------')
    logger.info('-------------- Create Before After Diagnosis -------------')
    logger.info('----------------------------------------------------------')
    
    logger.info('     after createBeforeafterDFLeads(beforeafterDFLags, index1, lead = lead)')
    logger.info('     beforeafterDF = createBeforeafterDFLeads(beforeafterDFLags, index1, lead = lead)')
    logger.info(f'     beforeafterDF.shape = {beforeafterDF.shape} \n')

    logger.info(f'     beforeafterDF.columns = {beforeafterDF.columns}')
    logger.info('---------------------------------------------------------- \n')
    logger.debug(f"\n {beforeafterDF[['prev1CPU', 'CPU', 'next1CPU']].head(10)}")
    logger.debug('----------------------------------------------------------')
    logger.debug(f"\n {beforeafterDF[['WorkerCount', 'prev1WorkerCount', 'next1WorkerCount']].head(10)}")
    logger.debug('----------------------------------------------------------')
    logger.debug(f"\n {beforeafterDF[['WorkerCount', 'prev1WorkerCount', 'next1WorkerCount']].tail(10)}")

    # In[131]: Assert
    logger.debug('----------------------------------------------------------')
    logger.debug('----------               Assert             --------------')
    logger.debug('----------------------------------------------------------')
    
    logger.debug(f'---------------- original _input_metrics length = {len(_input_metrics)}')
    logger.debug(f'---------------- beforeafterDF.shape[1] = {beforeafterDF.shape[1]}')
    logger.debug(f'---------------- {len(_input_metrics)} + 1 * 3')
    logger.debug('---------------------------------------------------------- \n')

    # ## ------------------------------------------------------------------------------------------------------
    # ## Linear Regression Calculate before-after differencies of WorkerCount (vm_number)
    # ## ------------------------------------------------------------------------------------------------------

    def calculateWorkerCountDifferences(beforeafterDF):
        new_beforeafterDF = beforeafterDF.copy()
        new_beforeafterDF['addedWorkerCount'] = new_beforeafterDF['next1WorkerCount'].values - new_beforeafterDF['WorkerCount']

        return new_beforeafterDF


    # In[133]: Explore data
    theBeforeAfterDF = calculateWorkerCountDifferences(beforeafterDF)
    
    logger.info('----------------------------------------------------------')
    logger.info('-------------- Create Before After Diagnosis -------------')
    logger.info('----------------------------------------------------------')

    logger.info('     after calculateWorkerCountDifferences(beforeafterDF)')
    logger.info(f'     theBeforeAfterDF.shape = {theBeforeAfterDF.shape} \n')
    # logger.info('')
    logger.info(f'     theBeforeAfterDF.columns = {theBeforeAfterDF.columns}')
    logger.debug(f"\n {theBeforeAfterDF[['WorkerCount', 'prev1WorkerCount', 'next1WorkerCount']].head(10)}")
    logger.debug(f"\n {theBeforeAfterDF[['WorkerCount', 'prev1WorkerCount', 'next1WorkerCount']].tail(10)}")
    logger.info('---------------------------------------------------------- \n')
    
    logger.info('----------------------------------------------------------')
    logger.info('------------ Create Before After Columns Done ------------')
    logger.info('----------------------------------------------------------')
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Linear Regression Create Before After Columns Done
    # ## ------------------------------------------------------------------------------------------------------
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Filter rows where actual WorkerCount != next1WorkerCount
    # ## ------------------------------------------------------------------------------------------------------
    
    
    # In[134]: Declare some functions
    def createScalingDF(theBeforeAfterDF):
        new_beforeafterDF = theBeforeAfterDF.copy()
        scalingDF = new_beforeafterDF[new_beforeafterDF.WorkerCount != new_beforeafterDF.next1WorkerCount]
        return scalingDF


    # In[135]: Collect rows where 'WorkerCount != next1WorkerCount' -> If this condition is True that means scalling has happened
    scalingDF = createScalingDF(theBeforeAfterDF)

    logger.info('----------------------------------------------------------')
    logger.info('---------- Select where worker != next1Worker ------------')
    logger.info('----------------------------------------------------------')
    
    logger.info('     after createScalingDF(theBeforeAfterDF)')
    logger.info(f'     scalingDF.shape = {scalingDF.shape} \n')
    # logger.info('')
    logger.info(f'     scalingDF.columns = \n {scalingDF.columns} \n')
    logger.info(f"\n {scalingDF[['WorkerCount', 'prev1WorkerCount', 'next1WorkerCount', 'addedWorkerCount']].head(2)}")
    logger.info(f"\n {scalingDF[['WorkerCount', 'prev1WorkerCount', 'next1WorkerCount', 'addedWorkerCount']].tail(2)}")
    logger.info('----------------------------------------------------------\n')
    
    logger.info('----------------------------------------------------------')
    logger.info('----- End of before-after scaling data preparation -------')
    logger.info('----------------------------------------------------------')
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## This is the end of before-after scaling data preparation
    # ## ------------------------------------------------------------------------------------------------------
        

    
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Linear regression calculation part
    # ## ------------------------------------------------------------------------------------------------------
    
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
        # logger.debug(y.head(1))
        # logger.debug(featuresDF.head(1))
        # logger.debug(X.head(2))
        # scalingDF[['CPU', 'next1CPU', 'WorkerCount', 'next1WorkerCount', 'addedWorkerCount']][0:3]
        return Xb, yb


    def calculateLinearRegressionModel(currentMetric, dataFrameA):
        newDataFrameA = dataFrameA.copy()
        Xa, ya = createInputAndTargetToLinearRegression(currentMetric, newDataFrameA)

        lr = LinearRegression(fit_intercept=True, normalize=False)
        lr.fit(Xa, ya)
        return lr


    def calculateLinearRegressionPrediction(metric, dataFrame, model):
        X, y = createInputAndTargetToLinearRegression(metric, dataFrame)
        model.fit(X, y)
        y_predicted = model.predict(X)

        logger.info(metric)
        evaluateGoodnessOfPrediction(y, y_predicted)
        logger.info('-----------------------------------')
        return y_predicted


    # In[144]: Store scalingDF in a new DF ->

    temporaryScalingDF = scalingDF.copy()
    
    
    # A potential problem can occure if you miss or leave it out the 'AVG_RR'
    # and 'SUM_RR' columns from pandas.DataFrame
    #
    # These two variables are quite different than any other 'input metrics'
    #
    # At this point of the program the 'metricNames' should not cointain these
    # two variables. 'metricNames is a list, so you should make sure that
    # 'metricNames' list does not contain any of them.
    #
    # The program creates linear regression model each element of the metricNames.
    # But be aware.
    # You must not create linear regression model for 'AVG_RR' and 'SUM_RR' respectively.
 


    # In[145]: The format as I will store Linear Regression models for each metric
    
    # print('_______________________________________________________')
    # d={}
    # for i in metricNames:
    #     d["model{0}".format(i)]="Hello " + i
    #     print(d)

    # d.get('modelCPU')
    # print('_______________________________________________________')
    

    # In[146]: Declare some functions, Calculate and store each Linear Regression model of each metrics

    def learningLinearRegression(scalingDF, temporaryScalingDF, metricNames):

        d={}

        for i in metricNames:

            # d["model{0}".format(i)]="Hello " + i
            
            model = calculateLinearRegressionModel(i, scalingDF)
            
            prediction = calculateLinearRegressionPrediction(i, scalingDF, model)

            # save model to the file system
            joblib.dump(model, 'models/saved_linearregression_model_' + i + '.pkl')

            # store each model by its name
            d["model{0}".format(i)] = model

            # el kéne tárolni
            temporaryScalingDF['predictedNext1'+i] = prediction

        return temporaryScalingDF, d


    # In[147]: Calculate and Store each Linear Regression model of each metrics. Predicted and Real values are stored in a new temporaryScalingDF.

    temporaryScalingDF, linearRegressionModels = learningLinearRegression(scalingDF, temporaryScalingDF, metricNames)


    # In[148]: Explore
    if( True ):
        print('____________________________________________________________________________________________')
        linearRegressionModelNames = linearRegressionModels.keys()
        print(linearRegressionModelNames)

        # modelCPU = linearRegressionModels.get('modelCPU')
        # print(type(modelCPU))
        # print(modelCPU)

        print('temporaryScalingDF.columns = ', temporaryScalingDF.columns)
        print('temporaryScalingDF.shape   = ', temporaryScalingDF.shape)
        print('____________________________________________________________________________________________\n')


    # In[152]: Visualize the linear relationship between the real and the predicted metrics values
    # ## ------------------------------------------------------------------------------------------------------            
    # ## Report
    # ## ------------------------------------------------------------------------------------------------------

    # Comment:
    #
    # This ipythonPlotMetricsRealAgainstPredictedRegression(temporaryScalingDF, metricNames) method
    # executes a quite time consuming process. It is not necessary for proper program running but
    # can be very useful for further investigation and data exploration.
    #
    # It is reasonable to call this when scale has happened, otherwise it gives back the same results.
    global _prev_temporary_scaling_df_row
    if( _prev_temporary_scaling_df_row != temporaryScalingDF.shape[0] ):
        _prev_temporary_scaling_df_row = temporaryScalingDF.shape[0]
        ipythonPlotMetricsRealAgainstPredictedRegression(temporaryScalingDF, metricNames)

    
    # ## ------------------------------------------------------------------------------------------------------            
    # ## Return with training_result
    # ## ------------------------------------------------------------------------------------------------------

    # TODO:
    # Insted of linearRegressionModels I have to figure out some better output, till then I will ignore it
    # training_result = [error_msg, goodness_of_fitt, linearRegressionModels]
    training_result = [error_msg, goodness_of_fitt]
    return training_result

    # ## ------------------------------------------------------------------------------------------------------
    # ## End of Learning Phase
    # ## ------------------------------------------------------------------------------------------------------
