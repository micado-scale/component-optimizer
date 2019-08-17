import logging
import logging.config

import opt_config

from utils import printNormalizedX, printNormalizedY

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
_training_samples_required = None
# Prométeusz nem fogad el szóközt ezért néha ez néha a felső aszerint, hogy melyik grafana filét olvassuk ki.
# _outsource_metrics = ['AVG RR', 'SUM RR']
_outsource_metrics = ['AVG_RR', 'SUM_RR']


# ## ------------------------------------------------------------------------------------------------------
# ## Define init method
# ## ------------------------------------------------------------------------------------------------------

def init(target_variable, input_metrics, worker_count, training_samples_required):
    
    logger = logging.getLogger('optimizer')
    
    logger.info('----------------------- trainer init ----------------------')
    
    global _target_variable
    _target_variable = target_variable[0]
    
    global _input_metrics
    _input_metrics = input_metrics
    
    global _worker_count
    _worker_count = worker_count[0]
    
    global _training_samples_required
    _training_samples_required = training_samples_required
    
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
    
    logger.info('----------------------- trainer init end ----------------------')


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
    cutFirstCases = 0                   # 0

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
        logger.info('---------------------------------------------------------------------------------')
        logger.info('                   --------------- DataFrame ------------------                  ')
        logger.info(f'df.columns  = {df.columns}')
        logger.info(f'df.shape    = {df.shape}')
        # logger.info(f'df.head()   = {df.head()}')
        logger.info('                   --------------- DataFrame ------------------                  ')


    # ## ------------------------------------------------------------------------------------------------------
    # ## Pre-processing
    # ## ------------------------------------------------------------------------------------------------------
    
    # Preprecess DataFrame
    preProcessedDF = preProcessing(df)   

    # Print DataFrame Info
    dataFrameInfo(preProcessedDF)

    # Set targetVariable
    targetVariable = _target_variable
    logger.info(f'target variable set = {targetVariable}')


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
    
    
    
    
    
    # 333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
    logger.info('----------------------------------------------------------')
    logger.info('------------------- set metricNames ----------------------')
    logger.info('----------------------------------------------------------')
    # na itt lesz majd az alapvető probléma az inputMetric és a metricNames között
    # ugyanis igazából nem tudjuk, hogy az első kettő az amit le kell vágnunk a listából
    # ezért dirty ha szóval itt kell leválogatnom, hogy
    # a metrics names-ben
    # dobjuk el azt a két fránya AVR_RR SUM_RR oszlopot 
    
    # Ezeeket az értékeket az init-ben adom át neki, annyi a különbség, hogy az első két változóra nincs szükségünk
    # metricNames = _input_metrics[2:]
    metricNames = _input_metrics.copy()

    for i in _outsource_metrics:
        logger.info(f'removed elements of the metricNames list = {i}')
        metricNames.remove(i)
    
    logger.info(f'metricNames = {metricNames}')
    logger.info(f'_input_metrics = {_input_metrics}')
    
    logger.info('----------------------------------------------------------')
    # 333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
    
    

    
    # In[14]: Drop First Cases
    def dropFirstCases(df, n):
        new_df = df.copy()
        filteredDF = new_df[new_df.index > n]
        return filteredDF

    # If in the begining of the samples have a lot of outliers
    filteredDF = dropFirstCases(preProcessedDF, cutFirstCases)


    # In[16]:
    # ## ------------------------------------------------------------------------------------------------------            
    # ## preProcessedDF let filteredDF
    # ## ------------------------------------------------------------------------------------------------------
    preProcessedDF = filteredDF
    
    logger.info('------------------- preProcessedDF -----------------------')
    logger.info('----------------------------------------------------------')
    logger.info(f'preProcessedDF.shape = {preProcessedDF.shape}')
    logger.info(f'preProcessedDF.columns = {preProcessedDF.columns}')
    # logger.info(f'preProcessedDF.head(2) = {preProcessedDF.head(2)}')
    logger.info('----------------------------------------------------------')
    
    ## itt a preProcessedDF shape még minden jó volt aztátn egyszer csak valamiért 1 soros lesz az egsész





    # In[26]:
    # ## ------------------------------------------------------------------------------------------------------            
    # ## Create a whole new DataFrame for Before After Data
    # ## ------------------------------------------------------------------------------------------------------

    logger.info('CreateBeforeAfter method')
    logger.info(f'preProcessedDF.columns = {preProcessedDF.columns}')
    logger.info(f'len(inputMetrics) = {len(inputMetrics)}')
    logger.info(f'inputMetrics = {inputMetrics}')

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


    
    # In[27]:
    # ## ------------------------------------------------------------------------------------------------------
    # ## Create new dataframe with lags
    # ## ------------------------------------------------------------------------------------------------------

    beforeafterDF = createBeforeafterDF(preProcessedDF, 1, inputMetrics)

    logger.info('CreateBeforeAfter method done')


    
    
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Set Features for Neural Network - these are the input variables
    # ## ------------------------------------------------------------------------------------------------------

    # In[28]: Declare some functions

    logger.info('----------------------------------------------------------')
    logger.info(f'_input_metrics {_input_metrics}')
    logger.info(f'inputMetrics {inputMetrics}')
    logger.info('----------------------------------------------------------')

    def setFeaturesAndTheirLags(df, columnNames):
        # X = df.iloc[:,0:9]
        X = df[columnNames]
        return X


    # In[29]:
    # ## ------------------------------------------------------------------------------------------------------
    # ## Set Features in other words this set will be the Input Variables
    # ## ------------------------------------------------------------------------------------------------------

    X = setFeaturesAndTheirLags(beforeafterDF, inputMetrics)

    logger.info('--------- SetFeaturesAndTheirLags method done ------------')
    logger.info('-------------------------- X -----------------------------')
    logger.info('----------------------------------------------------------')

    logger.info('----------------------------------------------------------')
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


    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Load MinMaxScalerXFull
    # ## ------------------------------------------------------------------------------------------------------
    
    # In[37]: Declare some functions
    def loadMinMaxScalerXFull():
        X_normalized_MinMaxScaler = joblib.load('models/scaler_normalizeX.save')

        return X_normalized_MinMaxScaler


    # In[38]: Load Saved Normalized Data (Normalizer)
    X_normalized_MinMaxScaler = loadMinMaxScalerXFull()
    
    logger.info('X_normalized_MinMaxScaler load done')



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
    evaluateGoodnessOfPrediction(y_normalized, y_predicted)




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
    # ## Linear Regression Learn
    # ## ------------------------------------------------------------------------------------------------------
    
    logger.info('----------------------------------------------------------')
    logger.info('-------------- Linear Regression start -------------------')
    logger.info('----------------------------------------------------------')
        
    # In[124]: Import dependencies

    from sklearn.linear_model import LinearRegression
    from sklearn import metrics


    # In[125]: Declare some functions

    # TODO:
    # Átvezetni valahogy, hogy a bemeneti változók fényében kezelje hogy hány változó van a dataframeben
    
    # na itt is van valami fos
    # egyrészt kiprintelem az Input Variables nevü inputVariables-t amiből úgy látom hogy
    # úgy vettem hogy 10 darab van, de miért és mik ezek az inputVariables-ek?
    
    # az inputVariables a függvényen belül lesz megcsinálva és annyi előnye van, hogy
    # a preProcessedDF vagy ami bemegy abból veszi az oszlop neveket
    
    # viszont csak 0-10-ig mivel de a preProcessedDF-nek 11 oszlopa van,
    # sajnos itt úgy veszem, hogy azért 0-10-ig mert a 11-ik a target változó
    # és arra minek csináljuk meg a lag-et.
    
    # Egyébként akár meg is csinálhatnánk talán kevesebb lenne úgy a baj?
    
    # Ilyen szempontból meg kurvárna nem konzisztens a programom
    # mert itt átírom akkor melyik másik helyen hasal el?
    # Müködni fog-e az advice, ott miért nem ezt a függvényt hívom, vagy ott is ezt hívom meg?
    
    # Ugy látom, hogy az advice-ban sehol nem szerepel a trainer aminek az az oka
    # hogy az Advice-nak semmi szüksége nincs a lag-ok-ra se a lead-ek-re
    # Ugyanis miután meg van tanulva egy model, az már csak a model beolvasásával törödik
    # és abban egyáltalán nem szerepelnek a lagok meg a lagek,
    
    # Ugyanakkor elszomorító, hogy ezt a függvényt, sőt ezeket a függvényeket nem szerveztem ki.
    # Mindegy ennek is megvolt a maga prakitus oka, a traineren kívül senkinek nincs rá szüksége.
    
    #
    
    # probléma ott kezdődik hogy már arra sem emlékezem, hogy kezelem le a bejövő adatokat?
    # hogy kerülnek azok be a csv-be?
    # a yaml amit kapok az egy dictionarry eddig világos elég ha belenézek, de kezdek én
    # egyáltalán valamit a sorrendel?
    
    # az opt_rest.py 116-ik sorában van az input adatok beolvasása, tehát még csak nem is
    # a train-ben, ez jó,
    # de mi van ha nem ugyan abban a sorrendben vannak az adatok a yaml-ben egy egy minta között?
    
    # ha valahol le kell kezelnem ezt a sorrendet akkor az az opt_rest 136-ik sora lesz.
    # ott van egy input_metrics változóm, ami úgy áll elő, hogy végig iterálok a sample változón
    # ami egy dictionary és annak a sample input_metrics részén ami egy lista key value párokból.
    
    # ha az a helyzet, hogy az egyes minták között a sorrend változik, akkor ezt ott kell kezelenem
    # úgy, hogy minden key value párt eltárolok és a megfelelő sorrendbe állítom és ugy irom át
    # egy pandas dataframebe azt pedig egy csv-be
    
    # szóval annyi jó hírem van, hogy amikor összerakom a pandasDataFramet és kiirom CSV-be
    # akkor én határozom meg a sorrendet és úgy láttam, hogy a target változót a legutolsó
    # helyre, a legutolsó oszlopba rakom.
    # Tehát beolvasásnál is a legutolsó lesz.
    
    # ezért itt a createBeforeafterDFLags függvényben mivel megkapja a DF-t
    # ezért igazából csak az utolsó elötti sorig kell elszámolnom
    
    
    def createBeforeafterDFLags(df, lag):
        beforeafterDFLags = df.copy()
        dfColumnsNumber = beforeafterDFLags.shape[1]
        print('dfColumnsNumber = ', dfColumnsNumber)
        # index = 10
        index = dfColumnsNumber - 1
        
        # inputVariables = np.flip(beforeafterDFLags.columns[0:10].ravel(), axis=-1)
        inputVariables = np.flip(beforeafterDFLags.columns[0:index].ravel(), axis=-1)
        print('Input Variables in createBeforeafterDFLags: ', inputVariables)

        for i in inputVariables:
            new_column = beforeafterDFLags[i].shift(lag)
            new_column_name = (str('prev') + str(1) + i)
            beforeafterDFLags.insert(loc=index, column=new_column_name, value=new_column)

        beforeafterDFLags = beforeafterDFLags[lag:]             # remove first row as we haven't got data in lag var

        return beforeafterDFLags, index                         # return not just the df but an int as well


    logger.info('----------------------------------------------------------')
    logger.info('------------ Linear Regression diagnosis -----------------')
    logger.info('----------------------------------------------------------')
    logger.info('preProcessedDF is the input of the createBeforeafterDFLags()')
    logger.info('A createBeforeafterDFLags(df, lag)-ben szokott eltörni valami xxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    logger.info('')
    
    logger.info(f'preProcessedDF.shape = {preProcessedDF.shape}')
    logger.info(preProcessedDF.head(3))
    
    
    
    
    # In[126]: Create lag variables (see above -> 'prev1CPU', 'prev1Inter', etc)
    beforeafterDFLags, index = createBeforeafterDFLags(preProcessedDF, 1)




    # In[128]: Declare some functions

    # Na itt viszont már para van itt viszont már tudnia kell, hogy mi is tulajdonképen
    # változók hossza
    def createBeforeafterDFLeads(df, index, lead = 1):
        beforeafterDFLeads = df.copy()
        # inputVariables = np.flip(beforeafterDFLeads.columns[0:10].ravel(), axis=-1)
        inputVariables = np.flip(beforeafterDFLeads.columns[0:index].ravel(), axis=-1)
        print('Input Variables in createBeforeafterDFLeads: ', inputVariables)

        # index = 10
        for i in inputVariables:
            new_column = beforeafterDFLeads[i].shift(-lead)
            new_column_name = (str('next') + str(1) + i) # Todo: rename str(lead)
            beforeafterDFLeads.insert(loc=index, column=new_column_name, value=new_column)

        beforeafterDFLeads = beforeafterDFLeads[:-lead]         # remove last row as we haven't got data in lead var

        beforeafterDFLeads = beforeafterDFLeads.iloc[:,:-1]     # remove last column - Latency

        return beforeafterDFLeads


    # In[129]: Create lead variables (see above -> 'next1CPU', 'next1Inter', etc)
    beforeafterDF = createBeforeafterDFLeads(beforeafterDFLags, index, lead = lead)

    
    logger.info('----------------------------------------------------------')
    logger.info('------------ Create Before After Diagnosis ---------------')
    logger.info('----------------------------------------------------------')
    
    print(beforeafterDF.columns)
    print(beforeafterDF.shape)
    print(beforeafterDF.head(2))
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Linear Regression Create Before After Columns Done
    # ## ------------------------------------------------------------------------------------------------------
    
    logger.info('----------------------------------------------------------')
    logger.info('---------- Create Before After Columns Done --------------')
    logger.info('----------------------------------------------------------')
    

    # In[131]: Assert
    logger.info('----------------------------------------------------------')
    logger.info('----------               Assert             --------------')
    logger.info('----------------------------------------------------------')
    
    print('---------------- original _input_metrics length = ', len(_input_metrics))
    print('---------------- beforeafterDF.shape[1] = ', beforeafterDF.shape[1])
    print('---------------- ', len(_input_metrics), ' + 1 * 3')

    logger.info('----------------------------------------------------------')


    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Linear Regression Calculate before-after differencies
    # ## ------------------------------------------------------------------------------------------------------

    def calculateWorkerCountDifferences(beforeafterDF):
        new_beforeafterDF = beforeafterDF.copy()
        new_beforeafterDF['addedWorkerCount'] = new_beforeafterDF['next1WorkerCount'].values - new_beforeafterDF['WorkerCount']

        return new_beforeafterDF


    # In[133]: Explore data
    theBeforeAfterDF = calculateWorkerCountDifferences(beforeafterDF)
    
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Filter rows where actual WorkerCount != next1WorkerCount
    # ## ------------------------------------------------------------------------------------------------------


    # In[134]: Declare some functions
    def createScalingDF(theBeforeAfterDF):
        new_beforeafterDF = theBeforeAfterDF.copy()
        scalingDF = new_beforeafterDF[new_beforeafterDF.WorkerCount != new_beforeafterDF.next1WorkerCount]

        return scalingDF


    # In[135]: Collect rows where 'WorkerCount != next1WorkerCount' -> If this condition is True that means scalling happened
    scalingDF = createScalingDF(theBeforeAfterDF)

    
    # ## ------------------------------------------------------------------------------------------------------
    # ## This is the end of before-after data preparation
    # ## ------------------------------------------------------------------------------------------------------
    
    logger.info(f'scalingDF.shape = {scalingDF.shape}')
    print(scalingDF.head())
    logger.info('---------------------------------------------------------------------------------------------')
    
    logger.info('----------------------------------------------------------')
    logger.info('--------- End of before-after data preparation -----------')
    logger.info('----------------------------------------------------------')
    

    
    
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
    
    
    # Az egész probléma onnan indult ki, hogy én rendre eldobtam korábban az első két
    # változót a metrikák közül, ezek az AVG RR és a SUM RR voltak
    #
    # A Bajt az okozta, ha ezek nem az első két változók voltak, ezt a problémát
    # még mindíg nem oldottam meg
    #
    # Ha itt a metricNames nem úgy néz ki ahogy ki kell néznie akkor baj van
    # ha nem azt a két változót dobom el akkor hiába keresi az ezekre
    # készített prediction modelleket nem fogja megtalálni őket
    #
    # Tehát itt is nagyon észnél kell lenni, hogy mire készítem el a lineáris regressziós
    # modelleket és mire nem
    #
    # A metricName-nek összhangban kell lennie a CSV filével amibe kiirom
    #
    # Elvileg arra a bizonyos két 'metrikára' is csináltam before
    # after értékek kiszámítását
    
    # tehát ellenőrizni kell, hogy a regresszió milyen adatot használ
    # és milyen változó nevek vannak a DF-ben és listában amin végig iterál
    
    # A ScalingDF-et használja


    # In[145]: The format as I will store Linear Regression models for each metric

    print('_______________________________________________________')
    d={}
    for i in metricNames:
        d["model{0}".format(i)]="Hello " + i
        print(d)

    d.get('modelCPU')
    print('_______________________________________________________')

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
    if( True ):
        print('____________________________________________________________________________________________')
        linearRegressionModelNames = linearRegressionModels.keys()
        print(linearRegressionModelNames)

        # modelCPU = linearRegressionModels.get('modelCPU')
        # print(type(modelCPU))
        # print(modelCPU)

        print(temporaryScalingDF.columns)
        print(temporaryScalingDF.shape)
        print('____________________________________________________________________________________________')


    # In[152]: Visualize
    if showPlots :
        from visualizerlinux import ipythonPlotMetricsRealAgainstPredicted
        ipythonPlotMetricsRealAgainstPredicted(temporaryScalingDF, metricNames)

    if showPlots :
        from visualizerlinux import ipythonPlotMetricsRealAgainstPredictedRegression
        ipythonPlotMetricsRealAgainstPredictedRegression(temporaryScalingDF, metricNames)

    
    training_result = [error_msg]
    return training_result

    # ## ------------------------------------------------------------------------------------------------------
    # ## End of Learning Phase
    # ## ------------------------------------------------------------------------------------------------------
