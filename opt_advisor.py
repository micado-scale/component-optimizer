import logging
import logging.config

import re
import os
import time

import numpy as np
import pandas as pd

from flask import jsonify

from utils import loadMinMaxScalerXFull, loadMinMaxScalerYFull
from utils import loadNeuralNetworkModel
from opt_utils import readCSV
from opt_utils import preProcessing
from opt_utils import renameVariable
from opt_utils import dropFirstCases
from utils import setMetricNames, setExtendedMetricNames

from linearregression import calculateLinearRegressionTerms
from visualizerlinux import VisualizePredictedYLine
from visualizerlinux import VisualizePredictedYWithWorkers
from visualizerlinux import VisualizePredictedXY2Line
from visualizerlinux import VisualizePredictedXY3Line
from visualizerlinux import VisualizePredictedXYLine
from visualizerlinux import VisualizePredictedXYLine

from sklearn.externals import joblib

pandas_dataframe_styles = {
    'font-family': 'monospace',
    'white-space': 'pre'
}

    
# ## ------------------------------------------------------------------------------------------------------
# ## Define some variables
# ## ------------------------------------------------------------------------------------------------------
    
target_metric_min = None
target_metric_max = None
target_variable = None
input_variables = None
worker_count_name = None
outsource_metrics = None
config = None
constants = None
maximumNumberIncreasableNode = 6  # must be positive
minimumNumberReducibleNode = -6   # must be negative
first_advice = None
prev_adviced_time = 0
prev_advice_vm_total_number = 0


# ## ------------------------------------------------------------------------------------------------------
# ## Define init method
# ## ------------------------------------------------------------------------------------------------------
    
def init(_target_metric, input_metrics, worker_count, _outsource_metrics, _config, _constants):
    
    logger = logging.getLogger('optimizer')
    
    logger.info('')
    logger.info('---------------------------  advisor init  --------------------------')
    logger.info('')
    
    global input_variables
    input_variables = input_metrics
    
    global worker_count_name
    worker_count_name = worker_count[0]
    
    global target_metric_min 
    target_metric_min = _target_metric[0].get('min_threshold')

    global target_metric_max
    target_metric_max = _target_metric[0].get('max_threshold')
    
    global target_variable
    target_variable = _target_metric[0].get('name')
    
    global outsource_metrics
    outsource_metrics = _outsource_metrics
    
    global config
    config = _config
    
    global constants
    constants = _constants
    
    global maximumNumberIncreasableNode
    if config.maximum_number_increasable_node is not None:
        maximumNumberIncreasableNode = config.maximum_number_increasable_node
    
    global minimumNumberReducibleNode
    if config.minimum_number_reducible_node is not None:
        minimumNumberReducibleNode = config.minimum_number_reducible_node
    
    global first_advice
    first_advice = True
    
    global prev_adviced_time
    prev_adviced_time = float('-inf')
    
    global prev_advice_vm_total_number
    prev_advice_vm_total_number = 0

    logger.info('     ----------------------------------------------')
    logger.info('     ---------- ADVISOR INIT DIAGNOSIS ------------')
    logger.info('     ----------------------------------------------')
    
    logger.info(f'     maximumNumberIncreasableNode = {maximumNumberIncreasableNode}')
    logger.info(f'     minimumNumberReducibleNode   = {minimumNumberReducibleNode}')
    
    logger.info(f'     target_metric_min = {target_metric_min}')
    logger.info(f'     target_metric_max = {target_metric_max}')
    logger.info(f'     target_variable = {target_variable}')
    logger.info('     ----------------------------------------')
    logger.info(f'     _target_metric = {_target_metric}')
    logger.info('     ----------------------------------------')
    logger.info(f'     input_metrics = {input_metrics}')
    logger.info(f'     worker_count_name = {worker_count_name}')
    logger.info(f'     _outsource_metrics = {_outsource_metrics}')
    logger.info(f'     outsource_metrics = {outsource_metrics}')
    
    logger.info('-----------------------  advisor init end  ----------------------')



# ## ------------------------------------------------------------------------------------------------------
# ## Define init advice_msg
# ## ------------------------------------------------------------------------------------------------------
    
def advice_msg(valid = False, phase = 'training', vm_number = 0, nn_error_rate = 1000, error_msg = None):
    if valid:
        return jsonify(dict(valid = valid, phase = phase, vm_number = vm_number, nn_error_rate = nn_error_rate, error_msg = 'Def')), 200
    else:
        return jsonify(dict(valid = valid, phase = phase, vm_number = vm_number, nn_error_rate = nn_error_rate, error_msg = error_msg)), 400

    
# ## ------------------------------------------------------------------------------------------------------
# ## Define run
# ## ------------------------------------------------------------------------------------------------------

def run(csfFileName, vm_number_from_sample, target_variable_from_sample, last = False):
    
    # Set logger
    logger = logging.getLogger('optimizer')
    
    logger.info('---------------------------  opt_advisor.run()  ---------------------------')
    
    # Time
    current_time = int(time.time())
    logger.info(f'                   int(time.time() = {current_time}')
    
    
    


    # Set the default message False
    return_msg = advice_msg(valid = False, phase = 'invalid', error_msg = 'Default message')
  
    # Set showPlots True
    showPlots = True
    
    # Set showPlot False if we intrested in only the last value, in this case there is no reasaon to create plot
    if( last ):
        showPlots = False
        

    logger.info('     ----------------------------------------------')
    logger.info('     ------------ ADVICE PAHSE STARTED ------------')
    logger.info('     ----------------------------------------------')

    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Load models which were saved until the training pahse
    # ## ------------------------------------------------------------------------------------------------------
    
    X_normalized_MinMaxScaler = loadMinMaxScalerXFull()
    y_normalized_MinMaxScaler = loadMinMaxScalerYFull()
    modelNeuralNet = loadNeuralNetworkModel()
    
    logger.info('--------------------- MODELS LOADED ----------------------')


    # ## ------------------------------------------------------------------------------------------------------
    # ## Do not touch this if you don't know what you do
    # ## ------------------------------------------------------------------------------------------------------
    
    # targetVariable = 'avg latency (quantile 0.5)'
    targetVariable = target_variable
    # testFileName = 'data/grafana_data_export_long_running_test.csv'      # original data
    # testFileName = 'data/test_data.csv'                                  # test data
    testFileName = csfFileName                                             # from parameter
    
    maximumNumberIncreasableNode = 6                                       # must be positive 6
    minimumNumberReducibleNode = -6                                        # must be negativ -4
    
    upperLimit = target_metric_max                                         # 4000000
    lowerLimit = target_metric_min                                         # 1000000

    
    logger.info('----------------------------------------------------------')
    logger.info(f'targetVariable = {targetVariable}')
    logger.info('----------------------------------------------------------')
    
    logger.info('----------------------------------------------------------')
    logger.info(f'target_variable_from_sample = {target_variable_from_sample}')
    logger.info('----------------------------------------------------------')
    
    logger.info('----------------------------------------------------------')
    logger.info(f'vm_number_from_sample = {vm_number_from_sample}')
    logger.info('----------------------------------------------------------')
    
    logger.info('----------------------------------------------------------')
    logger.info(f'testFileName = {testFileName}')
    logger.info(f'csfFileName = {csfFileName}')
    logger.info('----------------------------------------------------------')
    
    logger.info('----------------------------------------------------------')
    logger.info(f'maximumNumberIncreasableNode = {maximumNumberIncreasableNode}')
    logger.info(f'minimumNumberReducibleNode = {minimumNumberReducibleNode}')
    logger.info('----------------------------------------------------------')

    logger.info('----------------------------------------------------------')
    logger.info(f'lowerLimit parameter variable set = {lowerLimit}')
    logger.info(f'upperLimit parameter variable set = {upperLimit}')
    logger.info('----------------------------------------------------------')
    
    
    # In[159]:

    df = readCSV(testFileName)

    logger.info('----------------------------------------------------------')
    logger.info('----------------------- DF LOADED ------------------------')
    logger.info(f'df.shape               = {df.shape}')
    logger.info('------------------------ ADVISOR -------------------------')
    logger.info('----------------------------------------------------------')
    

    # ## ------------------------------------------------------------------------------------------------------
    # ## If there is not enough data in dataframe then return error message
    # ## ------------------------------------------------------------------------------------------------------
    
    logger.info('----------------------------------------------------------')
    logger.info('----------- Checking advisor data properties -------------')
    if df.shape[0] <= 0:
        error_msg = 'There is no training sample yet.'
        logger.error(error_msg)
        return advice_msg(valid = False, phase = 'invalid', error_msg = error_msg)
    
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## If there is not enough data in dataframe then return error message
    # ## ------------------------------------------------------------------------------------------------------
    
    logger.info('----------------------------------------------------------')
    logger.info('----------- Checking advisor data properties -------------')
    # Todo:
    # Azért ezt itt be lehet szívni, ha mondjuk csak 30 körönként tanítunk
    # és a tanulás limitje 300, akkor lehet, hogy csak a 310 körben lesz
    # meg az első tanulás
    # ez a szerencsétlen viszont már 300 után keresni fogja a modelt
    # amit persze nem talál majd
    # +++++++ szóval ezt az értéket meg kell még növelni a körök számával +1 ++++++++++++
    # if df.shape[0] <= 300:
    if df.shape[0] < 300:
        error_msg = 'There are not enough training samples yet.'
        logger.error(error_msg)
        return advice_msg(valid = False, phase = 'invalid', error_msg = error_msg)

    
    # ## ------------------------------------------------------------------------------------------------------
    # ## If there is not enough data in dataframe then return error message this part wont run at all
    # ## ------------------------------------------------------------------------------------------------------
    
    
    logger.info('----------------------------------------------------------')
    logger.info('----------- Checking advisor data properties -------------')
    if df.shape[0] < 1:
        logger.info('----------------------------------------------------------')
        logger.info('------- There is no training sample at all. -------')
        logger.info(f'---------------- We have only {df.shape[0]} sample ----------------')
        error_msg = 'There is no training sample yet.'
        return advice_msg(valid = False, phase = 'invalid', error_msg = error_msg)

    
    # ## ------------------------------------------------------------------------------------------------------
    # ## If there is not enough data in dataframe then return error message this part wont run at all
    # ## ------------------------------------------------------------------------------------------------------
    

    if( last == True ):
        
        logger.info('----------------------------------------------------------')
        logger.info('--------- There are enough training samples yet ----------')
        logger.info('--------------- Last row will be processed ---------------')
        
        pf = df[-1:]
        
        logger.info('----------------------------------------------------------')
        logger.info(f'pf shape               = {pf.shape}')
        for m in pf.columns:
            logger.info(f'Column names are = {m}, {pf[m].values}')
        
        # Assigne pf to df -> keep the code more coherent
        df = pf.copy()
        


    # ## ------------------------------------------------------------------------------------------------------
    # ## preProcessing() comes from opt_utis modul
    # ## ------------------------------------------------------------------------------------------------------
    
    preProcessedDF = preProcessing(df)

    logger.info('----------------------------------------------------------')
    logger.info('------------------ preProcessing done --------------------')
    
    
    WorkerCountName = None
    if( preProcessedDF.columns.contains('Worker count') ):
        WorkerCountName = 'Worker count'
    elif( preProcessedDF.columns.contains('vm_number') ):
        WorkerCountName = 'vm_number'
    else:
        WorkerCountName = 'Worker count'

        
    # ## ------------------------------------------------------------------------------------------------------
    # ## Better if WorkerCountName comes from init()
    # ## ------------------------------------------------------------------------------------------------------
        
    WorkerCountName = worker_count_name
    
    logger.info('----------------------------------------------------------')
    logger.info(f'WorkerCountName         = {WorkerCountName}')
    logger.info('----------------------------------------------------------')

    
    # Rename Worker count or vm_number to WorkerCount
    renamedDF = renameVariable(preProcessedDF, WorkerCountName, 'WorkerCount')

    filteredDF = renamedDF
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Better if WorkerCountName comes from init()
    # ## ------------------------------------------------------------------------------------------------------
    logger.info('----------------------------------------------------------')
    logger.info('------------------- set metricNames ----------------------')
    logger.info('----------------------------------------------------------')
   
    # metricNames         = setMetricNames(['CPU', 'Inter', 'CTXSW', 'KBIn', 'PktIn', 'KBOut', 'PktOut'])
    # metricNames         = setMetricNames(input_variables[2:])
    # Változtatás
    # metricNames         = setMetricNames(input_variables)
    # extendedMetricNames = setExtendedMetricNames(['CPU', 'Inter', 'CTXSW', 'KBIn', 'PktIn', 'KBOut', 'PktOut', 'WorkerCount'])

    # na itt lesz majd az alapvető probléma az inputMetric és a metricNames között
    # ugyanis igazából nem tudjuk, hogy az első kettő az amit le kell vágnunk a listából
    # ezért dirty ha szóval itt kell leválogatnom, hogy
    # a metrics names-ben
    # dobjuk el azt a két fránya AVR_RR SUM_RR oszlopot 
    
    # Ezeeket az értékeket az init-ben adom át neki, annyi a különbség, hogy az első két változóra nincs szükségünk
    # metricNames = _input_metrics[2:]
    metricNames = input_variables.copy()

    for i in outsource_metrics:
        logger.info(f'removed elements of the metricNames list = {i}')
        metricNames.remove(i)
    
    logger.info(f'metricNames = {metricNames}')
    logger.info(f'input_variables = {input_variables}')
    
    logger.info('----------------------------------------------------------')
    

        
    logger.info('----------------------------------------------------------')
    logger.info(f'metricNames              = {metricNames}')
    # logger.info(f'extendedMetricNames      = {extendedMetricNames}')
    logger.info(f'input_variables          = {input_variables}')
    logger.info('----------------------------------------------------------')
    
    
    

    # >#### Add new workers (increse the nuber of added Worker)

    # In[162]:

    def calculatePredictedLatencyWithVariousWorkers(modelNeuralNet, to):
        
        logger.info('------------ calculatePredictedLatencyWithVariousWorkers STARTED -------------')
        
        newDFForRegression = filteredDF.copy()
        nDD = filteredDF.copy()
        
        step = 0

        if( to == 0 ):
            print("")
            assert to != 0,"This value can not be 0."
        elif( to > 0 ):
            step = 1
            print('............. up maximum vm = ' + str(to) + ' ...........')
        elif( to < 0 ):
            step = -1
            print('............. down maximum vm = ' + str(to) + ' ...........')
            
        logger.info(f' to = {to}')
        logger.info(f' step = {step}')

        for j in range(0, to, step):
            
            logger.info(f' j = {j}')

            addedWorkerCount = j
            
            logger.info(f' addedWorkerCount = {addedWorkerCount}')

            newDFForRegression['addedWorkerCount'] = addedWorkerCount
            
            # logger.info(f'newDFForRegression.columns = {newDFForRegression.columns}')

            for i in metricNames:
                
                # logger.info('------------- inner loop started -----------------')
                
                # logger.info(f' metricsNames = {metricNames}')
                
                # logger.info(f' i = {i}')

                newDFForRegressionWithTerms = calculateLinearRegressionTerms(i, newDFForRegression)
                
                # logger.info('-------------- calculateLinearRegressionTerms STARTED -------------')

                # keep last three column - given metric, term1, term2
                X = newDFForRegressionWithTerms.iloc[:, [-3, -2, -1]]
                
                # logger.info('-------------- X Features Generated STARTED -------------')

                # load the proper current metric model
                modelForMetric = joblib.load('models/saved_linearregression_model_' + i + '.pkl')
                

                # print("------------     ", modelForMetric.get_params(), "     ------------")

                if( np.isinf(X).any()[1] ):
                    X['term1'] = np.where(np.isinf(X['term1'].values), X['metric'], X['term1'])
                    X['term2'] = np.where(np.isinf(X['term2'].values), 0, X['term2'])

                # create prediction and store in a new numpy.array object
                predictedMetric = modelForMetric.predict(X)


                # leave original metric value (just for fun and investigation) and store in a new column
                newDFForRegression['original' + i] = newDFForRegression[i]

                # store predicted value pretend as would be the original. for example predictedCPU will be CPU
                newDFForRegression[i] = predictedMetric
                nDD[i] = predictedMetric
                

                # print out the new data frame
                # newDFForRegression.head()


            newDFForNerualNetworkPrediction = newDFForRegression.copy()     

            # X must contain exactly the same columns as the model does
            X = newDFForNerualNetworkPrediction.iloc[:, :len(input_variables)]
            # print('11111ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
            # print(newDFForNerualNetworkPrediction.columns)
            # print(newDFForNerualNetworkPrediction.shape)
            # print(newDFForNerualNetworkPrediction.head())
            # print('22222ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
            # print(X.columns)
            # print(X.shape)
            # print(X.head())
            # print('33333ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
            # print(input_variables)
            # print(len(input_variables))
            # print('44444ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')


            # X must be normalized based on a previously created MinMaxScaler
            X_normalized_MinMaxScaler # the name of the MinMaxScaler

            X_normalized = X_normalized_MinMaxScaler.transform(X)

            #!!! insted of declare the location of the model here I use loadNeuralNetworkModel() methond inside the run() m.
            #!!!
            #!!!
            # modelNeuralNet = joblib.load('models/saved_mlp_model.pkl')
            modelNeuralNet = modelNeuralNet

            # create and store predicted values in a numpy.array object
            y_predicted_with_new_metrics = modelNeuralNet.predict(X_normalized)

            # denormalized predicted values
            y_predicted_with_new_metrics_denormalized = y_normalized_MinMaxScaler.inverse_transform(y_predicted_with_new_metrics.reshape(y_predicted_with_new_metrics.shape[0],1))

            newDFForNerualNetworkPrediction['predictedResponseTimeAdded' + str(j) + 'Worker'] = y_predicted_with_new_metrics
            newDFForNerualNetworkPrediction['denormalizedPredictedResponseTimeAdded' + str(j) + 'Worker'] = y_predicted_with_new_metrics_denormalized

            if(j == 0):
                investigationDF = newDFForNerualNetworkPrediction[[targetVariable, 'WorkerCount']]
                investigationDFDeNormalized = newDFForNerualNetworkPrediction[[targetVariable, 'WorkerCount']]
                #investigationDF = newDFForNerualNetworkPrediction[['predictedResponseTimeAdded0Worker']]
                #investigationDFDeNormalized = newDFForNerualNetworkPrediction[['denormalizedPredictedResponseTimeAdded0Worker']]

                
            ## itt kéne egy ellenőrzést csinálni, hogy ezt a módszert miért kifogásolja a panda
            ## van egy új df-em aminek az egyik oszlopát egyelővé akarom tenni egy másik df másik oszlopával
            ## le kéne ellenőrizni, hogy milyen változ
            investigationDF['predictedResponseTimeAdded' + str(j) + 'Worker'] = newDFForNerualNetworkPrediction[['predictedResponseTimeAdded' + str(j) + 'Worker']]
            investigationDFDeNormalized['denormalizedPredictedResponseTimeAdded' + str(j) + 'Worker'] = newDFForNerualNetworkPrediction[['denormalizedPredictedResponseTimeAdded' + str(j) + 'Worker']]

        return investigationDF, investigationDFDeNormalized


    # In[163]:

    investigationDFUp, investigationDFDeNormalizedUp = calculatePredictedLatencyWithVariousWorkers(modelNeuralNet, maximumNumberIncreasableNode)


    # In[164]:

    investigationDFDown, investigationDFDeNormalizedDown = calculatePredictedLatencyWithVariousWorkers(modelNeuralNet, minimumNumberReducibleNode)



    # ### Merge Up and Down Adviser

    # In[165]:

    print('Error--------------------------------------------------------------------------------------------')
    print('Mi a fenéért dobja el a változókat amikor az "investigationDFDeNormalizedDown" és a "investigationDFDeNormalizedUp"-ban')
    print('is más változók vannak')
    print('Ha konstans értékek vannak minden változóban akkor a drop_duplicates().T miatt dobja őket')

    investigationDeNormalizedDF = pd.concat([investigationDFDeNormalizedDown,
                                             investigationDFDeNormalizedUp], axis = 1).T.drop_duplicates().T


        
    # ## ------------------------------------------------------------------------------------------------------
    # ## Get Advice
    # ## ------------------------------------------------------------------------------------------------------

    # In[180]:

    logger.info('')
    logger.info('----------------------------------------------------------')
    logger.info('----------             Get Advice             ------------')
    logger.info('----------------------------------------------------------')
    
    
    logger.info('------ Get Actual Number of WorkerCount based on investigationDeNormalizedDF ------')
    
    # itt eldönthetem, hogy a dataframeből olvasom ki ezt az adatot, vagy paraméterként veszem át
    actual_worker_number = investigationDeNormalizedDF[['WorkerCount']].get_value(investigationDeNormalizedDF.index[0], 'WorkerCount')
    # másfelől lehet, hogy ezt az értéket az épen aktuális mintából kéne kivennem?!
    
    
    advice = 0
    advicedVM = 0
    advicedVM = actual_worker_number       # alapvetően beállithatóm, hogy a sample df-ben tárol vm legyen az aktuális
    countInRange = 0
    countViolatedUp = 0
    countViolatedDown = 0
    # actual_worker_number = 0
    # nem ezt már kiolvastam a sample df-ből (persze lehet, hogy az épen aktuális már nem ez)
    
    
    
    logger.info(f'  actual_worker_number                    = {actual_worker_number}')
    logger.info(f'  investigationDeNormalizedDF.index       = {investigationDeNormalizedDF.index}')
    logger.info(f'  type(investigationDeNormalizedDF.index) = {type(investigationDeNormalizedDF.index)}')
    logger.info(f'  investigationDeNormalizedDF.index[0]    = {investigationDeNormalizedDF.index[0]}')
    logger.info(f'  investigationDeNormalizedDF.shape       = {investigationDeNormalizedDF.shape}')
    logger.info('')
    logger.info(f'  investigationDeNormalizedDF.columns     = {investigationDeNormalizedDF.columns}')
    logger.info('-----------------------------------------------------------------------------------')

    logger.info('-----------------------------------------------------------------------------------')
    logger.info('                       init the advice storage data frame                          ')
    logger.info('-----------------------------------------------------------------------------------')
    
    advicedDF = investigationDeNormalizedDF.copy()
    advicedDF['advice'] = advice # az advice alapértelmezetben 0 tehát ez a rész a df-ben csupa 0 lesz
    advicedDF['postScaledTargetVariable'] = np.nan
    advicedDF['advicedVM'] = advicedVM
    # advicedDF['advicedVM'] = 0 # 0 helyett legyen a ténylegesen vagy annak vélt kiolvasott vm szám
    
    logger.info('')
    logger.info('-----------------------------------------------------------------------------------')
    logger.info(f'   advicedDF.shape   = {advicedDF.shape}')
    logger.info(f'   advicedDF.columns = {advicedDF.columns}')
    logger.info('-----------------------------------------------------------------------------------')
    
    
    logger.info('')
    logger.info('-----------------------------------------------------------------------------------')
    logger.info('               iterate thorough and try to find best candidate                     ')
    logger.info('-----------------------------------------------------------------------------------')

    for i in investigationDeNormalizedDF.index:
        logger.info('')
        logger.info('    This is the current case number of index in df what we operate on          ')
        logger.info(f'   investigationDeNormalizedDF.index = i = {i}')
        logger.info(f'   investigationDeNormalizedDF.index     = {investigationDeNormalizedDF.index}')
        logger.info('')
        logger.info('-----------------------------------------------------------------------------------')
        logger.info('                     iterate thorough possible solutions                           ')
        logger.info('-----------------------------------------------------------------------------------')
            
        calculatedDistance = 99999999999
        real = investigationDeNormalizedDF[[targetVariable]].get_value(i, targetVariable)
        
        # Tehát a problémám az, hogy ha nem jön be minta, vagy hiányos a minta ezért nem tudjuk
        # elfogadni, akkor a csv-ben és a csv-ből kiolvasott utolsó sor valami akár egészen
        # régi értéket is tud mutatni, akár a target változóra, amit fölébb 'real' néven
        # nevezek, akár a 'vm_count' a vm számára vonatkozóan.
        #
        # Ezért a részemről felmerült annak az igénye, hogy ezt a két értéket, ha úgy is
        # megkapjuk a mintával akkor onnan vett értéket olvasssuk ki.
        #
        # Nézzük meg, hogy egyáltalán lehetséges-e ez
        # Elvileg az 'opt_rest' 'sample' api kapja meg az adatokat
        # az advisor csak az eltárolt 'csv'-ből olvas ki bármit is
        
        
        
        logger.info('------------------------------------------------------')
        logger.info(f'    real target variable from dataframe = {real}')
        logger.info('------------------------------------------------------')
        
        logger.info('------------------------------------------------------')
        logger.info(f'    target_variable_from_sample = {target_variable_from_sample}')
        logger.info(f'    type = {type(target_variable_from_sample)}')
        logger.info(f'    target_variable_from_sample[0] = {target_variable_from_sample[0]}')
        logger.info('------------------------------------------------------')
        
        real = target_variable_from_sample[0]
        
        
        if( upperLimit > real and lowerLimit < real ):
            advice = 0
            actual_worker_number = investigationDeNormalizedDF[['WorkerCount']].get_value(i, 'WorkerCount')
            # advicedVM = investigationDeNormalizedDF[['WorkerCount']].get_value(i, 'WorkerCount')
            advicedVM = actual_worker_number
            # Ne a javaslatot, hanem a konkrét gép számot adja vissza
            advicedDF.ix[i,'advice'] = 0
            # advicedDF.ix[i,'advice'] = investigationDeNormalizedDF[['WorkerCount']]
            countInRange += 1
            logger.info('ok - target variable is in range           ')
        else:
            logger.info(f'threshold violation at index {str(i)}')
            if( upperLimit < real ):
                countViolatedUp += 1
                logger.info('threshold up violation')
                postScaledTargetVariable = np.nan # 0
                distance = float('inf')
                # for j in range(1, maximumNumberIncreasableNode):
                for j in range(0, maximumNumberIncreasableNode):
                    print('j = ', j)
                    # két feltételnek kell megfelelnie sorrendben legyen a legkisebb távolsága a felső limittől
                    # kettő legyen a felső limit alatt (utóbbi nem biztos, hogy teljesül)
                    varName = 'denormalizedPredictedResponseTimeAdded' + str(j) + 'Worker'
                    print('varName = ', varName)
                    relatedTargetVariable = investigationDeNormalizedDF.get_value(i, varName)
                    print('relatedTargetVariable = ', relatedTargetVariable)
                    calculatedDistance = abs(investigationDeNormalizedDF.get_value(i, varName) - upperLimit)
                    print('calculatedDistance = ', calculatedDistance)
                    print('distance = ', distance)
                    if( calculatedDistance < distance ):
                        distance = calculatedDistance
                        advice = j
                        postScaledTargetVariable = relatedTargetVariable
                    if( relatedTargetVariable < upperLimit ):
                        distance = calculatedDistance
                        print('distance = ', distance)
                        advice = j
                        postScaledTargetVariable = relatedTargetVariable
                        break
                   
                advicedVM = actual_worker_number + advice
                print('')
                print('advicedVM = ', advicedVM)
                print('lowest distance = ', distance)
                print('chosen advice = ', advice)
                print('postScaledTargetVariable = ', postScaledTargetVariable)
                print('')
                    
                advicedDF.ix[i,'advice'] = advice
                advicedDF.ix[i, 'postScaledTargetVariable'] = postScaledTargetVariable
            elif( lowerLimit > real ):
                countViolatedDown += 1
                logger.info('threshold down violation')
                postScaledTargetVariable = np.nan # 0
                distance = float('-inf')
                # for j in range(-1, -3, -1):
                for j in range(-1, minimumNumberReducibleNode, -1):
                    print('j = ', j)
                    # két feltételnek kell megfelelnie sorrendben legyen a legkisebb távolsága az alsó limittől
                    # kettő legyen az alsó limit fölött (utóbbi nem biztos, hogy teljesül)
                    varName = 'denormalizedPredictedResponseTimeAdded' + str(j) + 'Worker'
                    print('varName = ', varName)
                    relatedTargetVariable = investigationDeNormalizedDF.get_value(i, varName)
                    print('relatedTargetVariable = ', relatedTargetVariable)
                    calculatedDistance = abs(investigationDeNormalizedDF.get_value(i, varName) - lowerLimit)
                    print('calculatedDistance = ', calculatedDistance)
                    print('distance = ', distance)
                    if( calculatedDistance > distance ):
                        distance = calculatedDistance
                        advice = j
                        postScaledTargetVariable = relatedTargetVariable
                    if( relatedTargetVariable > lowerLimit ):
                        distance = calculatedDistance
                        print('distance = ', distance)
                        advice = j
                        postScaledTargetVariable = relatedTargetVariable
                        break
                
                advicedVM = actual_worker_number + advice
                print('')
                print('advicedVM = ', advicedVM)
                print('lowest distance = ', distance)
                print('chosen advice = ', advice)
                print('postScaledTargetVariable = ', postScaledTargetVariable)
                print('')
                advicedDF.ix[i, 'advice'] = advice
                advicedDF.ix[i, 'postScaledTargetVariable'] = postScaledTargetVariable
                # TODO
                # Ide írni még egy elif ágat, hogy nem talált jó megoldást
                # Egyelőre ilyen esetben legyen az, hogy hagyja a VM számot ott ahol van
                # Vagy majd kigondolok valami jobbat
                # Esetleg azt, hogy mi az ami a legközelebb áll az alsó vagy felső limithez
                # Attól függően, hogy melyik ágon futunk le



    # In[183]:
    logger.info(f'countInRange      = {countInRange}')
    logger.info(f'countViolatedDown = {countViolatedDown}')
    logger.info(f'countVilolatedUp  = {countViolatedUp}')


    # In[188]:
    if( last == False ):
        advicedDF.to_csv('outputs/adviseDF.csv', sep=';', encoding='utf-8')
        logger.info('outputs/adviseDF.csv saved')
    
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Set return_msg
    # ## ------------------------------------------------------------------------------------------------------
    
    phase = 'production'
    nn_error_rate = 0
    # Ez volt az egyéni javaslat, hogy mennyit adjon hozzá
    # vm_number_total = advice
    # Ez a konkrét javaslat, hogy hány gépnek kell szerepelnie
    vm_number_total = advicedVM
    output_filename = config.output_filename
    output_filename = config.get_property('output_filename')
    
    logger.info('----------------------------------------------')
    logger.info(f'advice = {advice}')
    logger.info(f'actual_worker_number = {actual_worker_number}')
    logger.info(f'vm_number_total = {vm_number_total}')
    logger.info(f'output_filename = {output_filename}')
    logger.info('----------------------------------------------')
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Set return_msg          the rest of the code is for report and persist data
    # ## ------------------------------------------------------------------------------------------------------
    
    current_time = int(time.time())
    global first_advice
    global prev_adviced_time
    global prev_advice_vm_total_number
    logger.info('---------------------------------------------------------------------------')
    logger.info(f'  first_advice = {first_advice}')
    logger.info(f'  prev_adviced_time = {prev_adviced_time}')
    logger.info(f'  current_time      = {current_time}')
    logger.info(f'  ellapsed time     = {current_time - prev_adviced_time}')
    logger.info(f'  prev_advice_vm_total_number = {prev_advice_vm_total_number}')
    logger.info(f'  vm_number_total = {vm_number_total}')
    logger.info('---------------------------------------------------------------------------')
    if( current_time - prev_adviced_time > 30 ):

        prev_adviced_time = current_time
        prev_advice_vm_total_number = vm_number_total
        first_advice = False
        
        logger.info('-------------------------  opt_advisor.run() run  -------------------------')
        logger.info(f'  first_advice = {first_advice}')
        logger.info(f'  prev_adviced_time = {prev_adviced_time}')
        logger.info(f'  prev_advice_vm_total_number = {prev_advice_vm_total_number}')
        logger.info(f'  vm_number_total = {vm_number_total}')
        logger.info('---------------------------------------------------------------------------')
    else:
        vm_number_total = prev_advice_vm_total_number
        logger.info('---------------------------------------------------------------------------')
        logger.info(f'  vm_number_total = {vm_number_total}')
        logger.info('---------------------------------------------------------------------------')
    
    
    return_msg = advice_msg(valid = True, phase = phase, vm_number = vm_number_total, nn_error_rate = nn_error_rate)
        
    # ## ------------------------------------------------------------------------------------------------------
    # ## Store and save metrics, advice and predictions in a csv file
    # ## ------------------------------------------------------------------------------------------------------
    
    logger.info('----------------------------------------------------------')
    logger.info(f'pf shape               = {pf.shape}')
    # for m in pf.columns:
        # logger.info(f'Column names are = {m}, {pf[m].values}')

    logger.info('----------------------------------------------------------')
    logger.info('                store adviced data csv                    ')
    logger.info('----------------------------------------------------------')
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Prepare dataframes to store metrics, advice and predictions
    # ## ------------------------------------------------------------------------------------------------------
    
    tf = pf.copy() # original csv stored input datas

    # Merge data frames
    tmp_columns = investigationDFDeNormalizedUp.columns[2:]
    tf = pd.merge(tf, investigationDFDeNormalizedUp[tmp_columns], right_index=True, left_index=True)
    
    # Merge data frames
    tmp_columns = investigationDFDeNormalizedDown.columns[2:]
    tf = pd.merge(tf, investigationDFDeNormalizedDown[tmp_columns], right_index=True, left_index=True)
    
    
    # ## ------------------------------------------------------------------------------------------------------
    # ## Store data frame in a csv file
    # ## ------------------------------------------------------------------------------------------------------
    
    # if advisedDF.csv file exists
    
    if(os.path.isfile(output_filename) == True):
        logger.info('----------------------------------------------------------')
        # read
        af = readCSV(output_filename)
        logger.info('      --------- read existing csv ---------    ')
        logger.info(f'      csv file name = {output_filename}')
        logger.info(f'      csv datafraeme shape = {af.shape}')
        
        # add advised vm_number
        tf['advised_vm_number'] = vm_number_total
        tf['post_scaled_target_variable'] = advicedDF['postScaledTargetVariable'].get_value(i, 'postScaledTargetVariable')
        
        logger.info('      --------- inner state ---------')
        logger.info(f'      inner state df shape = {tf.shape}')
        
        # append
        bf = af.copy()
        bf = bf.append(tf.copy(), ignore_index=True)
        
        logger.info('      --------- append new data ---------')
        logger.info(f'      appended df shape = {bf.shape}')
        
        logger.info('---------------- end of appending advice -----------------')
        logger.info('----------------------------------------------------------')

        logger.info('--------------------- save to csv ------------------------')
        # save
        bf.to_csv(output_filename, sep=',', encoding='utf-8', index=False)
        logger.info('------------- advice saved into csv file -----------------')
        
        
        # ## ------------------------------------------------------------------------------------------------------
        # ## Generate Report
        # ## ------------------------------------------------------------------------------------------------------
        min_threshold = constants.get('target_metrics')[0].get('min_threshold')
        max_threshold = constants.get('target_metrics')[0].get('max_threshold')
        
        # print(bf.columns)
        generate_report(bf, min_threshold, max_threshold)
        
        # ## ------------------------------------------------------------------------------------------------------
        # ## Generate Report End
        # ## ------------------------------------------------------------------------------------------------------
        
    # if advisedDF.csv file does not exist
    
    if(os.path.isfile(output_filename) != True):
        logger.info('------------- create new advice csv file -----------------')
        nf = tf.copy()
        nf['advised_vm_number'] = vm_number_total
        nf['post_scaled_target_variable'] = advicedDF['postScaledTargetVariable'].get_value(i, 'postScaledTargetVariable')
        # save
        nf.to_csv(output_filename, sep=',', encoding='utf-8', index=False)
        logger.info('------------- advice saved into csv file -----------------')
    
    return return_msg
    
    
# ## ------------------------------------------------------------------------------------------------------
# ## Generate report method
# ## ------------------------------------------------------------------------------------------------------

def generate_report(df, min_threshold, max_threshold):
    
    VisualizePredictedXY2Line(df[[target_variable]], df[['advised_vm_number']], target_variable, min_threshold, max_threshold)
    
    VisualizePredictedXY3Line(df[[target_variable]], \
                              df[['post_scaled_target_variable']], \
                              df[['advised_vm_number']], target_variable, min_threshold, max_threshold)
    
    # r = re.compile('.*0*.')
    r = re.compile('.*denormalized*.')
    visualised_columns = list(filter(r.match, df.columns))
    VisualizePredictedYLine(df[target_variable], df[visualised_columns], target_variable)
    
    r = re.compile('.*denormalized*.')
    visualised_columns = list(filter(r.match, df.columns))
    VisualizePredictedYWithWorkers(0, df[visualised_columns], target_variable)
    
    
