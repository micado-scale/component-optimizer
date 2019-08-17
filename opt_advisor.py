import logging
import logging.config

import os

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
from visualizerlinux import VisualizePredictedYLine, VisualizePredictedYWithWorkers
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


# ## ------------------------------------------------------------------------------------------------------
# ## Define init method
# ## ------------------------------------------------------------------------------------------------------
    
def init(_target_metric, input_metrics, worker_count, _outsource_metrics, _config, _constants):
    
    logger = logging.getLogger('optimizer')
    
    logger.info('')
    logger.info('----------------------- advisor init ----------------------')
    
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

    logger.info('     ----------------------------------------------')
    logger.info('     ---------- ADVISOR INIT DIAGNOSIS ------------')
    logger.info('     ----------------------------------------------')
    
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
    
    logger.info('----------------------- advisor init end ----------------------')



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

def run(csfFileName, last = False):
    

    # Set the default message False
    return_msg = advice_msg(valid = False, phase = 'invalid', error_msg = 'Default message')
  
    # Set showPlots True
    showPlots = True
    
    # Set showPlot False if we intrested in only the last value, in this case there is no reasaon to create plot
    if( last ):
        showPlots = False
        

    # Set logger
    logger = logging.getLogger('optimizer')
    
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
    minimumNumberReducibleNode = -4                                        # must be negativ -4
    
    upperLimit = target_metric_max                                         # 4000000
    lowerLimit = target_metric_min                                         # 1000000

    
    logger.info('----------------------------------------------------------')
    logger.info(f'targetVariable = {targetVariable}')
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
    # 333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
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
    # 333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
    

        
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
            
            logger.info(f'newDFForRegression.columns = {newDFForRegression.columns}')

            for i in metricNames:
                
                logger.info('------------- inner loop started -----------------')
                
                logger.info(f' metricsNames = {metricNames}')
                
                logger.info(f' i = {i}')

                newDFForRegressionWithTerms = calculateLinearRegressionTerms(i, newDFForRegression)
                
                logger.info('-------------- calculateLinearRegressionTerms STARTED -------------')

                # keep last three column - given metric, term1, term2
                X = newDFForRegressionWithTerms.iloc[:, [-3, -2, -1]]
                
                logger.info('-------------- X Features Generated STARTED -------------')

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
                newDFForRegression.head()


            newDFForNerualNetworkPrediction = newDFForRegression.copy()     

            # X must contain exactly the same columns as the model does
            X = newDFForNerualNetworkPrediction.iloc[:, :9]

            # X must be normalized based on a previously created MinMaxScaler
            X_normalized_MinMaxScaler # the name of the MinMaxScaler

            X_normalized = X_normalized_MinMaxScaler.transform(X)

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


    if( 10 > 1 ):
        print('------------------------------------------------------')
        print('investigationDeNormalizedDF.values.shape')
        print(investigationDeNormalizedDF.values.shape)
        print('------------------------------------------------------')

        print('------------------------------------------------------')
        print('investigationDFDeNormalizedDown.values.shape')
        print(investigationDFDeNormalizedDown.values.shape)
        print('------------------------------------------------------')

        print('------------------------------------------------------')
        print('investigationDFDeNormalizedUp.values.shape')
        print(investigationDFDeNormalizedUp.values.shape)
        print('------------------------------------------------------')

        print('------------------------------------------------------')
        print('investigationDFUp.values.shape')
        print(investigationDFUp.values.shape)
        print('------------------------------------------------------')

        print('------------------------------------------------------')
        print('investigationDFDown.values.shape')
        print(investigationDFDown.values.shape)
        print('------------------------------------------------------')

        print('------------------------------------------------------')
        print('investigationDFDeNormalizedUp.head(2)')
        print(investigationDFDeNormalizedUp.head(2))
        print('------------------------------------------------------')

        print('------------------------------------------------------')
        print('investigationDFDeNormalizedDown.head(2)')
        print(investigationDFDeNormalizedDown.head(2))
        print('------------------------------------------------------')


    # In[171]:

    if showPlots :
        visualisedData = investigationDFDown.columns[2:]
        VisualizePredictedYWithWorkers(0, investigationDFDown[visualisedData], targetVariable)


    # In[172]:

    if showPlots :
        visualisedColumns = investigationDFUp.columns[2:]
        VisualizePredictedYWithWorkers(0, investigationDFUp[visualisedColumns], targetVariable)


    # In[174]:

    if showPlots :
        visualisedColumns = investigationDFDeNormalizedUp.columns[2:]
        VisualizePredictedYWithWorkers(0, investigationDFDeNormalizedUp[visualisedColumns], targetVariable)


    # In[175]:

    if showPlots :
        visualisedColumns = investigationDFDeNormalizedUp.columns[2:]
        VisualizePredictedYLine(investigationDFDeNormalizedUp['avg latency (quantile 0.5)'],                         investigationDFDeNormalizedUp[visualisedColumns], targetVariable)


    # In[176]:

    if showPlots :
        VisualizePredictedYLine(investigationDFDeNormalizedUp[[targetVariable]],                         investigationDFDeNormalizedUp[['denormalizedPredictedResponseTimeAdded0Worker']], targetVariable)


    # In[177]:

    if showPlots :
        VisualizePredictedYLine(investigationDFDeNormalizedUp[[targetVariable]],                         investigationDFDeNormalizedUp[['denormalizedPredictedResponseTimeAdded0Worker']], targetVariable)


    # In[179]:
    
    if showPlots :
        from visualizerlinux import VisualizePredictedXYLine
        from visualizerlinux import VisualizePredictedXY2Line
        VisualizePredictedXYLine(0, investigationDFDeNormalizedUp[[targetVariable]], targetVariable, lowerLimit, upperLimit)

        
        
        
    # ## ------------------------------------------------------------------------------------------------------
    # ## Get Advice
    # ## ------------------------------------------------------------------------------------------------------

    # In[180]:

    advice = 0
    advicedVM = 0
    countInRange = 0
    countViolatedUp = 0
    countViolatedDown = 0
    actual_worker_number = 0

    advicedDF = investigationDeNormalizedDF.copy()
    advicedDF['advice'] = 0
    advicedDF['postScaledTargetVariable'] = np.nan
    advicedDF['advicedVM'] = 0
    
    logger.info('post advice init')
    
    # print('------------------------------------------------------')
    # print('investigationDeNormalizedDF.columns')
    # print(investigationDeNormalizedDF.columns)
    # print('------------------------------------------------------')

    for i in investigationDeNormalizedDF.index:
        distance = 99999999999
        calculatedDistance = 99999999999
        real = investigationDeNormalizedDF[[targetVariable]].get_value(i, targetVariable)
        if( upperLimit > real and lowerLimit < real ):
            advice = 0
            actual_worker_number = investigationDeNormalizedDF[['WorkerCount']].get_value(i, 'WorkerCount')
            # advicedVM = investigationDeNormalizedDF[['WorkerCount']].get_value(i, 'WorkerCount')
            advicedVM = actual_worker_number
            # Ne a javaslatot, hanem a konkrét gép számot adja vissza
            advicedDF.ix[i,'advice'] = 0
            # advicedDF.ix[i,'advice'] = investigationDeNormalizedDF[['WorkerCount']]
            countInRange += 1
            print("ok")
        else:
            print("threshold violation at index " + str(i))
            if( upperLimit < real ):
                countViolatedUp += 1
                # print("threshold up violation")
                advice = 0
                actual_worker_number = investigationDeNormalizedDF[['WorkerCount']].get_value(i, 'WorkerCount')
                postScaledTargetVariable = np.nan # 0
                distance = float('inf')
                for j in range(1, maximumNumberIncreasableNode):
                    # print(distance)
                    advice = 0
                    # két feltételnek kell megfelelnie sorrendben legyen a legkisebb távolsága a felső limittől
                    # kettő legyen a felső limit alatt (utóbbi nem biztos, hogy teljesül)
                    varName = 'denormalizedPredictedResponseTimeAdded' + str(j) + 'Worker'
                    relatedTargetVariable = investigationDeNormalizedDF.get_value(i, varName)
                    calculatedDistance = investigationDeNormalizedDF.get_value(i, varName)
                    if( calculatedDistance < upperLimit ):
                        distance = calculatedDistance
                        advice = j
                        advicedVM = actual_worker_number + advice
                        postScaledTargetVariable = relatedTargetVariable
                        break
                    # print(calculatedDistance)
                advicedDF.ix[i,'advice'] = advice
                advicedDF.ix[i, 'postScaledTargetVariable'] = postScaledTargetVariable
            elif( lowerLimit > real ):
                countViolatedDown += 1
                print("threshold down violation")
                advice = 0
                actual_worker_number = investigationDeNormalizedDF[['WorkerCount']].get_value(i, 'WorkerCount')
                postScaledTargetVariable = np.nan # 0
                distance = float('-inf')
                # TODO:
                # Change to for j in range (-1, minimumNumberReducibleNode, -1):
                # for j in range(-1, -3, -1):
                for j in range(-1, minimumNumberReducibleNode, -1):
                    # print(distance)
                    advice = 0
                    actual_worker_number = investigationDeNormalizedDF[['WorkerCount']].get_value(i, 'WorkerCount')
                    # két feltételnek kell megfelelnie sorrendben legyen a legkisebb távolsága az alsó limittől
                    # kettő legyen az alsó limit fölött (utóbbi nem biztos, hogy teljesül)
                    varName = 'denormalizedPredictedResponseTimeAdded' + str(j) + 'Worker'
                    print(varName)
                    # print('Error-------------nincs benne egy csomo oszlop-----------------------------------------------')
                    # print(investigationDeNormalizedDF.columns)
                    relatedTargetVariable = investigationDeNormalizedDF.get_value(i, varName)
                    # print('Error----------------------------------------------------------------------------------------')
                    calculateDistance = investigationDeNormalizedDF.get_value(i, varName)
                    if( calculateDistance > lowerLimit ):
                        distance = calculateDistance
                        advice = j
                        advicedVM = actual_worker_number + advice
                        postScaledTargetVariable = relatedTargetVariable
                        if( calculateDistance < upperLimit ):
                            distance = calculateDistance
                            advice = j
                            advicedVM = actual_worker_number + advice
                            postScaledTargetVariable = relatedTargetVariable
                            break
                        # break
                    # print(calculateDistance)
                advicedDF.ix[i, 'advice'] = advice
                advicedDF.ix[i, 'postScaledTargetVariable'] = postScaledTargetVariable
                # TODO
                # Ide írni még egy elif ágat, hogy nem talált jó megoldást
                # Egyelőre ilyen esetben legyen az, hogy hagyja a VM számot ott ahol van
                # Vagy majd kigondolok valami jobbat
                # Esetleg azt, hogy mi az ami a legközelebb áll az alsó vagy felső limithez
                # Attól függően, hogy melyik ágon futunk le


    # In[181]:
    # advicedDF.head(10).style.set_properties(**pandas_dataframe_styles).format("{:0.0f}")


    # In[182]:
    if showPlots :
        VisualizePredictedXYLine(advicedDF[['advice']] * 2000000, advicedDF[[targetVariable]], targetVariable, lowerLimit, upperLimit)


    # In[183]:
    print('countInRange      = ', countInRange)
    print('countViolatedDown = ', countViolatedDown)
    print('countVilolatedUp  = ', countViolatedUp)


    # In[184]:
    if showPlots :
        VisualizePredictedXY2Line(advicedDF[[targetVariable]], advicedDF[['advice']], targetVariable, lowerLimit, upperLimit)


    # In[186]:
    if showPlots :
        from visualizerlinux import VisualizePredictedXY3Line
        
        VisualizePredictedXY3Line(advicedDF[[targetVariable]],advicedDF[['postScaledTargetVariable']],advicedDF[['advice']],targetVariable, lowerLimit, upperLimit)


    # In[187]:
    # advicedDF.style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


    # In[188]:

    if( last == False ):
        advicedDF.to_csv('outputs/adviseDF.csv', sep=';', encoding='utf-8')
        logger.info('outputs/adviseDF.csv saved')
    
    
    # In[x]:
    
    phase = 'production'
    nn_error_rate = 0
    # Ez volt az egyéni javaslat, hogy mennyit adjon hozzá
    # vm_number_total = advice
    # Ez a konkrét javaslat, hogy hány gépnek kell szerepelnie
    vm_number_total = advicedVM
    
    logger.info('----------------------------------------------')
    logger.info(f'advice = {advice}')
    logger.info(f'actual_worker_number = {actual_worker_number}')
    logger.info(f'vm_number_total = {vm_number_total}')
    logger.info('----------------------------------------------')
    
    
    return_msg = advice_msg(valid = True, phase = phase, vm_number = vm_number_total, nn_error_rate = nn_error_rate)
    
    # TODO
    # ki kell íratnom egy változóba azt az értéket is az advisorban, hogy milyen válaszidőt prediktált volna
    # és ezt az értéket is hozzá kell írnom
    
    output_filename = config.output_filename
    output_filename = config.get_property('output_filename')
    
    logger.info('----------------------------------------------------------')
    logger.info(f'pf shape               = {pf.shape}')
    # for m in pf.columns:
        # logger.info(f'Column names are = {m}, {pf[m].values}')

    logger.info('----------------------------------------------------------')
    logger.info('                store adviced data csv                    ')
    logger.info('----------------------------------------------------------')
    
    tf = pf.copy() # original csv stored input datas
    
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

        logger.info('      --------- inner state ---------')
        logger.info(f'      inner state df shape = {tf.shape}')
        
        # appendálja
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
        
    # if advisedDF.csv file does not exist
    
    if(os.path.isfile(output_filename) != True):
        logger.info('------------- create new advice csv file -----------------')
        nf = tf.copy()
        nf['advised_vm_number'] = vm_number_total
        # save
        nf.to_csv(output_filename, sep=',', encoding='utf-8', index=False)
        logger.info('------------- advice saved into csv file -----------------')
    
    return return_msg
    
