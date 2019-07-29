import logging
import logging.config

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



# TODO:
# Csinálni egy init-et ahol beállítom a változókat

def advice_msg(valid = False, phase = 'training', vm_number = 0, nn_error_rate = 1000, error_msg = None):
    if valid:
        return jsonify(dict(valid = valid, phase = phase, vm_number = vm_number, nn_error_rate = nn_error_rate, error_msg = 'Def')), 200
    else:
        return jsonify(dict(valid = valid, phase = phase, vm_number = vm_number, nn_error_rate = nn_error_rate, error_msg = error_msg)), 400


def run(csfFileName, last = False):

    return_msg = advice_msg(valid = False, phase = 'invalid', error_msg = 'Default message')
    
    showPlots = True
    
    if( last ):
        showPlots = False
        

    # # Advice Phase - Production Phase

    logger = logging.getLogger('optimizer')
    
    logger.info('-------------------------- Advice Phase --------------------------')

    # In[157]:

    X_normalized_MinMaxScaler = loadMinMaxScalerXFull()
    y_normalized_MinMaxScaler = loadMinMaxScalerYFull()

    modelNeuralNet = loadNeuralNetworkModel()


    # In[158]:

    cutFirstCases = 0                                                      # 0
    targetVariable = 'avg latency (quantile 0.5)'
    # testFileName = 'data/grafana_data_export_long_running_test.csv'      # original data
    # testFileName = 'data/test_data.csv'                                  # test data
    # testFileName = 'data/test_data2.csv'                                 # test data
    # testFileName = 'data/micado0730715_v2.csv'
    # testFileName = 'data/nn_training_data.csv'
    testFileName = csfFileName                                             # from parameter
    
    maximumNumberIncreasableNode = 6                                       # must be positive 6
    minimumNumberReducibleNode = -4                                        # must be negativ -4

    upperLimit = 4000000                                                   # 4000000
    lowerLimit = 1000000                                                   # 1000000


    # In[159]:

    df = readCSV(testFileName)

    print(df.head())
    
    
    # In[x]:
    
    if( last == True ):
        
        # TODO:
        # Assert
        logger.info('-------- Last row will be processed --------')
        pf = df[-1:]
        logger.info(f'-------- pf head =\n {pf.head()}')
        logger.info(f'-------- pf shape =\n {pf.shape}')
        
        # Assigne pf to df -> keep the code more coherent
        df = pf.copy()
        
        logger.info('-------- Last row will be processed --------')
        
        
    # In[x]:
    logger.info('Checking advisor data properties')
    if df.shape[0] == 0:
        error_msg = 'There are no training samples yet.'
        logger.error(error_msg)
        return advice_msg(valid = False, phase = 'invalid', error_msg = error_msg)
    
    # In[160]:

    preProcessedDF = preProcessing(df)

    print(preProcessedDF.head())
    
    
    WorkerCountName = None
    if( preProcessedDF.columns.contains('Worker count') ):
        WorkerCountName = 'Worker count'
    elif( preProcessedDF.columns.contains('vm_number') ):
        WorkerCountName = 'vm_number'
    else:
        WorkerCountName = 'Worker count'
        
    logger.info(f'(WorkerCountName = {WorkerCountName}')
    

    # Rename Worker count or vm_number to WorkerCount
    renamedDF = renameVariable(preProcessedDF, WorkerCountName, 'WorkerCount')
    
    print(renamedDF.head())
    
    metricNames         = setMetricNames(['CPU', 'Inter', 'CTXSW', 'KBIn', 'PktIn', 'KBOut', 'PktOut'])
    extendedMetricNames = setExtendedMetricNames(['CPU', 'Inter', 'CTXSW', 'KBIn', 'PktIn', 'KBOut', 'PktOut', 'WorkerCount'])

   
    filteredDF = dropFirstCases(renamedDF, cutFirstCases)
    

    

    # >#### Add new workers (increse the nuber of added Worker)

    # In[162]:

    def calculatePredictedLatencyWithVariousWorkers(modelNeuralNet, to):

        newDFForRegression = filteredDF.copy()
        nDD = filteredDF.copy()

        step = 0

        if( to == 0 ):
            print("")
            assert to != 0,"This value can not be 0. Error in calculatePredictedLatencyWithVariousWorkers method set maximum number of scalable nodes."
        elif( to > 0 ):
            step = 1
            print('............. up maximum vm = ' + str(to) + ' ...........')
        elif( to < 0 ):
            step = -1
            print('............. down maximum vm = ' + str(to) + ' ...........')

        for j in range(0, to, step):

            addedWorkerCount = j

            newDFForRegression['addedWorkerCount'] = addedWorkerCount

            for i in metricNames:

                newDFForRegressionWithTerms = calculateLinearRegressionTerms(i, newDFForRegression)

                # print('newDFForRegression.shape   = ', newDFForRegression.shape)
                # print('newDFForRegression.head()  = ', newDFForRegression.head())
                # print('newDFForRegressionWithTerms.type = ', type(newDFForRegressionWithTerms))
                # print('newDFForRegressionWithTerms.shape = ', newDFForRegressionWithTerms.shape)
                # print('newDFForRegressionWithTerms.columns = ', newDFForRegressionWithTerms.columns)
                
                # print("------------     ", newDFForRegressionWithTerms.CPU.values[0], "     ------------")
                # print("------------     ", newDFForRegressionWithTerms.shape, "     ------------")
                # print("------------     ", newDFForRegressionWithTerms.columns, "     ------------")
                

                # keep last three column - given metric, term1, term2
                X = newDFForRegressionWithTerms.iloc[:, [-3, -2, -1]]

                # print("------------     ", X.shape, "     ------------")
                # print("------------     ", X.values[0], "     ------------") # Error ez az érték első eleme fix kéne hogy legyen
                # print("------------     ", X.values[-1], "     ------------")# ugyanakkor folyamatosan változik

                # load the proper current metric model
                modelForMetric = joblib.load('models/saved_linearregression_model_' + i + '.pkl')

                # print("------------     ", modelForMetric.get_params(), "     ------------")

                if( np.isinf(X).any()[1] ):
                    X['term1'] = np.where(np.isinf(X['term1'].values), X['metric'], X['term1'])
                    X['term2'] = np.where(np.isinf(X['term2'].values), 0, X['term2'])
                    # print('-----------')
                    # print(X.to_string())

                # create prediction and store in a new numpy.array object
                predictedMetric = modelForMetric.predict(X)


                # leave original metric value (just for fun and investigation) and store in a new column
                newDFForRegression['original' + i] = newDFForRegression[i]

                # store predicted value pretend as would be the original. for example predictedCPU will be CPU
                newDFForRegression[i] = predictedMetric
                nDD[i] = predictedMetric

                # print("------------     ", newDFForRegression[['CPU']].values[0], "    ------------")
                # print("------------     ", nDD[['CPU']].values[0], "    ------------")

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

            investigationDF['predictedResponseTimeAdded' + str(j) + 'Worker'] = newDFForNerualNetworkPrediction[['predictedResponseTimeAdded' + str(j) + 'Worker']]
            investigationDFDeNormalized['denormalizedPredictedResponseTimeAdded' + str(j) + 'Worker'] = newDFForNerualNetworkPrediction[['denormalizedPredictedResponseTimeAdded' + str(j) + 'Worker']]

        # print(newDFForNerualNetworkPrediction.columns)

        # print(investigationDFDeNormalized.columns)

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


    if( 0 > 1 ):
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


    # In[166]:

    # investigationDeNormalizedDF.head().style.set_properties(**pandas_dataframe_styles).format("{:0.3f}")
    # investigationDFUp.head().style.set_properties(**pandas_dataframe_styles).format("{:0.3f}")
    # investigationDFDown.head().style.set_properties(**pandas_dataframe_styles).format("{:0.3f}")
    # investigationDFDeNormalizedUp.head().style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")
    # investigationDFDeNormalizedDown.head().style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


    # In[171]:

    if showPlots :
        VisualizePredictedYWithWorkers(0, investigationDFDown[['predictedResponseTimeAdded0Worker',
                                                               'predictedResponseTimeAdded-1Worker',
                                                               'predictedResponseTimeAdded-2Worker',
                                                               'predictedResponseTimeAdded-3Worker']], targetVariable)


    # In[172]:

    if showPlots :
        VisualizePredictedYWithWorkers(0, investigationDFUp[['predictedResponseTimeAdded1Worker',
                                                             'predictedResponseTimeAdded2Worker',
                                                             'predictedResponseTimeAdded3Worker']], targetVariable)


    # In[173]:

    if showPlots :
        VisualizePredictedYWithWorkers(0, investigationDFUp[['predictedResponseTimeAdded0Worker',
                                                             'predictedResponseTimeAdded1Worker',
                                                             'predictedResponseTimeAdded2Worker',
                                                             'predictedResponseTimeAdded3Worker',
                                                             'predictedResponseTimeAdded4Worker',
                                                             'predictedResponseTimeAdded5Worker']], targetVariable)


    # In[174]:

    if showPlots :
        VisualizePredictedYWithWorkers(0, investigationDFDeNormalizedUp[['denormalizedPredictedResponseTimeAdded0Worker',
                                                                         'denormalizedPredictedResponseTimeAdded1Worker',
                                                                         'denormalizedPredictedResponseTimeAdded2Worker',
                                                                         'denormalizedPredictedResponseTimeAdded3Worker',
                                                                         'denormalizedPredictedResponseTimeAdded4Worker',
                                                                         'denormalizedPredictedResponseTimeAdded5Worker']], targetVariable)


    # In[175]:

    if showPlots :
        VisualizePredictedYLine(investigationDFDeNormalizedUp['avg latency (quantile 0.5)'],                         investigationDFDeNormalizedUp[['denormalizedPredictedResponseTimeAdded0Worker','denormalizedPredictedResponseTimeAdded1Worker','denormalizedPredictedResponseTimeAdded2Worker','denormalizedPredictedResponseTimeAdded3Worker','denormalizedPredictedResponseTimeAdded4Worker','denormalizedPredictedResponseTimeAdded5Worker']], targetVariable)


    # In[176]:

    if showPlots :
        VisualizePredictedYLine(investigationDFDeNormalizedUp[[targetVariable]],                         investigationDFDeNormalizedUp[['denormalizedPredictedResponseTimeAdded0Worker']], targetVariable)


    # In[177]:

    if showPlots :
        VisualizePredictedYLine(investigationDFDeNormalizedUp[[targetVariable]],                         investigationDFDeNormalizedUp[['denormalizedPredictedResponseTimeAdded0Worker']], targetVariable)


    # ### Get Advice

    # In[178]:



    # In[179]:
    
    if showPlots :
        from visualizerlinux import VisualizePredictedXYLine
        from visualizerlinux import VisualizePredictedXY2Line
        VisualizePredictedXYLine(0, investigationDFDeNormalizedUp[[targetVariable]], targetVariable, lowerLimit, upperLimit)


    # In[180]:

    advice = 0
    countInRange = 0
    countViolatedUp = 0
    countViolatedDown = 0

    advicedDF = investigationDeNormalizedDF.copy()
    advicedDF['advice'] = 0
    advicedDF['postScaledTargetVariable'] = np.nan
    
    logger.info('post advice init')
    
    # print('------------------------------------------------------')
    # print('investigationDeNormalizedDF.columns')
    # print(investigationDeNormalizedDF.columns)
    # print('------------------------------------------------------')

    for i in investigationDeNormalizedDF.index:
        distance = 99999999999
        real = investigationDeNormalizedDF[[targetVariable]].get_value(i, targetVariable)
        if( upperLimit > real and lowerLimit < real ):
            advice = 0
            advicedDF.ix[i,'advice'] = 0
            countInRange += 1
            print("ok")
        else:
            print("threshold violation at index " + str(i))
            if( upperLimit < real ):
                countViolatedUp += 1
                print("threshold up violation")
                advice = 0
                postScaledTargetVariable = np.nan # 0
                distance = float('inf')
                for j in range(1, 6):
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
                        postScaledTargetVariable = relatedTargetVariable
                        break
                    print(calculatedDistance)
                advicedDF.ix[i,'advice'] = advice
                advicedDF.ix[i, 'postScaledTargetVariable'] = postScaledTargetVariable
            elif( lowerLimit > real ):
                countViolatedDown += 1
                print("threshold down violation")
                advice = 0
                postScaledTargetVariable = np.nan # 0
                distance = float('-inf')
                for j in range(-1, -3, -1):
                    # print(distance)
                    advice = 0
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
                        postScaledTargetVariable = relatedTargetVariable
                        if( calculateDistance < upperLimit ):
                            distance = calculateDistance
                            advice = j
                            postScaledTargetVariable = relatedTargetVariable
                            break
                        # break
                    print(calculateDistance)
                advicedDF.ix[i, 'advice'] = advice
                advicedDF.ix[i, 'postScaledTargetVariable'] = postScaledTargetVariable


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

    if( last ):
        advicedDF.to_csv('outputs/adviceDF.csv', sep=';', encoding='utf-8')
    
    
    # In[x]:
    
    phase = 'production'
    nn_error_rate = 0
    vm_number_total = advice
    
    logger.info(f'advice = {advice}')
    
    return_msg = advice_msg(valid = True, phase = phase, vm_number = vm_number_total, nn_error_rate = nn_error_rate)
    return return_msg
    
    
    
