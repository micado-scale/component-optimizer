def run():
    # # Advice Phase - Production Phase

    # In[156]:

    import numpy as np
    import pandas as pd

    from utils import loadMinMaxScalerXFull, loadMinMaxScalerYFull
    from utils import loadNeuralNetworkModel
    from utils import readCSV
    from utils import preProcessing, renameVariable, setMetricNames, setExtendedMetricNames, dropFirstCases

    from linearregression import calculateLinearRegressionTerms

    from visualizerlinux import VisualizePredictedYLine, VisualizePredictedYWithWorkers

    from sklearn.externals import joblib

    pandas_dataframe_styles = {
        'font-family': 'monospace',
        'white-space': 'pre'
    }


    # In[157]:


    X_normalized_MinMaxScaler = loadMinMaxScalerXFull()
    y_normalized_MinMaxScaler = loadMinMaxScalerYFull()

    modelNeuralNet = loadNeuralNetworkModel()


    # In[158]:


    # Vigyázat ennek azonosnak kell lennie a korábbi értékkel különben para van (ezt valahogy ki kéne vezetni valami külső
    # fájlba, vagy csinálni valamilyen osztályt amiben ez el van tárolva)

    cutFirstCases = 0                                                      # 10
    targetVariable = 'avg latency (quantile 0.5)'
    testFileName = 'data/grafana_data_export_long_running_test.csv'        # original data
    testFileName = 'data/test_data.csv'                                    # test data
    testFileName = 'data/test_data2.csv'                                   # test data
    # testFileName = 'data/micado0730715_v2.csv'

    maximumNumberIncreasableNode = 6                                       # must be positive
    minimumNumberReducibleNode = -4                                        # must be negativ

    upperLimit = 4000000                                                   # 6000000
    lowerLimit = 3000000                                                   # 1000000


    # In[159]:


    newDF = readCSV(testFileName)


    # In[160]:


    newPreProcessedDF = preProcessing(newDF)

    newRenamedDF = renameVariable(newPreProcessedDF, 'Worker count', 'WorkerCount')

    metricNames         = setMetricNames(['CPU', 'Inter', 'CTXSW', 'KBIn', 'PktIn', 'KBOut', 'PktOut'])
    extendedMetricNames = setExtendedMetricNames(['CPU', 'Inter', 'CTXSW', 'KBIn', 'PktIn', 'KBOut', 'PktOut', 'WorkerCount'])

    newFilteredDF = dropFirstCases(newRenamedDF, cutFirstCases)


    # >#### Add new workers (increse the nuber of added Worker)

    # In[161]:


    metricNames


    # In[162]:


    def calculatePredictedLatencyWithVariousWorkers(modelNeuralNet, to):

        newDFForRegression = newFilteredDF.copy()
        nDD = newFilteredDF.copy()

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

                print("------------     ", newDFForRegressionWithTerms.CPU.values[1], "     ------------")
                print("------------     ", newDFForRegressionWithTerms.shape, "     ------------")
                print("------------     ", newDFForRegressionWithTerms.columns, "     ------------")

                # keep last three column - given metric, term1, term2
                X = newDFForRegressionWithTerms.iloc[:, [-3, -2, -1]]

                print("------------     ", X.shape, "     ------------")
                print("------------     ", X.values[0], "     ------------") # Error ez az érték első eleme fix kéne hogy legyen
                print("------------     ", X.values[-1], "     ------------")# ugyanakkor folyamatosan változik

                # load the proper current metric model
                modelForMetric = joblib.load('models/saved_linearregression_model_' + i + '.pkl')

                print("------------     ", modelForMetric.get_params(), "     ------------")

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

                print("------------     ", newDFForRegression[['CPU']].values[1], "    ------------")
                print("------------     ", nDD[['CPU']].values[1], "    ------------")

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

        print(newDFForNerualNetworkPrediction.columns)

        print(investigationDFDeNormalized.columns)

        return investigationDF, investigationDFDeNormalized


    # In[163]:

    investigationDFUp, investigationDFDeNormalizedUp = calculatePredictedLatencyWithVariousWorkers(modelNeuralNet, maximumNumberIncreasableNode)


    # In[164]:

    investigationDFDown, investigationDFDeNormalizedDown = calculatePredictedLatencyWithVariousWorkers(modelNeuralNet, minimumNumberReducibleNode)



    # ### Merge Up and Down Adviser

    # In[165]:


    investigationDeNormalizedDF = pd.concat([investigationDFDeNormalizedDown, investigationDFDeNormalizedUp], axis = 1).T.drop_duplicates().T

    investigationDeNormalizedDF.values.shape


    # In[166]:

    # investigationDeNormalizedDF.head().style.set_properties(**pandas_dataframe_styles).format("{:0.3f}")

    # investigationDFUp.head().style.set_properties(**pandas_dataframe_styles).format("{:0.3f}")

    # investigationDFDown.head().style.set_properties(**pandas_dataframe_styles).format("{:0.3f}")

    # investigationDFDeNormalizedUp.head().style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")

    # investigationDFDeNormalizedDown.head().style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


    # In[171]:


    VisualizePredictedYWithWorkers(0, investigationDFDown[['predictedResponseTimeAdded0Worker',
                                                           'predictedResponseTimeAdded-1Worker',
                                                           'predictedResponseTimeAdded-2Worker',
                                                           'predictedResponseTimeAdded-3Worker']], targetVariable)


    # In[172]:


    VisualizePredictedYWithWorkers(0, investigationDFUp[['predictedResponseTimeAdded1Worker',
                                                         'predictedResponseTimeAdded2Worker',
                                                         'predictedResponseTimeAdded3Worker']], targetVariable)


    # In[173]:


    VisualizePredictedYWithWorkers(0, investigationDFUp[['predictedResponseTimeAdded0Worker',
                                                         'predictedResponseTimeAdded1Worker',
                                                         'predictedResponseTimeAdded2Worker',
                                                         'predictedResponseTimeAdded3Worker',
                                                         'predictedResponseTimeAdded4Worker',
                                                         'predictedResponseTimeAdded5Worker']], targetVariable)


    # In[174]:


    VisualizePredictedYWithWorkers(0, investigationDFDeNormalizedUp[['denormalizedPredictedResponseTimeAdded0Worker',                                                                  'denormalizedPredictedResponseTimeAdded1Worker',                                                                  'denormalizedPredictedResponseTimeAdded2Worker',                                                                  'denormalizedPredictedResponseTimeAdded3Worker',                                                                  'denormalizedPredictedResponseTimeAdded4Worker',                                                                  'denormalizedPredictedResponseTimeAdded5Worker']], targetVariable)


    # In[175]:


    VisualizePredictedYLine(investigationDFDeNormalizedUp['avg latency (quantile 0.5)'],                         investigationDFDeNormalizedUp[['denormalizedPredictedResponseTimeAdded0Worker',                                                           'denormalizedPredictedResponseTimeAdded1Worker',                                                           'denormalizedPredictedResponseTimeAdded2Worker',                                                           'denormalizedPredictedResponseTimeAdded3Worker',                                                           'denormalizedPredictedResponseTimeAdded4Worker',                                                           'denormalizedPredictedResponseTimeAdded5Worker']], targetVariable)


    # In[176]:


    VisualizePredictedYLine(investigationDFDeNormalizedUp[[targetVariable]],                         investigationDFDeNormalizedUp[['denormalizedPredictedResponseTimeAdded0Worker']], targetVariable)


    # In[177]:


    VisualizePredictedYLine(investigationDFDeNormalizedUp[[targetVariable]],                         investigationDFDeNormalizedUp[['denormalizedPredictedResponseTimeAdded0Worker']], targetVariable)


    # ### Get Advice

    # In[178]:


    from visualizerlinux import VisualizePredictedXYLine
    from visualizerlinux import VisualizePredictedXY2Line


    # In[179]:


    VisualizePredictedXYLine(0, investigationDFDeNormalizedUp[[targetVariable]], targetVariable, lowerLimit, upperLimit)


    # In[180]:


    advice = 0
    countInRange = 0
    countViolatedUp = 0
    countViolatedDown = 0

    advicedDF = investigationDeNormalizedDF.copy()
    advicedDF['advice'] = 0
    advicedDF['postScaledTargetVariable'] = np.nan

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
                    print(distance)
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
                    print(distance)
                    advice = 0
                    # két feltételnek kell megfelelnie sorrendben legyen a legkisebb távolsága az alsó limittől
                    # kettő legyen az alsó limit fölött (utóbbi nem biztos, hogy teljesül)
                    varName = 'denormalizedPredictedResponseTimeAdded' + str(j) + 'Worker'
                    relatedTargetVariable = investigationDeNormalizedDF.get_value(i, varName)
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


    advicedDF.head(10).style.set_properties(**pandas_dataframe_styles).format("{:0.0f}")


    # In[182]:


    VisualizePredictedXYLine(advicedDF[['advice']] * 2000000, advicedDF[[targetVariable]], targetVariable, lowerLimit, upperLimit)


    # In[183]:


    print('countInRange      = ', countInRange)
    print('countViolatedDown = ', countViolatedDown)
    print('countVilolatedUp  = ', countViolatedUp)


    # In[184]:


    VisualizePredictedXY2Line(advicedDF[[targetVariable]], advicedDF[['advice']], targetVariable, lowerLimit, upperLimit)


    # In[185]:


    from visualizerlinux import VisualizePredictedXY3Line


    # In[186]:


    VisualizePredictedXY3Line(advicedDF[[targetVariable]],advicedDF[['postScaledTargetVariable']],advicedDF[['advice']],targetVariable, lowerLimit, upperLimit)


    # In[187]:


    advicedDF.style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


    # In[188]:


    advicedDF.to_csv('outputs/adviceDF.csv', sep=';', encoding='utf-8')

