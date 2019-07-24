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
    
    # print('Score = ', model.score(X, y))
    print(metric, 'MAE \t=\t{:0.2f}'.format(metrics.mean_absolute_error(y, y_predicted)))
    
    return y_predicted