
# coding: utf-8

# In[1]:


# import optimizer as op

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib

# import plotly
# import plotly.plotly as py
# import plotly.graph_objs as go
# import plotly.figure_factory as ff

np.set_printoptions(precision=3, suppress=True)

pandas_dataframe_styles = {
    'font-family': 'monospace',
    'white-space': 'pre'
}


# In[2]:


inputCSVFile   = 'data/grafana_data_export_long_running_test.csv'

targetVariable = 'avg latency (quantile 0.5)'


scaler_min = -1                     # 0
scaler_max = 1                      # 1
train_test_ratio = 0.3              # 0.3
activation_function = 'tanh'        # tanh, relu, logistic
neuronsWhole = 10                   # 10
neuronsTrainTest = 4                # 4
cutFirstCases = 10                  # 10

lead = 1                            # 1 default

showPlots = True                    # True


# In[3]:


def readCSV(filename):
    df = pd.read_csv(filename, sep=";", header="infer", skiprows=0, na_values="null" )

    # Return DataFrame
    return df


# In[4]:


# Read DataFrame
df = readCSV(inputCSVFile)


# In[5]:


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


# In[6]:



# Preprecess DataFrame
preProcessedDF = preProcessing(df)


# In[7]:


targetVariable = targetVariable


# In[8]:


def renameVariable(df, old_var_name, new_var_name):
    new_df = df.copy()
    new_df.rename(columns={old_var_name: new_var_name}, inplace=True)
    return new_df


# In[9]:


preProcessedDF = renameVariable(preProcessedDF, 'Worker count', 'WorkerCount')


# In[10]:


def setMetricNames(names):
    new_metricNames = names.copy()
    return new_metricNames


# In[11]:


metricNames = setMetricNames(['CPU', 'Inter', 'CTXSW', 'KBIn', 'PktIn', 'KBOut', 'PktOut'])


# In[12]:


def setExtendedMetricNames(names):
    new_extendedMetricNames = names.copy()
    return new_extendedMetricNames


# In[13]:


extendedMetricNames = setExtendedMetricNames(['CPU', 'Inter', 'CTXSW', 'KBIn', 'PktIn', 'KBOut', 'PktOut', 'WorkerCount'])

extendedMetricNames


# In[14]:


def dropFirstCases(df, n):
    new_df = df.copy()
    filteredDF = new_df[new_df.index > n]
    return filteredDF


# In[15]:


# because in the begining of the samples have a lot of outliers

filteredDF = dropFirstCases(preProcessedDF, cutFirstCases)


# In[16]:


preProcessedDF = filteredDF


# ### Correlation Matrix

# In[17]:


from visualizerlinux import CorrelationMatrixSave


# In[18]:


CorrelationMatrixSave(preProcessedDF)


# In[19]:


from visualizerlinux import ScatterPlots


# In[20]:


if showPlots : ScatterPlots(preProcessedDF, preProcessedDF[targetVariable], extendedMetricNames, targetVariable)


# In[21]:


from visualizerlinux import TimeLinePlot


# In[22]:


if showPlots : TimeLinePlot(preProcessedDF, targetVariable)


# In[23]:


from visualizerlinux import TimeLinePlots


# In[24]:


if showPlots : TimeLinePlots(preProcessedDF, extendedMetricNames)


# In[25]:



n = 1
for i in preProcessedDF.columns:
    print('AC(1)      ', i, '\t= ', np.round(preProcessedDF[i].autocorr(lag=1), 2))
    n = n+1
    if( n == 10 ):
        break


# ## Create a whole new DataFrame for Before After Data

# In[26]:


def createBeforeafterDF(df, lag):
    beforeafterDF = df.copy()
    inputVariables = np.flip(beforeafterDF.columns[0:9].ravel(), axis=-1)
    print('Input Variablels : ', inputVariables)

    index = 9
    for i in inputVariables:
        new_column = beforeafterDF[i].shift(lag)
        new_column_name = (i + str(1)) # Todo: rename str(lag)
        beforeafterDF.insert(loc=index, column=new_column_name, value=new_column)
    
    beforeafterDF = beforeafterDF[lag:]
    
    print('Before After DF columns: ', beforeafterDF.columns)
    
    return beforeafterDF


# In[27]:


beforeafterDF = createBeforeafterDF(preProcessedDF, 1)


# ### Set Features for Neural Network - these are the input variables

# In[28]:


def setFeaturesAndTheirLags(df):
    X = df.iloc[:,0:9]
    return X


# In[29]:


X = setFeaturesAndTheirLags(beforeafterDF)


# ### Set Target Variable for Neural Network - this is the target variable

# In[30]:


def setTarget(df, targetVariable):
    y = df[targetVariable]
    return y


# In[31]:


y = setTarget(beforeafterDF, targetVariable)


# In[32]:


y.values[0:10]


# In[33]:


y.head()


# In[34]:


y.describe()


# ### Normalize the whole X

# In[35]:


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


# In[36]:


X_normalized, X_normalized_MinMaxScaler = normalizeX(X)


# ### Load MinMaxScalerXFull
# 

# In[37]:


def loadMinMaxScalerXFull():
    X_normalized_MinMaxScaler = joblib.load('models/scaler_normalizeX.save')
    
    return X_normalized_MinMaxScaler


# In[38]:


X_normalized_MinMaxScaler = loadMinMaxScalerXFull()


# In[39]:


def printNormalizedX(X_normalized):
    print("X_normalized type        = ", type(X_normalized))
    print("X_normalizde dtype       = ", X_normalized.dtype)
    print("X_normalized shape       = ", X_normalized.shape)
    print("X_normalized ndim        = ", X_normalized.ndim)
    print("X_normalized[:,0].max()  = ", X_normalized[:,0].max())
    print("X_normalized[:,0].min()  = ", X_normalized[:,0].min())


# In[40]:


printNormalizedX(X_normalized)


# In[41]:


X_normalized[1]


# In[42]:


X_denormalized = X_normalized_MinMaxScaler.inverse_transform(X_normalized)


# In[43]:


X_denormalized[1]


# In[44]:


X_denormalized[-1]


# ### Normalize the whole y

# In[45]:


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


# In[46]:


y_normalized, y_normalized_MinMaxScaler = normalizeY(y)


# In[47]:


def printNormalizedY(y_normalized):
    """Void. Print normalizeY(df) values"""
    
    print("y_normalized type        = ", type(y_normalized))
    print("y_normalized dtype       = ", y_normalized.dtype)
    print("y_normalized shape       = ", y_normalized.shape)
    print("y_normalized ndim        = ", y_normalized.ndim)
    print("y_normalized[:].max()    = ", y_normalized[:].max())
    print("y_normalized[:].min()    = ", y_normalized[:].min())


# In[48]:


printNormalizedY(y_normalized)


# In[49]:


y_normalized[0:3]


# ### Load MinMaxScalerYFull

# In[50]:


def loadMinMaxScalerYFull():
    y_normalized_MinMaxScaler = joblib.load('models/scaler_normalizeY.save')
    
    return y_normalized_MinMaxScaler


# In[51]:


y_normalized_MinMaxScaler = loadMinMaxScalerYFull()


# In[52]:


y_denormalized = y_normalized_MinMaxScaler.inverse_transform(y_normalized.reshape(y_normalized.shape[0],1))


# In[53]:


y_denormalized[0:3]


# In[54]:


y_denormalized[-3:]


# ## Train Neural Network with Optimizer Class, trainMultiLayerRegressor method

# In[55]:


def trainMultiLayerRegressor(X_normalized, y_normalized, activation, neuronsWhole):

    # Train Neural Network
    mlp = MLPRegressor(hidden_layer_sizes=neuronsWhole,                        max_iter=250,                        activation=activation,                        solver="lbfgs",                        learning_rate="constant",                        learning_rate_init=0.01,                        alpha=0.01,                        verbose=False,                        momentum=0.9,                        early_stopping=False,                        tol=0.00000001,                        shuffle=False,                        # n_iter_no_change=20, \
                       random_state=1234)

    mlp.fit(X_normalized, y_normalized)
    
    # ide kéne beilleszteni a modell elmentését
    joblib.dump(mlp, 'models/saved_mlp_model.pkl')

    return mlp


# In[56]:


# Train Neural Network
mlp = trainMultiLayerRegressor(X_normalized, y_normalized, activation_function, neuronsWhole)


# In[57]:


def predictMultiLayerRegressor(mlp, X_normalized):
    y_predicted = mlp.predict(X_normalized)

    return y_predicted


# In[58]:


# Create prediction
y_predicted = predictMultiLayerRegressor(mlp, X_normalized)


# In[59]:


from optimizer import evaluateGoodnessOfPrediction


# In[60]:


evaluateGoodnessOfPrediction(y_normalized, y_predicted)


# ### Visualize Data

# In[61]:


from visualizerlinux import VisualizePredictedYScatter


# In[62]:


VisualizePredictedYScatter(y_normalized, y_predicted, targetVariable)


# In[63]:


from visualizerlinux import VisualizePredictedYLine, VisualizePredictedYLineWithValues


# In[64]:


VisualizePredictedYLineWithValues(y_normalized, y_predicted, targetVariable, 'Normalized')


# ### De-normlaize
# 
# I want to see the result in original scale. I don't care about the X but the y_normalized and y_predcited.
# 
# 

# In[65]:


y_denormalized = y_normalized_MinMaxScaler.inverse_transform(y_normalized.reshape(y_normalized.shape[0],1))

y_predicted_denormalized = y_normalized_MinMaxScaler.inverse_transform(y_predicted.reshape(y_predicted.shape[0],1))


# ### Can I visualize the de-normalized data as well?

# In[66]:


VisualizePredictedYLineWithValues(y_denormalized, y_predicted_denormalized, targetVariable, 'Denormalized')


# ### Compare the Original Target Variable and the mean of its Predicted Values

# In[67]:


meanOfOriginalPandasDataframe = y.values.mean()
meanOfOriginalTargetVariable  = y_denormalized.mean()
meanOfPredictedTargetVariable = y_predicted_denormalized.mean()

print('mean original pandas dataframe = ', meanOfOriginalPandasDataframe)
print('mean original target variable  = ', meanOfOriginalTargetVariable)
print('mean predicted target variable = ', meanOfPredictedTargetVariable)


# ### De-normalizer function

# In[68]:


def denormalizeX(X_normalized, X_normalized_MinMaxScaler):
    X_denormalized = X_normalized_MinMaxScaler.inverse_transform(X_normalized)
    return X_denormalized


# In[69]:


X_denormalized = denormalizeX(X_normalized, X_normalized_MinMaxScaler)


# In[70]:


X_denormalized[1]


# In[71]:


X_normalized[1]


# In[72]:


X_denormalized[-1]


# In[73]:


X_normalized[-1]


# In[74]:


def denormalizeY(y_normalized, y_normalized_MinMaxScaler):
    y_denormalized = y_normalized_MinMaxScaler.inverse_transform(y_normalized.reshape(y_normalized.shape[0],1))
    return y_denormalized


# In[75]:


y_denormalized = denormalizeY(y_normalized, y_normalized_MinMaxScaler)

y_predicted_denormalized = denormalizeY(y_predicted, y_normalized_MinMaxScaler)


# In[76]:


VisualizePredictedYLineWithValues(y_denormalized, y_predicted_denormalized, targetVariable, 'Denormalized')


# ## Create Train-Test-Validation set
# 
# ### Split Data

# In[77]:


def splitDataFrame(X, y, testSize):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize,                                                         random_state=12345,                                                         shuffle=False,                                                         stratify=None)

    return X_train, X_test, y_train, y_test


# In[78]:


# Split DataFrame
X_train, X_test, y_train, y_test = splitDataFrame(X, y, train_test_ratio)


# In[79]:


print(X_train.count())


# In[80]:


print(X_test.count())


# In[81]:


print("y_train.count() = ", y_train.count())
print("y_test.count()  = ", y_test.count())


# In[82]:


from utils import compareTwoVariables


# In[83]:


compareTwoVariables(X_train, X_test, 'CPU')


# In[84]:


compareTwoVariables(X_train, X_test, 'CTXSW')


# In[85]:


compareTwoVariables(X_train, X_test, 'AVG RR')


# In[86]:


VisualizePredictedYLineWithValues(y.values, y_train.values, targetVariable, 'Denormalized')


# In[87]:


VisualizePredictedYLineWithValues(y, y_train, targetVariable, 'Denormalized')


# In[88]:


VisualizePredictedYLineWithValues(0, y_test, targetVariable, 'Denormalized')


# In[89]:


VisualizePredictedYLineWithValues(0, y_test.values, targetVariable, 'Denormalized')


# In[90]:


VisualizePredictedYLineWithValues(y_train, y_train.values, targetVariable, 'Denormalized')


# ### Train Test set comparison

# In[91]:


from utils import printInfoTrainTestSet


# In[92]:


printInfoTrainTestSet(y_train, y_test)


# In[93]:


y_train.values[0:10]


# In[94]:


y_test.values[10:20]


# ### Normalize Data - based on Train set - It is forbiden to use Test set

# In[95]:


def normalizeXTrainTest(X_train, X_test):
    scaler = MinMaxScaler(feature_range=(scaler_min, scaler_max))
    scaler.fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    return X_train_normalized, X_test_normalized, scaler


# In[96]:


X_train_normalized, X_test_normalized, X_normalized_MinMaxScalerTrainTest = normalizeXTrainTest(X_train, X_test)


# In[97]:


print("X_train_normalized max = ", X_train_normalized.max())
print("X_train_normalized min = ", X_train_normalized.min())
print("X_test_normalized max  = ", X_test_normalized.max())
print("X_test_normalized min  = ", X_test_normalized.min())


# In[98]:


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


# In[99]:


y_train_normalized, y_test_normalized, y_normalized_MinMaxScalerTrainTest = normalizeYTrainTest(y_train, y_test)


# In[100]:


print("y_train_normalized max = ", y_train_normalized.max())
print("y_train_normalized min = ", y_train_normalized.min())
print("y_test_normalized max  = ", y_test_normalized.max())
print("y_test_normalized min  = ", y_test_normalized.min())


# In[101]:


from utils import printInfoNumpyArrays


# In[102]:


printInfoNumpyArrays(y_train_normalized, y_test_normalized)


# In[103]:


printInfoNumpyArrays(y_train, y_test)


# In[104]:


for i in range(0,9):
    print(X_test_normalized[:,i].min())
    print(X_test_normalized[:,i].max())
    print('--------------------------')


# In[105]:


X_train_denormalized = denormalizeX(X_train_normalized, X_normalized_MinMaxScalerTrainTest)


# In[106]:


X_test_denormalized = denormalizeX(X_test_normalized, X_normalized_MinMaxScalerTrainTest)


# In[107]:


y_train_denormalized = denormalizeY(y_train_normalized, y_normalized_MinMaxScalerTrainTest)


# In[108]:


y_test_denormalized = denormalizeY(y_test_normalized, y_normalized_MinMaxScalerTrainTest)


# In[109]:


VisualizePredictedYLineWithValues(y.values, y_train_denormalized, targetVariable, 'Denormalized')


# In[110]:


VisualizePredictedYLineWithValues(y.values[len(y_train_denormalized):], y_test_denormalized, targetVariable, 'Denormalized')


# In[111]:


# this is the same as did it before, when whole dataset was trained

def trainMultiLayerRegressor(X_train_normalized, y_train_normalized, activation, neuronsTrainTest):

    # Train Neural Network
    mlp = MLPRegressor(hidden_layer_sizes=neuronsTrainTest,                        max_iter=250,                        activation=activation,                        solver="lbfgs",                        learning_rate="constant",                        learning_rate_init=0.01,                        alpha=0.01,                        verbose=False,                        momentum=0.9,                        early_stopping=False,                        tol=0.00000001,                        shuffle=False,                        # n_iter_no_change=200, \
                       random_state=1234)

    mlp.fit(X_train_normalized, y_train_normalized)
    
    # ide kéne beilleszteni a modell elmentését
    joblib.dump(mlp, 'models/saved_mlp_model_train_test.pkl')

    return mlp


# In[112]:


# Train Neural Network
mlp = trainMultiLayerRegressor(X_train_normalized, y_train_normalized, activation_function, neuronsTrainTest)


# In[113]:


def predictMultiLayerRegressor(mlp, X_normalized):
    y_predicted = mlp.predict(X_normalized)

    return y_predicted


# In[114]:


# Create prediction
y_train_predicted = predictMultiLayerRegressor(mlp, X_train_normalized)

# Create prediction
y_test_predicted = predictMultiLayerRegressor(mlp, X_test_normalized)


# In[115]:


evaluateGoodnessOfPrediction(y_train_normalized, y_train_predicted)
print('---------------------')
evaluateGoodnessOfPrediction(y_test_normalized, y_test_predicted)


# In[116]:


y_train_denormalized = denormalizeY(y_train_normalized, y_normalized_MinMaxScalerTrainTest)

y_test_denormalized = denormalizeY(y_test_normalized, y_normalized_MinMaxScalerTrainTest)

y_train_predicted_denormalized = denormalizeY(y_train_predicted, y_normalized_MinMaxScalerTrainTest)

y_test_predicted_denormalized = denormalizeY(y_test_predicted, y_normalized_MinMaxScalerTrainTest)


# In[117]:


from visualizerlinux import ScatterPlotsTrainTest


# In[118]:


ScatterPlotsTrainTest(y_train_denormalized, y_train_predicted_denormalized,                       y_test_denormalized, y_test_predicted_denormalized, targetVariable)


# In[119]:


ScatterPlotsTrainTest(y_train_normalized, y_train_predicted,                       y_test_normalized, y_test_predicted, targetVariable)


# In[120]:


VisualizePredictedYLine(y_train_denormalized, y_train_predicted_denormalized, targetVariable, lines = True)


# In[121]:


VisualizePredictedYLine(y_train_normalized, y_train_predicted, targetVariable, lines = True)


# In[122]:


VisualizePredictedYLine(y_test_denormalized, y_test_predicted_denormalized, targetVariable, lines = True)


# In[123]:


VisualizePredictedYLine(y_test_normalized, y_test_predicted, targetVariable, lines = True)


# In[124]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[125]:


def createBeforeafterDFLags(df, lag):
    beforeafterDFLags = df.copy()
    inputVariables = np.flip(beforeafterDFLags.columns[0:10].ravel(), axis=-1)
    print('Input Variablels : ', inputVariables)

    index = 10
    for i in inputVariables:
        new_column = beforeafterDFLags[i].shift(lag)
        new_column_name = (str('prev') + str(1) + i) # Todo: rename str(lag)
        beforeafterDFLags.insert(loc=index, column=new_column_name, value=new_column)

    beforeafterDFLags = beforeafterDFLags[lag:]             # remove first row as we haven't got data in lag var
    
    return beforeafterDFLags


# In[126]:


beforeafterDFLags = createBeforeafterDFLags(preProcessedDF, 1)


# In[127]:


beforeafterDFLags.columns


# In[128]:


def createBeforeafterDFLeads(df, lead = 1):
    beforeafterDFLeads = df.copy()
    inputVariables = np.flip(beforeafterDFLeads.columns[0:10].ravel(), axis=-1)
    print('Input Variablels : ', inputVariables)

    index = 10
    for i in inputVariables:
        new_column = beforeafterDFLeads[i].shift(-lead)
        new_column_name = (str('next') + str(1) + i) # Todo: rename str(lead)
        beforeafterDFLeads.insert(loc=index, column=new_column_name, value=new_column)

    beforeafterDFLeads = beforeafterDFLeads[:-lead]             # remove last row as we haven't got data in lead var
    
    beforeafterDFLeads = beforeafterDFLeads.iloc[:,:-1]     # remove last column - Latency
    
    return beforeafterDFLeads


# In[129]:


beforeafterDF = createBeforeafterDFLeads(beforeafterDFLags, lead = lead)


# In[130]:


beforeafterDF.columns


# In[131]:


# assert

a_colName = beforeafterDF.columns[-1]
a_cols = beforeafterDF.shape[1]

assert a_colName == 'prev1WorkerCount', "This column name is: {0} insted of prev1WorkerCount".format(a_colName)
assert a_cols == 30, "This column number is: {0} insted of 17".format(a_colName)


# In[132]:


def calculateWorkerCountDifferences(beforeafterDF):
    new_beforeafterDF = beforeafterDF.copy()
    new_beforeafterDF['addedWorkerCount'] = new_beforeafterDF['next1WorkerCount'].values - new_beforeafterDF['WorkerCount']
    
    return new_beforeafterDF


# In[133]:


theBeforeAfterDF = calculateWorkerCountDifferences(beforeafterDF)


# In[134]:


def createScalingDF(theBeforeAfterDF):
    new_beforeafterDF = theBeforeAfterDF.copy()
    scalingDF = new_beforeafterDF[new_beforeafterDF.WorkerCount != new_beforeafterDF.next1WorkerCount]
    
    return scalingDF


# In[135]:


scalingDF = createScalingDF(theBeforeAfterDF)


# In[136]:


beforeafterMetricsDF = scalingDF.copy()

for i in metricNames:
    # print(i)
    changeInMetricAfterScale = beforeafterMetricsDF['next1'+i]-beforeafterMetricsDF[i]
    beforeafterMetricsDF['changed1'+i] = changeInMetricAfterScale


# In[137]:


beforeafterMetricsDF[['prev1CPU','CPU','next1CPU','changed1CPU','prev1WorkerCount','WorkerCount','next1WorkerCount']]. head(10).style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


# In[138]:


beforeafterMetricsDF[['prev1CPU','CPU','next1CPU','changed1CPU','prev1WorkerCount','WorkerCount','next1WorkerCount']]. head(10).style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


# In[139]:


beforeafterMetricsDF[['changed1CPU', 'changed1Inter', 'changed1CTXSW', 'changed1KBIn',                       'changed1KBOut', 'changed1PktIn', 'changed1PktOut', 'addedWorkerCount']]. groupby(['addedWorkerCount'], as_index=False).mean().style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


# In[140]:


beforeafterMetricsDF[['changed1CPU', 'changed1Inter', 'changed1CTXSW', 'changed1KBIn',                       'changed1KBOut', 'changed1PktIn', 'changed1PktOut', 'addedWorkerCount']]. groupby(['addedWorkerCount'], as_index=False).count().style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


# In[141]:


print(theBeforeAfterDF.shape)

print(scalingDF.shape)


# In[142]:


metricNames


# In[143]:




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
    # print(metric, 'MAE \t=\t{:0.2f}'.format(metrics.mean_absolute_error(y, y_predicted)))
    
    # todo: refactor
    print(metric)
    evaluateGoodnessOfPrediction(y, y_predicted)
    print('-----------------------------------')
    
    return y_predicted


# In[144]:


temporaryScalingDF = scalingDF.copy()


# In[145]:


d={}
for i in metricNames:
    d["model{0}".format(i)]="Hello " + i
    # print(d)
    
d.get('modelCPU')


# In[146]:



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


# In[147]:


temporaryScalingDF, linearRegressionModels = learningLinearRegression(scalingDF, temporaryScalingDF, metricNames)


# In[148]:


linearRegressionModelNames = linearRegressionModels.keys()

print(linearRegressionModelNames)

modelCPU = linearRegressionModels.get('modelCPU')

print(type(modelCPU))


# In[149]:


temporaryScalingDF.columns


# In[150]:


temporaryScalingDF.shape


# In[151]:


temporaryScalingDF.shape


# In[152]:


from visualizerlinux import ipythonPlotMetricsRealAgainstPredicted


# In[153]:


metricNames = ['CPU', 'Inter', 'CTXSW', 'KBIn', 'PktIn', 'KBOut', 'PktOut']

ipythonPlotMetricsRealAgainstPredicted(temporaryScalingDF, metricNames)


# In[154]:


from visualizerlinux import ipythonPlotMetricsRealAgainstPredictedRegression


# In[155]:


ipythonPlotMetricsRealAgainstPredictedRegression(temporaryScalingDF, metricNames)


# ### End of Learning Phase

# <a id="test_begin"></a>
# 
# # Advice Phase - Production Phase

# In[156]:


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




investigationDFUp, investigationDFDeNormalizedUp = calculatePredictedLatencyWithVariousWorkers(modelNeuralNet,                                                                                                maximumNumberIncreasableNode)



# In[164]:




investigationDFDown, investigationDFDeNormalizedDown = calculatePredictedLatencyWithVariousWorkers(modelNeuralNet,                                                                                                    minimumNumberReducibleNode)



# ### Merge Up and Down Adviser

# In[165]:


investigationDeNormalizedDF = pd.concat([investigationDFDeNormalizedDown,                              investigationDFDeNormalizedUp], axis = 1).T.drop_duplicates().T

investigationDeNormalizedDF.values.shape


# In[166]:


investigationDeNormalizedDF.head().style.set_properties(**pandas_dataframe_styles).format("{:0.3f}")


# In[167]:


investigationDFUp.head().style.set_properties(**pandas_dataframe_styles).format("{:0.3f}")


# In[168]:


investigationDFDown.head().style.set_properties(**pandas_dataframe_styles).format("{:0.3f}")


# In[169]:


investigationDFDeNormalizedUp.head().style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


# In[170]:


investigationDFDeNormalizedDown.head().style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


# In[171]:


VisualizePredictedYWithWorkers(0, investigationDFDown[['predictedResponseTimeAdded0Worker',                                                        'predictedResponseTimeAdded-1Worker',                                                        'predictedResponseTimeAdded-2Worker',                                                        'predictedResponseTimeAdded-3Worker']], targetVariable)


# In[172]:


VisualizePredictedYWithWorkers(0, investigationDFUp[['predictedResponseTimeAdded1Worker',                                                      'predictedResponseTimeAdded2Worker',                                                      'predictedResponseTimeAdded3Worker']], targetVariable)


# In[173]:


VisualizePredictedYWithWorkers(0, investigationDFUp[['predictedResponseTimeAdded0Worker',                                               'predictedResponseTimeAdded1Worker',                                               'predictedResponseTimeAdded2Worker',                                               'predictedResponseTimeAdded3Worker',                                               'predictedResponseTimeAdded4Worker',                                               'predictedResponseTimeAdded5Worker']], targetVariable)


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


VisualizePredictedXYLine(0, investigationDFDeNormalizedUp[[targetVariable]],                          targetVariable, lowerLimit, upperLimit)


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


VisualizePredictedXYLine(advicedDF[['advice']] * 2000000, advicedDF[[targetVariable]],                          targetVariable, lowerLimit, upperLimit)


# In[183]:


print('countInRange      = ', countInRange)
print('countViolatedDown = ', countViolatedDown)
print('countVilolatedUp  = ', countViolatedUp)


# In[184]:


VisualizePredictedXY2Line(advicedDF[[targetVariable]], advicedDF[['advice']],                          targetVariable, lowerLimit, upperLimit)


# In[185]:


from visualizerlinux import VisualizePredictedXY3Line


# In[186]:


VisualizePredictedXY3Line(advicedDF[[targetVariable]],                           advicedDF[['postScaledTargetVariable']],                           advicedDF[['advice']],                           targetVariable, lowerLimit, upperLimit)


# In[187]:


advicedDF.style.set_properties(**pandas_dataframe_styles).format("{:0.2f}")


# In[188]:


advicedDF.to_csv('outputs/adviceDF.csv', sep=';', encoding='utf-8')

