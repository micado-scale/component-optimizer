import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
import imgkit
import math

titlesize = 24
labelsize = 20

def CorrelationMatrixSave(df):

    # Set CSS properties for table elements in dataframe
    table_props = [
      ('border-style', 'solid lightgrey'),
      ('border-color', 'lightgrey'),
      ('border-spacing', '0px'),
      ('border-width', '0px 1px 1px 0px')
      ]
    
    # Set CSS properties for th elements in dataframe
    th_props = [
      ('font-size', '12px'),
      ('font-family', 'Arial'),
      ('text-align', 'right'),
      ('height', '30px'),
      ('border-style', 'solid lightgrey'),
      ('border-color', 'lightgrey'),
      ('border-spacing', '0px'),
      ('border-width', '0px 1px 1px 0px')
      ]

    # Set CSS properties for tr elements in dataframe
    tr_props = [
      ('font-size', '12px'),
      ('font-family', 'Arial'),
      ('text-align', 'right'),
      ('height', '30px'),
      ('border-style', 'solid lightgrey'),
      ('border-color', 'lightgrey'),
      ('border-spacing', '0px'),
      ('border-width', '0px 1px 1px 0px')
      ]

    # Set CSS properties for td elements in dataframe
    td_props = [
      ('font-size', '12px'),
      ('font-family', 'monospace'),
      ('text-align', 'right'),
      ('white-space', 'pre'),
      ('height', '30px'),
      ('line-height', 'normal'),
      ('border', 'none'),
      ('display', 'table-cell'),
      ('padding', '0.5em 0.5em'),
      ('border-style', 'solid lightgrey'),
      ('border-color', 'lightgrey'),
      ('border-spacing', '0px'),
      ('border-width', '0px 1px 1px 0px')
      ]

    # Set table styles
    table_styles = [
      dict(selector="table", props=table_props),
      dict(selector="th", props=th_props),
      dict(selector="tr", props=tr_props),
      dict(selector="td", props=td_props)
      ]
    
    cm = df.corr().style.set_properties(**{'font-size': '12pt', 'font-family': 'monospace', 'text-align': 'right', 'white-space': 'pre'}).set_table_styles(table_styles).format("{:0.3f}").background_gradient(cmap='Blues')

    hf = cm.render()
    imgkit.from_string(hf, 'images/CorrelationMatrix.png')
    pass


def ScatterPlots(x, y, extendedMetricNames, ylabel):
    fig = plt.figure(figsize=(20,20))
    
    row = math.ceil(len(extendedMetricNames)/2); col = 2; shape = 10; labelsize = 18; color = 'r'; marker = 'o'

    j = 1
    for i in extendedMetricNames:

        plt.subplot(row, col, j)
        plt.title('Latency vs ' + i, fontsize=titlesize)
        plt.scatter(x[i], y, s=shape, c=color, marker=marker)
        plt.xlabel(i, fontsize=labelsize)
        plt.ylabel(ylabel, fontsize=labelsize)
        j = j + 1

    plt.tight_layout(w_pad=15, h_pad=3)
    fig.savefig('images/InnerStateVariableVsTargetVariable.png')
    fig.savefig('static/InnerStateVariableVsTargetVariable.png')

    # plt.show()
    plt.close()
    pass
    

    
def TimeLinePlot(df, x):    
    fig = plt.figure(figsize=(20,7))

    row = 1; col = 1; fontsize = 24; labelsize = 18

    tmp_mean = df[x].mean()
    tmp_std  = df[x].std()

    plt.subplot(row, col, 1)
    plt.title(x, fontsize = fontsize)
    plt.plot(df[x])
    plt.xlabel('Time', fontsize = labelsize)
    plt.ylabel(x, fontsize = labelsize)
    plt.axhline(y = tmp_mean, color = 'grey')
    plt.axhline(y = tmp_mean + tmp_std, color = '0.1', linewidth = 0.8, linestyle = ':')
    plt.axhline(y = tmp_mean + 2*tmp_std, color = 'grey', linewidth = 0.5, linestyle = ':')
    plt.text(y = tmp_mean, x = 0, s = 'Mean', fontsize = 20)
    plt.text(y = tmp_mean + tmp_std, x = 0, s = '+1Std', fontsize = 20)
    plt.text(y = tmp_mean + 2*tmp_std, x = 0, s = '+2Std', fontsize = 20)

    fig.savefig('images/AverageLatencyQuantileZeroPointFiveTimeLines.png')
    fig.savefig('static/AverageLatencyQuantileZeroPointFiveTimeLines.png')

    # plt.show()
    plt.close()
    pass
    
def TimeLinePlots(df, extendedMetricNames):
    fig = plt.figure(figsize=(20,20))

    row = math.ceil(len(extendedMetricNames)/2); col = 2; shape = 10; labelsize = 18; color = 'r'; marker = 'o'

    j = 1
    for i in extendedMetricNames:

        plt.subplot(row, col, j)
        plt.title('Time vs ' + i, fontsize=titlesize)
        # plt.plot(preProcessedDF[i], c=color, marker=marker)
        plt.plot(df[i], c=color)
        plt.xlabel('Time', fontsize=labelsize)
        plt.ylabel(i, fontsize=labelsize)
        j = j + 1

    plt.tight_layout(w_pad=15, h_pad=3)
    fig.savefig('images/InnerStateVariableTimeLines_2.png')
    fig.savefig('static/InnerStateVariableTimeLines_2.png')

    # plt.show()
    plt.close()
    pass

def VisualizePredictedYScatter(y_normalized, y_predicted, targetVariable):

    fig = plt.figure(figsize=(20,7))
    # plt.subplot(1, 1, 1)
    plt.title(targetVariable + ' vs predicted ' + targetVariable + ' by Neural Network', fontsize=titlesize)
    plt.scatter(x = y_normalized, y = y_predicted, s = 20)
    plt.xlabel('Normalized real Response Time', fontsize=labelsize)
    plt.ylabel('Normalized estimated Response Time', fontsize=labelsize)
    fig.savefig('images/Y_normailizedVsY_predictedLatency.png')
    fig.savefig('static/Y_normailizedVsY_predictedLatency.png')

    # plt.show()
    plt.close()
    pass

def VisualizePredictedYLine(y_normalized, y_predicted, targetVariable, lines = False):
    
    fig = plt.figure(figsize=(20,7))
    # plt.subplot(1, 1, 1)
    plt.title(targetVariable + ' and its predicted values', fontsize=titlesize)
    plt.plot(y_normalized)
    plt.plot(y_predicted)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.xlabel('Time', fontsize=labelsize)
    plt.ylabel('Normalized (-1,+1) latencies', fontsize=labelsize)
    if( lines ):
        plt.axhline(y = y_predicted.max(), color = 'grey', alpha = 0.8, linestyle = '--')
        plt.axhline(y = y_predicted.min(), color = 'grey', linestyle = '--')
    plt.gca().legend(('real','predicted'), loc = 2, prop={'size': 16})
    fig.savefig('images/AverageLatencyQuantileZeroPointFiveTimeLinesAndPredictedValuesAgainstTime.png')
    fig.savefig('static/AverageLatencyQuantileZeroPointFiveTimeLinesAndPredictedValuesAgainstTime.png')
    
    # plt.show()
    plt.close()
    pass

def VisualizePredictedYLineWithValues(y_normalized, y_predicted, targetVariable, name = 'Normalized'):
    
    fig = plt.figure(figsize=(20,7))
    # plt.subplot(1, 1, 1)
    plt.title(targetVariable + ' and its predicted values', fontsize=titlesize)
    plt.plot(y_normalized)
    plt.plot(y_predicted)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.xlabel('Time', fontsize=labelsize)
    plt.ylabel(name + ' latencies', fontsize=labelsize)
    plt.axhline(y = y_predicted.max(), color = 'grey', alpha = 0.8, linestyle = '--')
    plt.axhline(y = y_predicted.min(), color = 'grey', linestyle = '--')
    plt.text(y = y_predicted.max() - 0.1, x = -0, s = str(y_predicted.max())[:15], fontsize = 14, horizontalalignment='left')
    plt.text(y = y_predicted.min() + 0.02, x = -0, s = str(y_predicted.min())[:15], fontsize = 14, horizontalalignment='left')
    plt.gca().legend(('real','predicted'), loc = 2, prop={'size': 16})
    fig.savefig('images/VisualizePredictedYLineWithValues' + name + '.png')
    fig.savefig('static/VisualizePredictedYLineWithValues' + name + '.png')
    
    # plt.show()
    plt.close()
    pass

def VisualizePredictedYWithWorkers(y_normalized, y_predicted, targetVariable):
    
    fig = plt.figure(figsize=(20,7))
    # plt.subplot(1, 1, 1)
    plt.title(targetVariable + ' and its predicted values', fontsize=titlesize)
    plt.plot(y_predicted)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.xlabel('Time', fontsize=labelsize)
    plt.ylabel('Normalized (-1,+1) latencies', fontsize=labelsize)
    variables = y_predicted.columns
    plt.gca().legend((variables), loc = 2, prop={'size': 16})

    fig.savefig('images/PredictedYByWorkers.png')
    fig.savefig('static/PredictedYByWorkers.png')
    
    # plt.show()
    plt.close()
    pass

def VisualizePredictedXYLine(y_normalized, y_predicted, targetVariable, lowerLimit, upperLimit):
    
    fig = plt.figure(figsize=(20,7))
    # plt.subplot(1, 1, 1)
    plt.title('Response time and proposed numb. of resources', fontsize=titlesize)
    plt.plot(y_normalized)
    plt.plot(y_predicted)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.xlabel('Timeline', fontsize=labelsize)
    plt.ylabel('Response time (ms)', fontsize=labelsize)
    plt.axhline(y = lowerLimit, color = 'grey')
    plt.axhline(y = upperLimit, color = 'grey')
    plt.gca().legend(('Advided numb. of new resources','Measured response time (ms)'), loc = 2, prop={'size': 12})
    fig.savefig('images/AverageLatencyQuantileZeroPointFiveTimeLinesAndPredictedValuesAgainstTime_2.jpg')
    fig.savefig('static/AverageLatencyQuantileZeroPointFiveTimeLinesAndPredictedValuesAgainstTime_2.jpg')
    
    # plt.show()
    plt.close()
    pass

def VisualizePredictedXY2Line(y1, y2, targetVariable, lowerLimit, upperLimit):

    fig, ax1 = plt.subplots(figsize=(20,7))
    
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']
    # print(colors)
    
    ax2 = ax1.twinx()
    p1 = ax1.plot(y1, color = '#1f77b4', label = 'Response time (ms)')
    p2 = ax2.plot(y2, color = '#ff7f0e', label = 'Proposedd numb. of resources')

    ax1.set_xlabel('Timeline', fontsize = labelsize)
    ax1.set_ylabel('Response time (ms)', fontsize=labelsize)
    ax2.set_ylabel('Proposedd numb. of resources', fontsize=labelsize)
    ax2.set_ylim([-2, 12])

    ax1.axhline(y = lowerLimit, color = 'grey')
    ax1.axhline(y = upperLimit, color = 'grey')

    # ax1.text(y = lowerLimit, x = 0, s = r'{:d}'.format(lowerLimit), va = 'bottom', ha = 'center', fontsize = 12)
    # ax1.text(y = upperLimit, x = 0, s = upperLimit, va = 'bottom', ha = 'center', fontsize = 12)
        
    plt.title('Response time and proposed numb. of resources', fontsize=titlesize)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    
    ps = p1+p2
    labels = [label.get_label() for label in ps]
    ax1.legend(ps, labels, loc = 2, prop = {'size': 12})

    fig.savefig('images/AverageLatencyQuantileZeroPointFiveTimeLinesAndPredictedValuesAgainstTime_3.png')
    fig.savefig('static/AverageLatencyQuantileZeroPointFiveTimeLinesAndPredictedValuesAgainstTime_3.png')
    
    # plt.show()
    plt.close()
    pass

def VisualizePredictedXY3Line(y1, y2, y3, targetVariable, lowerLimit, upperLimit):

    fig, ax1 = plt.subplots(figsize=(20,7))
    
    ax2 = ax1.twinx()
    p1 = ax1.plot(y1, color = '#1f77b4', label = 'Response time (ms)')
    p2 = ax1.plot(y2, color = '#000000', label = 'Predicted Response time (ms)')
    p3 = ax2.plot(y3, color = '#ff7f0e', label = 'Proposedd numb. of resources')

    ax1.set_xlabel('Timeline', fontsize = labelsize)
    ax1.set_ylabel('Response time (ms)', fontsize=labelsize)
    ax2.set_ylabel('Proposedd numb. of resources', fontsize=labelsize)
    ax2.set_ylim([-2, 12])

    ax1.axhline(y = lowerLimit, color = 'grey')
    ax1.axhline(y = upperLimit, color = 'grey')
        
    plt.title('Response time and proposed numb. of resources', fontsize=titlesize)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    
    ps = p1+p2+p3
    labels = [label.get_label() for label in ps]
    ax1.legend(ps, labels, loc = 2, prop = {'size': 12})

    fig.savefig('images/ResponseTimeAdvicePredicted.png')
    fig.savefig('static/ResponseTimeAdvicePredicted.png')
    
    # plt.show()
    plt.close()
    pass

def VisualizePredictedXY4Line(y1, y2, y3, y4, targetVariable, lowerLimit, upperLimit):

    fig, ax1 = plt.subplots(figsize=(20,7))
    
    ax2 = ax1.twinx()
    p1 = ax1.plot(y1, color = '#1f77b4', label = 'Response time (ms)')
    p2 = ax1.plot(y2, color = '#000000', label = 'Predicted Response time (ms)')
    p3 = ax2.plot(y3, color = '#ff7f0e', label = 'Proposedd numb. of resources')
    # p4 = ax2.plot(y4, color = '#008238', label = 'Actual numb. of resources')
    # p4 = ax2.plot(y4, color = '#00755E', label = 'Actual numb. of resources')
    p4 = ax2.plot(y4, color = '#670000', label = 'Actual numb. of resources')

    ax1.set_xlabel('Timeline', fontsize = labelsize)
    ax1.set_ylabel('Response time (ms)', fontsize=labelsize)
    ax2.set_ylabel('Proposedd numb. of resources', fontsize=labelsize)
    ax2.set_ylim([-2, 12])

    ax1.axhline(y = lowerLimit, color = 'grey')
    ax1.axhline(y = upperLimit, color = 'grey')
        
    plt.title('Response time and proposed numb. of resources', fontsize=titlesize)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    
    ps = p1+p2+p3+p4
    labels = [label.get_label() for label in ps]
    ax1.legend(ps, labels, loc = 2, prop = {'size': 12})

    fig.savefig('images/ResponseTimeAdvicePredictedWorkerCount.png')
    fig.savefig('static/ResponseTimeAdvicePredictedWorkerCount.png')
    
    # plt.show()
    plt.close()
    pass

def ScatterPlotsTrainTest(y_train, y_train_predicted, y_test, y_test_predicted, targetVariable):
    fig = plt.figure(figsize=(20,7))

    fontsize = 10

    plt.subplot(1, 2, 1)
    plt.title(targetVariable + ' vs its predicted values by Neural Network on train data set', fontsize=fontsize)
    plt.scatter(x = y_train, y = y_train_predicted, s = 10)

    plt.subplot(1, 2, 2)
    plt.title(targetVariable + ' vs its predicted values by Neural Network on test data set', fontsize=fontsize)
    plt.scatter(x = y_test, y = y_test_predicted, s = 10)

    fig.savefig('images/Y_denormailizedVsY_predictedLatency.png')
    fig.savefig('static/Y_denormailizedVsY_predictedLatency.png')

    # plt.show()
    plt.close()
    pass

def ipythonPlotMetricsRealAgainstPredicted(temporaryScalingDF, metricNames):
    row = math.ceil(len(metricNames)/2); col = 2; j = 1
    fig = plt.figure(figsize=(20,30))
    for i in metricNames:
        plt.subplot(row, col, j)
        plt.scatter(temporaryScalingDF['next1'+str(i)], temporaryScalingDF['predictedNext1'+str(i)], s = 10)
        plt.xlabel('next1'+str(i))
        plt.ylabel('predictedNext1'+str(i))
        j = j + 1
        
    fig.savefig('images/InnerStateMetricsRegression.png', bbox_inches = 'tight', pad_inches = 0)
    fig.savefig('static/InnerStateMetricsRegression.png', bbox_inches = 'tight', pad_inches = 0)
    
    # plt.show()
    plt.close()
    pass

def ipythonPlotMetricsRealAgainstPredictedRegression(temporaryScalingDF, metricNames):
    j = 1
    
    # fig = plt.figure(figsize=(32,20))
    sns.set(style="white", color_codes=True)
    for i in metricNames:        
        g = sns.jointplot(x=temporaryScalingDF['next1'+str(i)], y=temporaryScalingDF['predictedNext1'+str(i)], kind='reg', ratio=3)
        
        g.savefig('images/InnerStateMetricRegression' + str(j) + '.jpg')
        g.savefig('static/InnerStateMetricRegression' + str(j) + '.jpg')
        
        j = j + 1
        
        plt.close()
    
    # plt.show()
    # plt.close()
    pass