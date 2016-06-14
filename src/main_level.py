
import readSaveData as getData
import datetime  
import code
import plotting as customPlot
import numpy as np
import utilities as util
import matplotlib.pyplot as plt
import math
import networkTrainingSetup as setup
import statsmodels.api as sm

from sklearn import datasets, preprocessing
from sklearn.cross_validation import train_test_split
from neupy import algorithms, layers, estimators, environment
from numbers import Real
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier,RandomForestRegressor
from sklearn.cross_validation import cross_val_score

environment.reproducible()

showPlots = 0

# 0 means re-read xlsx spreadhseet and then save data objects based on spreadsheet details below
# otherwise use previously saved data objects (.pickle files) as inputs 
loadXlData = 0;

if(loadXlData):
    fileName = "ASX_ForecastNN_Data.xlsx"
       
    #a string for the name of each worksheet tab in the spreadsheet to be read in
    workbookSheetNames = ['EXCHANGE RATES',
                          'USD EXRATES AND GOLD',
                          'CM YIELDS GB',
                          'IR AND YIELDS - MONEY MARKET',
                          'COMMODITY PRICES',
                          'GDP AND INCOME',
                          'CPI',
                          'US T-Bill 3MO',
                          'US T-Bill 1Y',
                          'Indices'
                          ]   
    #the first row entry in the corresponding worksheet containing the row of codes 
    #describing each data column
    firstRowOfCodesPerSheet = [11,
                               11,
                               11,
                               11,
                               11,
                               11,
                               11,
                               6,
                               6,
                               2,
                                ]
    
    getData.readExcelSpreadsheet_SaveInputs(fileName,workbookSheetNames,firstRowOfCodesPerSheet)


ASX_ForecastNN_Numeric = getData.load_obj('ASX_ForecastNN_Numeric')
ASX_ForecastNN_Unicodes = getData.load_obj('ASX_ForecastNN_Unicodes')
ASX_ForecastNN_Dates = getData.load_obj('ASX_ForecastNN_Dates')
ASX_ForecastNN_SheetNames = getData.load_obj('ASX_ForecastNN_SheetNames')
    
datesIndex = 0
numericDataIndex = 1
returnsCalcOption = 'LOG_DIFF' #rel_diff

#choose data between these dates 
#startDate = datetime.datetime(2000, 4, 6)
startDate = datetime.datetime(2000, 5, 9)
endDate = datetime.datetime(2016, 5, 9)

assert(startDate.isoweekday() in range(1, 6)),"startDate is not a weekday - choose another startDate"
assert(endDate.isoweekday() in range(1, 6)),"endDate is not a weekday - choose another endDate"

dataPlotLabelsIdx = 0
dataWorksheetIdx = 1
dataCodeIdx = 2
lagIdx = 3
pricesInputs = []

# choice of data added to list given by information below
# ['data label name', 'worksheet name of data' , 'code string' , 'lag index']
# i.e. ['FX AUD/USD 1MO($)','EXCHANGE RATES','FXRUSD', 1]
# 'FX AUD/USD 1MO($)' = the data label for AUD USD FX rates, to be used for plots 
# 'EXCHANGE RATES' = the name of the worksheet where the data is found
# 'FXRUSD' = the code string of the data 
# '1' = number of days to lag data with respect to startDate and endDate above

lagAmericanMarketClose = -1
lagAsianMarketClose = -1
lagAsianMarketOpen = 0

pricesInputs.append(['S&P500_DAILY_PX_LAST','Indices','SPX_PX_LAST',lagAmericanMarketClose])
pricesInputs.append(['STFINL_DAILY_PX_LAST','Indices','STFINL_PX_LAST',lagAmericanMarketClose])
pricesInputs.append(['SHCOMP_DAILY_PX_LAST','Indices','SHCOMP_PX_LAST',lagAsianMarketClose])
pricesInputs.append(['ASX200_DAILY_PX_OPEN','Indices','AS51_PX_OPEN',lagAsianMarketOpen])
pricesInputs.append(['ASX200_DAILY_PX_LAST','Indices','AS51_PX_LAST',lagAsianMarketOpen])
pricesInputs.append(['ASX200_INDX_GROSS_DAILY_DIV','Indices','AS51_INDX_GROSS_DAILY_DIV',lagAsianMarketOpen])
pricesInputs.append(['ASX200_INDX_NET_DAILY_DIV','Indices','AS51_INDX_NET_DAILY_DIV',lagAsianMarketOpen])
pricesInputs.append(['AUDUSD_CURRENCY','Indices','AUDUSD_Curncy',lagAmericanMarketClose])
pricesInputs.append(['XAU_CURRENCY','Indices','XAU_Curncy',lagAmericanMarketClose])
pricesInputs.append(['CRUDEOIL_COMMODITY','Indices','CL1_Comdty',lagAmericanMarketClose])
pricesInputs.append(['AUD1Y_GOVT','Indices','GTAUD1Y_Govt',lagAsianMarketClose])
pricesInputs.append(['90D_BANKBILL','Indices','IR1_Comdty',lagAsianMarketClose])
pricesInputs.append(['OIS_1M','Indices','ADSOA_Curncy',lagAsianMarketClose])
pricesInputs.append(['OIS_3M','Indices','ADSOC_Curncy',lagAsianMarketClose])
pricesInputs.append(['AUD1Y_SWAP','Indices','ADSWAP1_Curncy',lagAsianMarketClose])
pricesInputs.append(['AUD10Y_GOVT','Indices','XM1_Comdty',lagAsianMarketClose])
pricesInputs.append(['USD10Y_TNOTE','Indices','TY1_Comdty',lagAmericanMarketClose])
pricesInputs.append(['USDJPY_CURRENCY','Indices','USDJPY_Curncy',lagAmericanMarketClose])
pricesInputs.append(['ASX200_ACCUMULATION_OPEN','Indices','ASA51_PX_OPEN',lagAsianMarketOpen])
pricesInputs.append(['ASX200_ACCUMULATION_LAST','Indices','ASA51_PX_LAST',lagAsianMarketOpen])

useReturn = 0
usePrice = 1
predictorInputs = []
# predictorInputs.append(['S&P500_DAILY_PX_LAST',useReturn])
# predictorInputs.append(['STFINL_DAILY_PX_LAST',useReturn])
# predictorInputs.append(['SHCOMP_DAILY_PX_LAST',useReturn])
# predictorInputs.append(['ASX200_INDX_GROSS_DAILY_DIV',usePrice])
# predictorInputs.append(['AUDUSD_CURRENCY',useReturn])
# predictorInputs.append(['XAU_CURRENCY',useReturn])
# predictorInputs.append(['CRUDEOIL_COMMODITY',useReturn])
# predictorInputs.append(['90D_BANKBILL',useReturn])
# predictorInputs.append(['OIS_1M',useReturn])
# predictorInputs.append(['OIS_3M',useReturn])
# predictorInputs.append(['AUD1Y_SWAP',useReturn])
# predictorInputs.append(['AUD10Y_GOVT',useReturn])
# predictorInputs.append(['USD10Y_TNOTE',useReturn])
# predictorInputs.append(['USDJPY_CURRENCY',useReturn])
predictorInputs.append(['S&P500_DAILY_PX_LAST',useReturn])
predictorInputs.append(['STFINL_DAILY_PX_LAST',useReturn])
predictorInputs.append(['SHCOMP_DAILY_PX_LAST',useReturn])
predictorInputs.append(['AUDUSD_CURRENCY',useReturn])
predictorInputs.append(['XAU_CURRENCY',useReturn])
predictorInputs.append(['CRUDEOIL_COMMODITY',useReturn])
predictorInputs.append(['90D_BANKBILL',useReturn])
predictorInputs.append(['AUD10Y_GOVT',useReturn])
predictorInputs.append(['USD10Y_TNOTE',useReturn])
predictorInputs.append(['USDJPY_CURRENCY',useReturn])

targetOpenPriceLabel = 'ASX200_DAILY_PX_OPEN'
targetLastPriceLabel = 'ASX200_DAILY_PX_LAST'
benchmarkOpenLabel = 'ASX200_ACCUMULATION_OPEN'
benchmarkCloseLabel = 'ASX200_ACCUMULATION_LAST'
targetLabel = 'ASX200_INTRADAY_'+returnsCalcOption+'_RETS'
targetChoice = 'LEVEL'
testStrategy = 'LONG_SHORT'
dailyInterestRate = 0.03/365

if(targetChoice == 'DIRECTION'):
    targetLabel = targetLabel +'_DIR'
    classifyDirection = 'DUAL_VEC'
else:
    classifyDirection = None

isIntraDayClassification = 0

pricesLabels = []
pricesData = []
pricesDates = []

returnsLabels = []
returnsData = []
returnsDates = []

for inputChoice in pricesInputs:
    code = inputChoice[dataCodeIdx]
    dataLocation = inputChoice[dataWorksheetIdx]
    lag = inputChoice[lagIdx]
    
    inputStartDate = startDate + datetime.timedelta(days=(lag-1))    
    #check if the result is a weekend, then shift again
    if(inputStartDate.isoweekday() > 5):
        inputStartDate = inputStartDate + datetime.timedelta(days=-2) 
            
    inputEndDate = endDate + datetime.timedelta(days=(lag))
    #check if the result is a weekend, then shift again
    if(inputEndDate.isoweekday() > 5):
        inputEndDate = inputEndDate + datetime.timedelta(days=-2) 
           
    requestedData = getData.returnColDataInPeriod(inputStartDate,inputEndDate,dataLocation,code,ASX_ForecastNN_SheetNames,ASX_ForecastNN_Dates,ASX_ForecastNN_Unicodes,ASX_ForecastNN_Numeric)
    pricesData, pricesDates, pricesLabels = util.appendData(requestedData[numericDataIndex],
                                                       requestedData[datesIndex], 
                                                       inputChoice[dataPlotLabelsIdx], 
                                                       pricesData,
                                                       pricesDates,
                                                       pricesLabels)
    
    rets,retsDates = util.getReturns(requestedData[numericDataIndex],requestedData[datesIndex],returnsCalcOption)
    returnsData, returnsDates, returnsLabels = util.appendData(rets,
                                                   retsDates, 
                                                   inputChoice[dataPlotLabelsIdx], 
                                                   returnsData,
                                                   returnsDates,
                                                   returnsLabels)
 
    assert(pricesData[len(pricesData)-1].count(None) == 0),"Gathered Price Data: '" + pricesLabels[len(pricesLabels)-1] + "' contains missing values"
    assert(returnsData[len(returnsData)-1].count(None) == 0),"Computed Returns Data: '" + returnsLabels[len(returnsLabels)-1] + "' contains missing values"
    assert(pricesDates[len(pricesDates)-1].count(None) == 0),"Dates for: '" + pricesLabels[len(pricesLabels)-1] + "' contains missing values"
    assert(returnsDates[len(returnsDates)-1].count(None) == 0),"Dates for: '" + returnsLabels[len(returnsLabels)-1] + "' contains missing values"

dataContainer = getData.dataContainer(pricesData,pricesDates,pricesLabels,returnsData,returnsDates,returnsLabels)

if ((targetLabel in dataContainer.returnsLabels) | (targetLabel in dataContainer.pricesLabels)):
    target, targetDates, targetPrices, targetPricesDates = setup.createTarget(dataContainer,targetLabel,targetChoice,classifyDirection,returnsCalcOption) 
    targetLabel = targetLabel +'_RETS'
    if(targetChoice == 'DIRECTION'):
        targetLabel = targetLabel +'_DIR'
else:
    isIntraDayClassification = 1
    target, targetDates, targetOpenPrices, targetPrices, targetPricesDates = setup.createIntraDayTarget(dataContainer,targetOpenPriceLabel,targetLastPriceLabel,targetLabel,targetChoice,classifyDirection,returnsCalcOption) 

if(showPlots):
    customPlot.subPlotData(dataContainer.pricesDates,dataContainer.pricesData,dataContainer.pricesLabels,'Prices Data')
    #customPlot.correlationHeatMap(dataContainer.pricesData,dataContainer.pricesLabels,'Prices Correlation Heatmap')
    customPlot.correlationHeatMap(dataContainer.returnsData,dataContainer.returnsLabels,'Returns Correlation Heatmap')

npArrayTarget = util.convertTargetListToNumpyArray(target)
npArrayTargetPrices = np.array(targetPrices)
npArrayBenchmarkOpenPrices = np.array(dataContainer.pricesData[dataContainer.pricesLabels.index(benchmarkOpenLabel)][1:])
npArrayBenchmarkClosePrices = np.array(dataContainer.pricesData[dataContainer.pricesLabels.index(benchmarkCloseLabel)][1:])
 
if(isIntraDayClassification):
    npArrayTargetOpenPrices = np.array(targetOpenPrices)

predictorInputData,predictorLabels =  setup.createPredictorVariables(dataContainer,predictorInputs,returnsCalcOption)
npArrayPredictorInputData = np.transpose(np.array(predictorInputData))

binCount = 50
if(showPlots):
    #customPlot.plotHist(dataContainer.returnsData,binCount,dataContainer.returnsLabels,'Returns Data Histogram')    
    customPlot.plotHist(predictorInputData,binCount,predictorLabels,'Predictor Variable Histogram')
    customPlot.plotHist([target],binCount,[targetLabel],'Target Histogram')

# check all predictor variables allocated correctly 
# scale inputs and output and train
#standardise inputs 
predictorInputData_scaler = preprocessing.StandardScaler()
#0.6585 is the max of second derivative of tanh function, yields better training results
target_scaler = preprocessing.MinMaxScaler(feature_range=(-0.6585, 0.6585))

trainTestSplit = 0.6
targetScaled = target_scaler.fit_transform(npArrayTarget)
predictorInputDataScaled =  predictorInputData_scaler.fit_transform(npArrayPredictorInputData)

print '\nList of predictor labels\n'
for idx in range(len(predictorLabels)):
    print str(idx+1) + ' '+ predictorLabels[idx]
    
testInputImportance = 0
if(testInputImportance):   
    # fit an Extra Trees model to the data
    inputSelectionModel = ExtraTreesClassifier(criterion='entropy')
    targetScaledClassDir = np.array([math.copysign(1,npArrayTarget[i]) for i in range(len(npArrayTarget))])
    inputSelectionModel.fit(predictorInputDataScaled, targetScaledClassDir)
    scores = 100*inputSelectionModel.feature_importances_
    
#     rf = ExtraTreesClassifier(criterion='entropy')#RandomForestRegressor(n_estimators=20, max_depth=4)#ExtraTreesClassifier(criterion='entropy')
#     scoresLabels = []
#     scores = []
#     for i in range(predictorInputDataScaled.shape[1]):
#         score = cross_val_score(rf, predictorInputDataScaled[:, i:i+1], targetScaledClassDir, scoring='accuracy',cv=10)
#         scores.append(round(np.mean(score),2))
#         scoresLabels.append((round(np.mean(score), 2),predictorLabels[i]))
# 
#     print '\nPredictor Inputs sorted by cross-validated performance (% accuracy)\n'
#     for feature in sorted(scoresLabels,reverse=True):
#         idx = predictorLabels.index(feature[1])
#         print 'Input '+ str(idx+1) +': \tScore: '+ str(feature[0]) +'\tName: '+ feature[1]
    
    # display the relative importance of each attribute
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(predictorLabels)), scores,label=predictorLabels,color='r')
    plt.xlabel('Predictor Labels')
    #plt.ylabel('% accuracy')
    plt.ylabel('Relative Importance')
    #plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.05))
    plt.title('Predictor Input cross-validated performance to classify: '+targetLabel +'_DIR')
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(len(predictorLabels))+0.5, minor=False)   
    plt.grid(True)       
    #ax.set_xticklabels(predictorLabels, minor=False,rotation = 'vertical')
    ax.set_xticklabels(range(1,len(predictorLabels)+1))
    plt.show() 
    
    inputSelectionModel = RandomForestRegressor(n_estimators=len(predictorLabels), max_depth=5)
    inputSelectionModel.fit(predictorInputDataScaled, npArrayTarget[:,0])
    featureImportanceContent = 100*inputSelectionModel.feature_importances_
    scores = featureImportanceContent

#     rf = RandomForestRegressor(n_estimators=20, max_depth=6)
#     scoresLabels = []
#     scores = []    
#     for i in range(predictorInputDataScaled.shape[1]):
#         score = cross_val_score(rf, predictorInputDataScaled, npArrayTarget[:,0], scoring='r2',cv=10)
#         scores.append(round(np.mean(score),2))
#         scoresLabels.append((round(np.mean(score), 2),predictorLabels[i]))
#  
#     print '\nPredictor Inputs sorted by cross-validated performance (R2 score)\n'
#     for feature in sorted(scoresLabels,reverse=True):
#         idx = predictorLabels.index(feature[1])
#         print 'Input '+ str(idx+1) +': \tScore: '+ str(feature[0]) +'\tName: '+ feature[1]
    
    # display the relative importance of each attribute
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(predictorLabels)), scores,label=predictorLabels,color='r')
    plt.xlabel('Predictor Labels')
    #plt.ylabel('R2 Score')
    plt.ylabel('Relative Importance')
    #plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.05))
    plt.title('Predictor Inputs cross-validated performance on: '+targetLabel)
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(len(predictorLabels))+0.5, minor=False)   
    plt.grid(True)       
    #ax.set_xticklabels(predictorLabels, minor=False,rotation = 'vertical')
    ax.set_xticklabels(range(1,len(predictorLabels)+1))
    plt.show() 

firstIdxTest = int(math.ceil(len(targetDates)*trainTestSplit))
trainRange = range(0,firstIdxTest)
testRange = range(firstIdxTest,len(targetDates))

x_train = predictorInputDataScaled[trainRange,:]
y_train = targetScaled[trainRange]

x_test = predictorInputDataScaled[testRange,:]
y_test = targetScaled[testRange]

checkInputScaling = 0
if(checkInputScaling):
    #check the scaling and standardising of inputs 
    customPlot.plotHist(np.transpose(predictorInputDataScaled).tolist(),binCount,predictorLabels,'Predictor Variables Scaled')
    customPlot.plotHist([targetScaled.tolist()],binCount,[targetLabel],'Target Scaled')

runMLP = 1
if(runMLP):
    numHiddenNeurons = 29
    Feedforward_MLP_NetworkName = 'Feedforward_MLP_'+ targetChoice    
    trainNetwork = 0
      
    if(trainNetwork):   
        print('Feedforward MLP Level Results')                                                                                     
        network = algorithms.ConjugateGradient(
              connection=[
                 layers.Tanh(npArrayPredictorInputData.shape[1]),
                 layers.Tanh(numHiddenNeurons),
                 layers.Output(npArrayTarget.shape[1]),
              ],
              search_method='golden',
              update_function='fletcher_reeves',
              addons=[algorithms.LinearSearch],
              error='rmse',
              verbose=True,
              show_epoch=10,
              )
                 
        network.train(x_train, y_train, x_test,y_test,epochs=110)
        getData.save_network(network,Feedforward_MLP_NetworkName)  
          
        #network = getData.load_network(Feedforward_MLP_NetworkName)
        
        y_train_Predict = util.applyNetworkReturnOriginalScale(x_train,target_scaler,network)[:,0]
        y_train_Target = target_scaler.inverse_transform(y_train[:,0])
               
        trainDates = [targetDates[i] for i in trainRange]
        histTitle = 'Target Returns Error - Trained '+Feedforward_MLP_NetworkName
        plotTitle = 'Returns Performance - Trained ' + Feedforward_MLP_NetworkName
        util.predictorResultsSummary(binCount,histTitle,plotTitle,targetLabel,trainDates,y_train_Target,y_train_Predict)
               
    else:
        network = getData.load_network(Feedforward_MLP_NetworkName)
         
    network.plot_errors()      
    
    y_test_Predict = util.applyNetworkReturnOriginalScale(x_test,target_scaler,network)[:,0]
    y_test_Target = target_scaler.inverse_transform(y_test[:,0])
           
    testDates = [targetDates[i] for i in testRange]
    histTitle = 'Target Returns Error - Test '+Feedforward_MLP_NetworkName
    plotTitle = 'Returns Performance - Test ' + Feedforward_MLP_NetworkName
    util.predictorResultsSummary(binCount,histTitle,plotTitle,targetLabel,testDates,y_test_Target,y_test_Predict)
      
    strategyPNL,buyHoldPNL,accumulationPNL = util.getIntraDayPNL(npArrayBenchmarkClosePrices,y_test_Predict,[npArrayTargetOpenPrices[i] for i in testRange],[npArrayTargetPrices[i] for i in testRange],dailyInterestRate,testStrategy)
    customPlot.plotPerformance(testDates,accumulationPNL,strategyPNL,'Buy-Hold','Strategy','PNL Chart',testStrategy + ' Strategy - Test '+Feedforward_MLP_NetworkName)                                
    print(Feedforward_MLP_NetworkName+": Strategy PNL = {}, Buy-Hold PNL = {}".format(
        strategyPNL[-1], accumulationPNL[-1]
    )) 
                
runGRNN = 1
if(runGRNN):
    grnnStd = np.linspace(0.1, 2, 75)    
    #grnnStd = [1.25]
    GRNN_NetworkName = 'GRNN_Network'                      
    trainNetwork = 1
    if(trainNetwork):  
        RMSLE_GRNN = []
        minRMSLE = 1e10
        bestStd = 0
        splitIdx = int(len(trainRange)/2)
        trainRangeFold1 = trainRange[:splitIdx]
        trainRangeFold2 = trainRange[(splitIdx+1):]
        print('GRNN Training Results - Test Std dev input')  
        for x in grnnStd:
            grnnNW = algorithms.GRNN(std=x, verbose=False)
            grnnNW.train(x_train[trainRangeFold1,:], y_train[trainRangeFold1,:])
            networkRMSLE = estimators.rmsle(y_train[trainRangeFold2,:], grnnNW.predict(x_train[trainRangeFold2,:])[:,0])
            
            if(minRMSLE > networkRMSLE):
                minRMSLE = networkRMSLE
                bestStd = x
                
            RMSLE_GRNN.append(networkRMSLE)

        plt.figure
        p1, = plt.plot(grnnStd, RMSLE_GRNN,'b')
        plt.xlabel('GRNN Std.')
        plt.ylabel('Train RMSLE')
        plt.grid(True)       
        plt.title('Train RMSLE to determine GRNN Std. Input')
        plt.show() 
    
        grnnNW = algorithms.GRNN(std=bestStd, verbose=True)
        grnnNW.train(x_train, y_train)       
        getData.save_network(grnnNW,GRNN_NetworkName)               
            
        y_train_Predict = util.applyNetworkReturnOriginalScale(x_train,target_scaler,grnnNW)[:,0]
        y_train_Target = target_scaler.inverse_transform(y_train[:,0])
               
        trainDates = [targetDates[i] for i in trainRange]
        histTitle = 'Target Returns Error - Trained '+GRNN_NetworkName
        plotTitle = 'Returns Performance - Trained ' + GRNN_NetworkName
        util.predictorResultsSummary(binCount,histTitle,plotTitle,targetLabel,trainDates,y_train_Target,y_train_Predict)
            
    else:
        grnnNW = getData.load_network(GRNN_NetworkName)  
          
    grnnNW.plot_errors()      

    y_test_Predict = util.applyNetworkReturnOriginalScale(x_test,target_scaler,grnnNW)[:,0]
    y_test_Target = target_scaler.inverse_transform(y_test[:,0])
           
    testDates = [targetDates[i] for i in testRange]
    histTitle = 'Target Returns Error - Test '+GRNN_NetworkName
    plotTitle = 'Returns Performance - Test ' + GRNN_NetworkName
    util.predictorResultsSummary(binCount,histTitle,plotTitle,targetLabel,testDates,y_test_Target,y_test_Predict)

    strategyPNL,buyHoldPNL,accumulationPNL = util.getIntraDayPNL(npArrayBenchmarkClosePrices,y_test_Predict,[npArrayTargetOpenPrices[i] for i in testRange],[npArrayTargetPrices[i] for i in testRange],dailyInterestRate,testStrategy)
    customPlot.plotPerformance([targetPricesDates[i] for i in testRange],accumulationPNL,strategyPNL,'Buy-Hold','Strategy','PNL Chart',testStrategy +' Strategy -  Test '+GRNN_NetworkName)                                
    print(GRNN_NetworkName+": Strategy PNL = {}, Buy-Hold PNL = {}".format(
        strategyPNL[-1], accumulationPNL[-1]
    ))  
        
runOLS = 0
if(runOLS):
    OLS_ObjName = 'OLS_LevelPredictor'
    trainOLS = 0
    
    x_train = sm.add_constant(x_train)
    x_test = sm.add_constant(x_test)
        
    if(trainOLS): 
        model = sm.OLS(y_train, x_train)
        OLS_LevelPredictor = model.fit()  
        getData.save_obj(OLS_LevelPredictor,OLS_ObjName)         
        print(OLS_LevelPredictor.summary())
        
        y_train_Predict = target_scaler.inverse_transform(OLS_LevelPredictor.predict(x_train))
        y_train_Target = target_scaler.inverse_transform(y_train[:,0])
               
        trainDates = [targetDates[i] for i in trainRange]
        histTitle = 'Target Returns Error - Trained ' + OLS_ObjName
        plotTitle = 'Returns Performance - Trained ' + OLS_ObjName
        util.predictorResultsSummary(binCount,histTitle,plotTitle,targetLabel,trainDates,y_train_Target,y_train_Predict)         
        
    else:
        OLS_LevelPredictor = getData.load_obj(OLS_ObjName)         

    y_test_Predict = target_scaler.inverse_transform(OLS_LevelPredictor.predict(x_test))
    y_test_Target = target_scaler.inverse_transform(y_test[:,0])
      
    testDates = [targetDates[i] for i in testRange]
    histTitle = 'Target Returns Error - Test '+OLS_ObjName
    plotTitle = 'Returns Performance - Test ' + OLS_ObjName
    util.predictorResultsSummary(binCount,histTitle,plotTitle,targetLabel,testDates,y_test_Target,y_test_Predict)

    strategyPNL,buyHoldPNL,accumulationPNL = util.getIntraDayPNL(npArrayBenchmarkClosePrices,y_test_Predict,[npArrayTargetOpenPrices[i] for i in testRange],[npArrayTargetPrices[i] for i in testRange],dailyInterestRate,testStrategy)
    customPlot.plotPerformance([targetPricesDates[i] for i in testRange],accumulationPNL,strategyPNL,'Buy-Hold','Strategy','PNL Chart',testStrategy + ' Strategy - Test '+OLS_ObjName)                                
    print(OLS_ObjName+": Strategy PNL = {}, Buy-Hold PNL = {}".format(
        strategyPNL[-1], accumulationPNL[-1]
    ))   