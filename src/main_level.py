
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
pricesInputs.append(['USD10Y_GOVT','Indices','TY1_Comdty',lagAmericanMarketClose])
pricesInputs.append(['USDJPY_CURRENCY','Indices','USDJPY_Curncy',lagAmericanMarketClose])
pricesInputs.append(['ASX200_ACCUMULATION_OPEN','Indices','ASA51_PX_OPEN',lagAsianMarketOpen])
pricesInputs.append(['ASX200_ACCUMULATION_LAST','Indices','ASA51_PX_LAST',lagAsianMarketOpen])

useReturn = 0
usePrice = 1
predictorInputs = []
predictorInputs.append(['S&P500_DAILY_PX_LAST',useReturn])
predictorInputs.append(['STFINL_DAILY_PX_LAST',useReturn])
predictorInputs.append(['SHCOMP_DAILY_PX_LAST',useReturn])
predictorInputs.append(['AUDUSD_CURRENCY',useReturn])
predictorInputs.append(['XAU_CURRENCY',useReturn])
predictorInputs.append(['CRUDEOIL_COMMODITY',useReturn])
predictorInputs.append(['90D_BANKBILL',useReturn])
predictorInputs.append(['AUD10Y_GOVT',useReturn])
predictorInputs.append(['USD10Y_GOVT',useReturn])
predictorInputs.append(['USDJPY_CURRENCY',useReturn])

targetOpenPriceLabel = 'ASX200_DAILY_PX_OPEN'
targetLastPriceLabel = 'ASX200_DAILY_PX_LAST'
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

if(showPlots):
    customPlot.subPlotData(dataContainer.pricesDates,dataContainer.pricesData,dataContainer.pricesLabels,'Prices Data')
    customPlot.correlationHeatMap(dataContainer.pricesData,dataContainer.pricesLabels,'Prices Correlation Heatmap')

if ((targetLabel in dataContainer.returnsLabels) | (targetLabel in dataContainer.pricesLabels)):
    target, targetDates, targetPrices, targetPricesDates = setup.createTarget(dataContainer,targetLabel,targetChoice,classifyDirection,returnsCalcOption) 
    targetLabel = targetLabel +'_RETS'
    if(targetChoice == 'DIRECTION'):
        targetLabel = targetLabel +'_DIR'
else:
    isIntraDayClassification = 1
    target, targetDates, targetOpenPrices, targetPrices, targetPricesDates = setup.createIntraDayTarget(dataContainer,targetOpenPriceLabel,targetLastPriceLabel,targetLabel,targetChoice,classifyDirection,returnsCalcOption) 

npArrayTarget = util.convertTargetListToNumpyArray(target)
npArrayTargetPrices = np.array(targetPrices)
accumulationOpenPrices = dataContainer.pricesData[dataContainer.pricesLabels.index('ASX200_ACCUMULATION_OPEN')][1:]
accumulationClosePrices = dataContainer.pricesData[dataContainer.pricesLabels.index('ASX200_ACCUMULATION_LAST')][1:]
 
if(isIntraDayClassification):
    npArrayTargetOpenPrices = np.array(targetOpenPrices)

predictorInputData,predictorLabels =  setup.createPredictorVariables(dataContainer,predictorInputs,returnsCalcOption)
#convert all targets to numpy arrays
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

# split and apply pre-processing to data
# x_train, x_test, y_train, y_test = train_test_split(
#     predictorInputData_scaler.fit_transform(npArrayPredictorInputData),
#     targetScaled,
#     train_size=trainTestSplit
# )

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
    Feedforward_MLP_NetworkName_extended = 'Feedforward_MLP_ext'+ targetChoice    
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
        #not working, cant have numbers in name
        #networkName = 'ConjGrad_tanh<'+str(npArrayPredictorInputData.shape[1])+'>_tanh<'+str(numHiddenNeurons)+'>_output<'+str(npArrayTarget.shape[1])+'>'
        getData.save_network(network,Feedforward_MLP_NetworkName_extended)  
         
        #network = getData.load_network(Feedforward_MLP_NetworkName_extended)
        
        y_train_Predict = target_scaler.inverse_transform(network.predict(x_train))
        y_train_TargetScaled = target_scaler.inverse_transform(y_train[:,0])
        trainErrs = y_train_TargetScaled - y_train_Predict[:,0]
    
        customPlot.plotHist([trainErrs.tolist()],binCount,[targetLabel +'_ERROR'],'Target Returns Error - Trained MLP')
        
        customPlot.plotPerformance([targetDates[i] for i in trainRange],y_train_TargetScaled,y_train_Predict,'True','Est',targetLabel,'Returns Performance - Trained MLP')                  
        
        print 'Trained MLP Results - Intra Day Returns Performance'
        print 'Mean err = ',np.mean(trainErrs)
        print 'RMSE = ', np.sqrt(np.mean(trainErrs**2))
        
        #use level estimated returns vectors to infer target direction 
        targetDirection = np.array([math.copysign(1,y_train_TargetScaled[i]) for i in range(len(y_train_TargetScaled))])
        estTargetDirection = np.array([math.copysign(1,y_train_Predict[i]) for i in range(len(y_train_Predict))])
        
        print('Trained MLP Level Est - Classification Results')                                                                     
        print(Feedforward_MLP_NetworkName_extended+": Guessed {} out of {} = {}% correct".format(
            np.sum(targetDirection == estTargetDirection), targetDirection.size, 100*np.sum(targetDirection == estTargetDirection)/targetDirection.size
        )) 
    else:
        network = getData.load_network(Feedforward_MLP_NetworkName_extended)
         
    network.plot_errors()      
    y_test_Predict = target_scaler.inverse_transform(network.predict(x_test))
    y_test_TargetScaled = target_scaler.inverse_transform(y_test[:,0])
    trainErrs = y_test_TargetScaled - y_test_Predict[:,0]

    customPlot.plotHist([trainErrs.tolist()],binCount,[targetLabel +'_ERROR'],'Target Returns Error - Test MLP')
    customPlot.plotPerformance([targetDates[i] for i in testRange],y_test_TargetScaled,y_test_Predict,'True','Est',targetLabel,'Returns Performance - Test MLP')                  
        
    print 'Test MLP Results - Intra Day Returns Performance'
    print 'Mean err = ',np.mean(trainErrs)
    print 'RMSE = ', np.sqrt(np.mean(trainErrs**2))
    
    #use level estimated returns vectors to infer target direction 
    targetDirection = np.array([math.copysign(1,y_test_TargetScaled[i]) for i in range(len(y_test_TargetScaled))])
    estTargetDirection = np.array([math.copysign(1,y_test_Predict[i]) for i in range(len(y_test_Predict))])
    
    print('Test MLP Level Est - Classification Results')                                                                     
    print(Feedforward_MLP_NetworkName_extended+": Guessed {} out of {} = {}% correct".format(
        np.sum(targetDirection == estTargetDirection), targetDirection.size, 100*np.sum(targetDirection == estTargetDirection)/targetDirection.size
    ))

    strategyPNL,buyHoldPNL,accumulationPNL = util.getIntraDayPNL(accumulationClosePrices,y_test_Predict,[npArrayTargetOpenPrices[i] for i in testRange],[npArrayTargetPrices[i] for i in testRange],dailyInterestRate,testStrategy)
    customPlot.plotPerformance([targetPricesDates[i] for i in testRange],accumulationPNL,strategyPNL,'Buy-Hold','Strategy','PNL Chart',testStrategy + ' Strategy - Test MLP')                                
    print(Feedforward_MLP_NetworkName_extended+": Strategy PNL = {}, Buy-Hold PNL = {}".format(
        strategyPNL[-1], accumulationPNL[-1]
    )) 
        
#     #tranform target to prices  
#     if(isIntraDayClassification):
#         estPrices,temp = util.transformIntraDayPrices(npArrayTargetOpenPrices,
#                                                      npArrayEstTarget,
#                                                      targetDates,
#                                                      returnsCalcOption)
#     else:
#         estPrices,temp = util.transformPrices(npArrayTargetPrices[0],
#                                               npArrayEstTarget,
#                                               targetDates,
#                                               returnsCalcOption)
#          
#     npArrayEstTargetPrices = np.array(estPrices)
#     #plot estimated target prices performance
#     customPlot.plotPerformance(targetPricesDates,npArrayTargetPrices,npArrayEstTargetPrices,'True','Est',targetLastPriceLabel,'Price Performance - MLP')     
                   
#     targetPricesErrs = npArrayTargetPrices - npArrayEstTargetPrices
#     customPlot.plotHist([targetPricesErrs.tolist()],binCount,[targetLastPriceLabel+'_ERROR'],'Target Prices Error - MLP')
#      
#     print 'Feedforward NN Results - Intra Day Price Performance'
#     print 'Mean err = ',np.mean(targetPricesErrs)
#     print 'RMSE = ', np.sqrt(np.mean(targetPricesErrs**2))
    
#     #tranform target to prices  
#     if(isIntraDayClassification):
#         adjClosePrices,temp = util.transformIntraDayAdjPrices(npArrayTargetOpenPrices,
#                                                              npArrayTargetPrices,
#                                                              npArrayEstTarget,
#                                                              targetDates,
#                                                              returnsCalcOption)
#  
#     npArrayAdjClosePrices = np.array(adjClosePrices)
#          
#     #plot estimated target prices performance
#     customPlot.plotPerformance(targetPricesDates,npArrayTargetPrices,npArrayAdjClosePrices,'True','Est',targetLastPriceLabel,'Adjusted Close Price Performance - MLP')                                
#     strategyPNL,buyHoldPNL = util.getIntraDayPNL2(npArrayEstTarget,npArrayTargetOpenPrices,npArrayTargetPrices,None,'LONG_SHORT')
#     customPlot.plotPerformance(targetPricesDates,buyHoldPNL,strategyPNL,'BuyHold','Strategy','PNL Chart','Long Short Strategy - MLP')                                
#  
#     targetPricesErrs = npArrayTargetPrices - npArrayAdjClosePrices
#     customPlot.plotHist([targetPricesErrs.tolist()],binCount,[targetLastPriceLabel+'_ERROR'],'Target Close Prices Error - MLP')
#       
#     print 'Feedforward NN Results - Intra Day Close Price Performance'
#     print 'Mean err = ',np.mean(targetPricesErrs)
#     print 'RMSE = ', np.sqrt(np.mean(targetPricesErrs**2))
        
runGRNN = 1
if(runGRNN):
    #grnnStd = np.linspace(0.5, 3, 50)    
    grnnStd = [1.25]
    GRNN_NetworkName = 'GRNN_Network'                      
    trainNetwork = 1
    if(trainNetwork):  
        print('GRNN Classification Results - Test Std dev input')  
        for x in grnnStd:
            grnnNW = algorithms.GRNN(std=x, verbose=True)
            grnnNW.train(x_train, y_train)
            
            getData.save_network(grnnNW,GRNN_NetworkName)               
            
        y_train_Predict = target_scaler.inverse_transform(grnnNW.predict(x_train))
        y_train_TargetScaled = target_scaler.inverse_transform(y_train[:,0])
        trainErrs = y_train_TargetScaled - y_train_Predict[:,0]
    
        print 'Trained GRNN Results - Intra Day Returns Performance'
        print 'Mean err = ',np.mean(trainErrs)
        print 'RMSE = ', np.sqrt(np.mean(trainErrs**2))
        
        #use level estimated returns vectors to infer target direction 
        targetDirection = np.array([math.copysign(1,y_train_TargetScaled[i]) for i in range(len(y_train_TargetScaled))])
        estTargetDirection = np.array([math.copysign(1,y_train_Predict[i]) for i in range(len(y_train_Predict))])
        
        print('Trained GRNN Level Est - Classification Results')                                                                     
        print(GRNN_NetworkName+": Guessed {} out of {} = {}% correct".format(
            np.sum(targetDirection == estTargetDirection), targetDirection.size, 100*np.sum(targetDirection == estTargetDirection)/targetDirection.size
        ))
    else:
        grnnNW = getData.load_network(GRNN_NetworkName)  
          
    grnnNW.plot_errors()      

    y_test_Predict = target_scaler.inverse_transform(grnnNW.predict(x_test))
    y_test_TargetScaled = target_scaler.inverse_transform(y_test[:,0])
    trainErrs = y_test_TargetScaled - y_test_Predict[:,0]

    customPlot.plotHist([trainErrs.tolist()],binCount,[targetLabel +'_ERROR'],'Target Returns Error - Test GRNN')
    customPlot.plotPerformance([targetDates[i] for i in testRange],y_test_TargetScaled,y_test_Predict,'True','Est',targetLabel,'Returns Performance - Test GRNN')                  
        
    print 'Test GRNN Results - Intra Day Returns Performance'
    print 'Mean err = ',np.mean(trainErrs)
    print 'RMSE = ', np.sqrt(np.mean(trainErrs**2))
    
    #use level estimated returns vectors to infer target direction 
    targetDirection = np.array([math.copysign(1,y_test_TargetScaled[i]) for i in range(len(y_test_TargetScaled))])
    estTargetDirection = np.array([math.copysign(1,y_test_Predict[i]) for i in range(len(y_test_Predict))])
    
    print('Test GRNN Level Est - Classification Results')                                                                     
    print(GRNN_NetworkName+": Guessed {} out of {} = {}% correct".format(
        np.sum(targetDirection == estTargetDirection), targetDirection.size, 100*np.sum(targetDirection == estTargetDirection)/targetDirection.size
    ))

    strategyPNL,buyHoldPNL,accumulationPNL = util.getIntraDayPNL(accumulationClosePrices,y_test_Predict,[npArrayTargetOpenPrices[i] for i in testRange],[npArrayTargetPrices[i] for i in testRange],dailyInterestRate,testStrategy)
    customPlot.plotPerformance([targetPricesDates[i] for i in testRange],accumulationPNL,strategyPNL,'Buy-Hold','Strategy','PNL Chart',testStrategy +' Strategy -  Test GRNN')                                
    print(GRNN_NetworkName+": Strategy PNL = {}, Buy-Hold PNL = {}".format(
        strategyPNL[-1], accumulationPNL[-1]
    ))  
        
#     #tranform target to prices  
#     if(isIntraDayClassification):
#         estPrices,temp = util.transformIntraDayPrices(npArrayTargetOpenPrices,
#                                                      npArrayEstTarget,
#                                                      targetDates,
#                                                      returnsCalcOption)
#     else:
#         estPrices,temp = util.transformPrices(npArrayTargetPrices[0],
#                                               npArrayEstTarget,
#                                               targetDates,
#                                               returnsCalcOption)
#         
#     npArrayEstTargetPrices = np.array(estPrices)
#     #plot estimated target prices performance
#     customPlot.plotPerformance(targetPricesDates,npArrayTargetPrices,npArrayEstTargetPrices,'True','Est',targetLastPriceLabel,'Price Performance - GRNN')                                
#     
#     targetPricesErrs = npArrayTargetPrices - npArrayEstTargetPrices
#     customPlot.plotHist([targetPricesErrs.tolist()],binCount,[targetLastPriceLabel+'_ERROR'],'Target Prices Error - GRNN')
#     
#     print 'GRNN Results - Intra Day Price Performance'
#     print 'Mean err = ',np.mean(targetPricesErrs)
#     print 'RMSE = ', np.sqrt(np.mean(targetPricesErrs**2))

runOLS = 1
if(runOLS):
    OLS_ObjName = 'OLS_LevelPredictor_ext'
    trainOLS = 1
    predictorInputs.append(['ASX200_INDX_GROSS_DAILY_DIV',usePrice])
    predictorInputData,predictorLabels =  setup.createPredictorVariables(dataContainer,predictorInputs,returnsCalcOption)
    #convert all targets to numpy arrays
    npArrayPredictorInputData = np.transpose(np.array(predictorInputData))
    predictorInputDataScaled =  predictorInputData_scaler.fit_transform(npArrayPredictorInputData)

    firstIdxTest = int(math.ceil(len(targetDates)*trainTestSplit))
    trainRange = range(0,firstIdxTest)
    testRange = range(firstIdxTest,len(targetDates))
     
    x_train = predictorInputDataScaled[trainRange,:]
    y_train = targetScaled[trainRange]
     
    x_test = predictorInputDataScaled[testRange,:]
    y_test = targetScaled[testRange]
    
    x_train = sm.add_constant(x_train)
    x_test = sm.add_constant(x_test)
        
    if(trainOLS): 
        model = sm.OLS(y_train, x_train)
        OLS_LevelPredictor = model.fit()  
        getData.save_obj(OLS_LevelPredictor,OLS_ObjName)         
        print(OLS_LevelPredictor.summary())
        
        y_train_Predict = target_scaler.inverse_transform(OLS_LevelPredictor.predict(x_train))
        y_train_TargetScaled = target_scaler.inverse_transform(y_train[:,0])
        trainErrs = y_train_TargetScaled - y_train_Predict
    
        customPlot.plotHist([trainErrs.tolist()],binCount,[targetLabel +'_ERROR'],'Target Returns Error - Trained OLS')
        customPlot.plotPerformance([targetDates[i] for i in trainRange],y_train_TargetScaled,y_train_Predict,'True','Est',targetLabel,'Returns Performance - Trained OLS')                  
        
        print 'Trained OLS Results - Intra Day Returns Performance'
        print 'Mean err = ',np.mean(trainErrs)
        print 'RMSE = ', np.sqrt(np.mean(trainErrs**2))
        
        #use level estimated returns vectors to infer target direction 
        targetDirection = np.array([math.copysign(1,y_train_TargetScaled[i]) for i in range(len(y_train_TargetScaled))])
        estTargetDirection = np.array([math.copysign(1,y_train_Predict[i]) for i in range(len(y_train_Predict))])
        
        print('Trained OLS Level Est - Classification Results')                                                                     
        print(OLS_ObjName+": Guessed {} out of {} = {}% correct".format(
            np.sum(targetDirection == estTargetDirection), targetDirection.size, 100*np.sum(targetDirection == estTargetDirection)/targetDirection.size
        ))
        
    else:
        OLS_LevelPredictor = getData.load_obj(OLS_ObjName)         

    y_test_Predict = target_scaler.inverse_transform(OLS_LevelPredictor.predict(x_test))
    y_test_TargetScaled = target_scaler.inverse_transform(y_test[:,0])
    trainErrs = y_test_TargetScaled - y_test_Predict
    
    customPlot.plotHist([trainErrs.tolist()],binCount,[targetLabel +'_ERROR'],'Target Returns Error - Test OLS')
    customPlot.plotPerformance([targetDates[i] for i in testRange],y_test_TargetScaled,y_test_Predict,'True','Est',targetLabel,'Returns Performance - Test OLS')                  
        
    print 'Test OLS Results - Intra Day Returns Performance'
    print 'Mean err = ',np.mean(trainErrs)
    print 'RMSE = ', np.sqrt(np.mean(trainErrs**2))
    
    #use level estimated returns vectors to infer target direction 
    targetDirection = np.array([math.copysign(1,y_test_TargetScaled[i]) for i in range(len(y_test_TargetScaled))])
    estTargetDirection = np.array([math.copysign(1,y_test_Predict[i]) for i in range(len(y_test_Predict))])
    
    print('Test OLS Level Est - Classification Results')                                                                     
    print(OLS_ObjName+": Guessed {} out of {} = {}% correct".format(
        np.sum(targetDirection == estTargetDirection), targetDirection.size, 100*np.sum(targetDirection == estTargetDirection)/targetDirection.size
    ))

    strategyPNL,buyHoldPNL,accumulationPNL = util.getIntraDayPNL(accumulationClosePrices,y_test_Predict,[npArrayTargetOpenPrices[i] for i in testRange],[npArrayTargetPrices[i] for i in testRange],dailyInterestRate,testStrategy)
    customPlot.plotPerformance([targetPricesDates[i] for i in testRange],accumulationPNL,strategyPNL,'Buy-Hold','Strategy','PNL Chart',testStrategy + ' Strategy - Test OLS')                                
    print(OLS_ObjName+": Strategy PNL = {}, Buy-Hold PNL = {}".format(
        strategyPNL[-1], accumulationPNL[-1]
    ))   