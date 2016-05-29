
import readSaveData as getData
import datetime  
import code
import plotting as customPlot
import numpy as np
import utilities as util
import matplotlib.pyplot as plt
import math
import networkTrainingSetup as setup

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
#endDate = datetime.datetime(2003, 4, 29)
#startDate = datetime.datetime(2000, 1, 4)
startDate = datetime.datetime(2011, 5, 9)
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

useReturn = 0
usePrice = 1
predictorInputs = []
predictorInputs.append(['S&P500_DAILY_PX_LAST',useReturn])
predictorInputs.append(['STFINL_DAILY_PX_LAST',useReturn])
predictorInputs.append(['SHCOMP_DAILY_PX_LAST',useReturn])
predictorInputs.append(['ASX200_INDX_GROSS_DAILY_DIV',usePrice])
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
   
#targetChoice = 'DIRECTION'
if(targetChoice == 'DIRECTION'):
    targetLabel = targetLabel +'_DIR'
    classifyDirection = 'DUAL_VEC'
else:
    classifyDirection = None

isIntraDayClassification = 0

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
if(isIntraDayClassification):
    npArrayTargetOpenPrices = np.array(targetOpenPrices)

predictorInputData,predictorLabels =  setup.createPredictorVariables(dataContainer,predictorInputs,returnsCalcOption)
#convert all targets to numpy arrays
npArrayPredictorInputData = np.transpose(np.array(predictorInputData))

binCount = 50
if(showPlots):
    customPlot.plotHist(dataContainer.returnsData,binCount,dataContainer.returnsLabels,'Returns Data Histogram')
    customPlot.plotHist(predictorInputData,binCount,predictorLabels,'Predictor Variable Histogram')
    customPlot.plotHist([target],binCount,[targetLabel],'Target Histogram')

# check all predictor variables allocated correctly 
# scale inputs and output and train
#standardise inputs 
predictorInputData_scaler = preprocessing.StandardScaler()
#0.6585 is the max of second derivative of tanh function, yields better training results
target_scaler = preprocessing.MinMaxScaler(feature_range=(-0.6585, 0.6585))

# split and apply pre-processing to data
x_train, x_test, y_train, y_test = train_test_split(
    predictorInputData_scaler.fit_transform(npArrayPredictorInputData),
    target_scaler.fit_transform(npArrayTarget),
    train_size=0.6
)

checkInputScaling = 0
if(checkInputScaling):
    #check the scaling and standardising of inputs 
    predictorInputDataScaled = np.transpose(predictorInputData_scaler.fit_transform(npArrayPredictorInputData))
    targetScaled = target_scaler.fit_transform(npArrayTarget)
    customPlot.plotHist(predictorInputDataScaled.tolist(),binCount,predictorLabels,'Predictor Variables Scaled')
    customPlot.plotHist([targetScaled.tolist()],binCount,[targetLabel],'Target Scaled')

# checkReturnsToPricesCalcs = 1
# if(checkReturnsToPricesCalcs):
#     test,testDates = util.transformPrices(dataContainer.pricesData[dataContainer.pricesLabels.index('S&P500_DAILY_PX_LAST')][0],
#                                           dataContainer.returnsData[dataContainer.returnsLabels.index('S&P500_DAILY_PX_LAST')],
#                                           dataContainer.returnsDates[dataContainer.returnsLabels.index('S&P500_DAILY_PX_LAST')],
#                                           returnsCalcOption)
#      
#     print 'Calc Prices Sum error = ' + str(np.sum(np.array(test) - np.array(dataContainer.pricesData[dataContainer.pricesLabels.index('S&P500_DAILY_PX_LAST')])))
#      
#     test,testDates = util.transformIntraDayPrices(dataContainer.pricesData[dataContainer.pricesLabels.index(targetOpenPriceLabel)],
#                                           dataContainer.returnsData[dataContainer.returnsLabels.index(targetLabel)],
#                                           dataContainer.returnsDates[dataContainer.returnsLabels.index(targetLabel)],
#                                           returnsCalcOption)
#      
#     print 'Calc Intraday Prices Sum error = ' + str(np.sum(np.array(test) - np.array(dataContainer.pricesData[dataContainer.pricesLabels.index(targetLastPriceLabel)])))

numHiddenNeurons = 22
Feedforward_MLP_NetworkName = 'Feedforward_MLP_'+ targetChoice    
trainNetwork = 0                   
 
if(trainNetwork):   
    print('Feedforward MLP Level Results')                                                                                     
    cgnet = algorithms.ConjugateGradient(
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
          show_epoch=50,
          )
         
    cgnet.train(x_train, y_train, x_test,y_test,epochs=350)
    #not working, cant have numbers in name
    #networkName = 'ConjGrad_tanh<'+str(npArrayPredictorInputData.shape[1])+'>_tanh<'+str(numHiddenNeurons)+'>_output<'+str(npArrayTarget.shape[1])+'>'
    getData.save_network(cgnet,Feedforward_MLP_NetworkName)   
else:
    cgnet = getData.load_network(Feedforward_MLP_NetworkName)
    
cgnet.plot_errors()

# trueTrainTarget = np.transpose(target_scaler.inverse_transform(y_train))
# estrainTarget = np.transpose(target_scaler.inverse_transform(cgnet.predict(x_train)))
# customPlot.subPlotData(pricesDates,pricesData,pricesLabels,'Prices Data')
# 
# trueTestTarget = np.transpose(target_scaler.inverse_transform(y_test))
# estTestTarget = np.transpose(target_scaler.inverse_transform(cgnet.predict(x_test)))
# customPlot.subPlotData(pricesDates,pricesData,pricesLabels,'Prices Data')

npArrayEstTarget = target_scaler.inverse_transform(cgnet.predict(predictorInputData_scaler.fit_transform(npArrayPredictorInputData)))

#plot estimated target performance
plt.figure
p1, = plt.plot(targetDates, npArrayTarget,'b')
p2, = plt.plot(targetDates, npArrayEstTarget,'r')
plt.xlabel('Time')
plt.ylabel(targetLabel)
plt.grid(True)       
plt.legend([p1, p2], ['True','Est'])
plt.title('Returns Performance - MLP')
plt.show()            

#tranform target to prices  
if(isIntraDayClassification):
    estPrices,temp = util.transformIntraDayPrices(npArrayTargetOpenPrices,
                                         npArrayEstTarget,
                                         targetDates,
                                         returnsCalcOption)
else:
    estPrices,temp = util.transformPrices(npArrayTargetPrices[0],
                                          npArrayEstTarget,
                                          targetDates,
                                          returnsCalcOption)
    
npArrayEstTargetPrices = np.array(estPrices)

#plot estimated target prices performance
plt.figure
p1, = plt.plot(targetPricesDates, npArrayTargetPrices,'b')
p2, = plt.plot(targetPricesDates, npArrayEstTargetPrices,'r')
plt.xlabel('Time')
plt.ylabel(targetLastPriceLabel)
plt.grid(True)  
plt.legend([p1, p2], ['True', 'Est'])   
plt.title('Price Performance - MLP')
plt.show()                

targetErrs = npArrayTarget - npArrayEstTarget
customPlot.plotHist([targetErrs.tolist()],binCount,[targetLabel +'_ERROR'],'Target Returns Error - MLP')

print 'Feedforward NN Results - Intra Day Returns Performance'
print 'Mean err = ',np.mean(targetErrs)
print 'RMSE = ', np.sqrt(np.mean(targetErrs**2))

targetPricesErrs = npArrayTargetPrices - npArrayEstTargetPrices
customPlot.plotHist([targetPricesErrs.tolist()],binCount,[targetLastPriceLabel+'_ERROR'],'Target Prices Error - MLP')

print 'Feedforward NN Results - Intra Day Price Performance'
print 'Mean err = ',np.mean(targetPricesErrs)
print 'RMSE = ', np.sqrt(np.mean(targetPricesErrs**2))

#grnnStd = np.linspace(0.5, 2, 200)    
grnnStd = [1.8]
GRNN_NetworkName = 'GRNN_Network'                      
trainNetwork = 1
if(trainNetwork):  
    print('GRNN Classification Results - Test Std dev input')  
    for x in grnnStd:
        nw = algorithms.GRNN(std=x, verbose=False)
        nw.train(x_train, y_train)
       
        getData.save_network(nw,GRNN_NetworkName)
    
        y_testEst = target_scaler.inverse_transform(nw.predict(x_test))
        y_testTrue = target_scaler.inverse_transform(y_test)
        
        print("Std dev {}: RMSE = {}".format(
            x, np.sqrt(np.mean((y_testEst-y_testTrue)**2)))
        )  
else:
    nw = getData.load_network(GRNN_NetworkName)  
      
npArrayEstTarget = target_scaler.inverse_transform(nw.predict(predictorInputData_scaler.fit_transform(npArrayPredictorInputData)))

#plot estimated target performance
plt.figure
p1, = plt.plot(targetDates, npArrayTarget,'b')
p2, = plt.plot(targetDates, npArrayEstTarget,'r')
plt.xlabel('Time')
plt.ylabel(targetLabel)
plt.grid(True)       
plt.legend([p1, p2], ['True','Est'])
plt.title('Returns Performance - GRNN')
plt.show()            

#tranform target to prices  
if(isIntraDayClassification):
    estPrices,temp = util.transformIntraDayPrices(npArrayTargetOpenPrices,
                                         npArrayEstTarget,
                                         targetDates,
                                         returnsCalcOption)
else:
    estPrices,temp = util.transformPrices(npArrayTargetPrices[0],
                                          npArrayEstTarget,
                                          targetDates,
                                          returnsCalcOption)
    
npArrayEstTargetPrices = np.array(estPrices)

#plot estimated target prices performance
plt.figure
p1, = plt.plot(targetPricesDates, npArrayTargetPrices,'b')
p2, = plt.plot(targetPricesDates, npArrayEstTargetPrices,'r')
plt.xlabel('Time')
plt.ylabel(targetLastPriceLabel)
plt.grid(True)  
plt.legend([p1, p2], ['True', 'Est'])   
plt.title('Price Performance - GRNN')
plt.show()                

targetErrs = npArrayTarget - npArrayEstTarget
customPlot.plotHist([targetErrs.tolist()],binCount,[targetLabel +'_ERROR'],'Target Returns Error - GRNN')

print 'GRNN Results - Intra Day Returns Performance'
print 'Mean err = ',np.mean(targetErrs)
print 'RMSE = ', np.sqrt(np.mean(targetErrs**2))

targetPricesErrs = npArrayTargetPrices - npArrayEstTargetPrices
customPlot.plotHist([targetPricesErrs.tolist()],binCount,[targetLastPriceLabel+'_ERROR'],'Target Prices Error - GRNN')

print 'GRNN Results - Intra Day Price Performance'
print 'Mean err = ',np.mean(targetPricesErrs)
print 'RMSE = ', np.sqrt(np.mean(targetPricesErrs**2))