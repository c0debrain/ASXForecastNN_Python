
import readSaveData as getData
import datetime  
import code
import plotting as customPlot
import numpy as np
import utilities as util
import matplotlib.pyplot as plt
import math

from sklearn import datasets, preprocessing
from sklearn.cross_validation import train_test_split
from neupy import algorithms, layers, estimators, environment
from numbers import Real

environment.reproducible()

showPlots = 0;

# 1 means re-read xlsx spreadhseet and then save data objects based on spreadsheet details below
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

labels = []
pricesData = []
pricesDates = []
returnsData = []
returnsDates = []

for inputChoice in pricesInputs:
    labels.append(inputChoice[dataPlotLabelsIdx])
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
    pricesData.append(requestedData[numericDataIndex])
    pricesDates.append(requestedData[datesIndex])

    rets,retsDates = util.getReturns(requestedData[numericDataIndex],requestedData[datesIndex],returnsCalcOption)
    returnsData.append(rets)
    returnsDates.append(retsDates)
 
    assert(pricesData[len(pricesData)-1].count(None) == 0),"Gathered Price Data: '" + labels[len(labels)-1] + "' contains missing values"
    assert(returnsData[len(returnsData)-1].count(None) == 0),"Computed Returns Data: '" + labels[len(labels)-1] + "' contains missing values"
    assert(pricesDates[len(pricesDates)-1].count(None) == 0),"Dates for: '" + labels[len(labels)-1] + "' contains missing values"

if (showPlots):
    customPlot.subPlotData(pricesDates,pricesData,labels,'Prices Data')
    customPlot.correlationHeatMap(pricesData,labels,'Prices Correlation Heatmap')

targetLabel = 'ASX200_INTRADAY_'+returnsCalcOption+'_RETS'
targetOpenPriceLabel = 'ASX200_DAILY_PX_OPEN'
targetLastPriceLabel = 'ASX200_DAILY_PX_LAST'
    
#targetChoice = 'LEVEL'
targetChoice = 'DIRECTION'
if(targetChoice == 'DIRECTION'):
    classifyDirection = 'SINGLE_VEC' #single_vec

if targetLabel in labels:
    target = returnsData[labels.index(targetLabel)]
    if(targetChoice == 'DIRECTION'):
        target = [util.classifyReturnDirection(math.copysign(1,target[i]),classifyDirection) for i in range(len(target))]
    targetDates = returnsDates[labels.index(targetOpenPriceLabel)]    
else:
    target,targetDates = util.getIntraDayReturns(pricesData[labels.index(targetOpenPriceLabel)],pricesData[labels.index(targetLastPriceLabel)],pricesDates[labels.index(targetLastPriceLabel)],returnsCalcOption)
    if(targetChoice == 'DIRECTION'):
        target = [util.classifyReturnDirection(math.copysign(1,target[i]),classifyDirection) for i in range(len(target))]
    returnsData.append(target)
    returnsDates.append(targetDates)
    labels.append(targetLabel)

predictorInputs = []
useReturn = 0
usePrice = 1

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

predictorInputData = []
predictorLabels = []

for predictor in predictorInputs:
    dataIdx = labels.index(predictor[0])
    if(predictor[1] == usePrice):
        predictorLabels.append(labels[dataIdx])
        #get copy of prices data and
        #remove first price to match number of inputs from returns vectors
        copyPricesData = pricesData[dataIdx][:]
        copyPricesData.pop(0)
        predictorInputData.append(copyPricesData)
    elif(predictor[1] == useReturn):
        predictorLabels.append(labels[dataIdx]+'_'+ returnsCalcOption+'_RETS')
        predictorInputData.append(returnsData[dataIdx])

binCount = 50
if (showPlots):
    customPlot.plotHist(returnsData,binCount,labels,'Returns Data Histogram')
    customPlot.plotHist(predictorInputData,binCount,predictorLabels,'Predictor Variable Histogram')

# check all predictor variables allocated correctly 
# then scale inputs and train

# scale inputs and output
predictorInputData_scaler = preprocessing.StandardScaler()
target_scaler = preprocessing.MinMaxScaler()

npArrayPredictorInputData = np.transpose(np.array(predictorInputData))
npArrayTarget = np.transpose(np.array(target))

# x_train = npArrayPredictorInputData
# y_train = npArrayTarget
# 
# x_test = x_train
# y_test = y_train

x_train, x_test, y_train, y_test = train_test_split(
    predictorInputData_scaler.fit_transform(npArrayPredictorInputData),
    target_scaler.fit_transform(npArrayTarget),
    train_size=0.85
 )

#hiddenNeurons = [5,10,15,20,25,30,40,50,60]
hiddenNeurons = [22]

for x in hiddenNeurons:
    cgnet = algorithms.ConjugateGradient(
        connection=[
            layers.Tanh(len(predictorInputData)),
            layers.Tanh(x),
            layers.Output(1),
        ],
        search_method='golden',
        update_function='fletcher_reeves',
        addons=[algorithms.LinearSearch],
        verbose=False
    )
    cgnet.train(x_train, y_train, x_test, y_test, epochs=10)

    cgnet.plot_errors()
    
    y_predict = cgnet.predict(x_test)
    #real = target_scaler.inverse_transform(y_test)
    #predicted = target_scaler.inverse_transform(y_predict)
     
    #error = estimators.rmsle(real, predicted)
    #print error
    error = estimators.rmsle(y_test, y_predict)
    
    print 'Hidden neurons:\t', x, '\tError:\t', round(error,3)


#pnnStd = np.linspace(0.5, 3, 51)
pnnStd = [1.25]

for x in pnnStd:
    nw = algorithms.PNN(std=x, verbose=False)
    nw.train(x_train, y_train)
    #nw.plot_errors()

    y_predict = nw.predict(x_test)
    #real = target_scaler.inverse_transform(y_test)
    #predicted = target_scaler.inverse_transform(y_predict)
     
    #error = estimators.rmsle(real, predicted)
    #print error
    error = estimators.rmsle(y_test, y_predict)
    print 'Std dev:\t', x, '\tError:\t', round(error,3)
