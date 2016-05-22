
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

environment.reproducible()

# 0 means re-read xlsx spreadhseet and then save data objects based on spreadsheet details below
# otherwise use previously saved data objects (.pickle files) as inputs 
loadData = 1;

if(loadData == 0):
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
returnsCalcOption = 'LOG_DIFF'

#choose data between these dates 
# startDate = datetime.datetime(2000, 4, 6)
# endDate = datetime.datetime(2016, 4, 29)
startDate = datetime.datetime(2016, 4, 4)
endDate = datetime.datetime(2016, 5, 2)

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
pricesInputs.append(['ASX200_INDX_GROSS_DAILY_DIV','Indices','AS51_INDX_GROSS_DAILY_DIV',lagAsianMarketClose])
pricesInputs.append(['ASX200_INDX_NET_DAILY_DIV','Indices','AS51_INDX_NET_DAILY_DIV',lagAsianMarketClose])

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

customPlot.subPlotData(pricesDates,pricesData,labels)
customPlot.correlationHeatMap(pricesData,labels)

targetLabel = 'ASX200_INTRADAY_'+returnsCalcOption+'_RETS'
targetOpenPriceLabel = 'ASX200_DAILY_PX_OPEN'
targetLastPriceLabel = 'ASX200_DAILY_PX_LAST'
    
targetChoice = 'LEVEL'

if targetLabel in labels:
    target = returnsData[labels.index(targetLabel)]
    if(targetChoice == 'DIRECTION'):
        target = [math.copysign(1,target[i]) for i in range(len(target))]
    targetDates = returnsDates[labels.index(targetOpenPriceLabel)]    
else:
    target,targetDates = util.getIntraDayReturns(pricesData[labels.index(targetOpenPriceLabel)],pricesData[labels.index(targetLastPriceLabel)],pricesDates[labels.index(targetLastPriceLabel)],returnsCalcOption)
    if(targetChoice == 'DIRECTION'):
        target = [math.copysign(1,target[i]) for i in range(len(target))]
    returnsData.append(target)
    returnsDates.append(targetDates)
    labels.append(targetLabel)

predictorInputs = []
useReturn = 0
usePrice = 1

predictorInputs.append(['S&P500_DAILY_PX_LAST',useReturn])
predictorInputs.append(['STFINL_DAILY_PX_LAST',useReturn])
predictorInputs.append(['SHCOMP_DAILY_PX_LAST',useReturn])
predictorInputs.append(['ASX200_INDX_NET_DAILY_DIV',usePrice])

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
customPlot.plotHist(returnsData,binCount,labels)
customPlot.plotHist(predictorInputData,binCount,predictorLabels)

# check all predictor variables allocated correctly 
# then scale inputs and train

# scale inputs and output
predictorInputData_scaler = preprocessing.StandardScaler()
target_scaler = preprocessing.MinMaxScaler()

# x_train, x_test, y_train, y_test = train_test_split(
#     data_scaler.fit_transform(predictorInputData),
#     target_scaler.fit_transform(target),
#      train_size=0.85
#  )
# 
# cgnet = algorithms.ConjugateGradient(
#      connection=[
#          layers.Tanh(13),
#         layers.Tanh(50),
#          layers.RoundedOutput(1, decimals=1),
#      ],
#      search_method='golden',
#      update_function='fletcher_reeves',
#      addons=[algorithms.LinearSearch],
#      verbose=False
#      )
# 
# cgnet.train(x_train, y_train, epochs=100)
# y_predict = cgnet.predict(x_test)
# 
# real = target_scaler.inverse_transform(y_test)
# predicted = target_scaler.inverse_transform(y_predict)
# 
# error = estimators.rmsle(real, predicted)

