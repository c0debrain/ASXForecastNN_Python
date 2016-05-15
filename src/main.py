
import matplotlib.pyplot as plt
import readSaveData as getData
import datetime  
import code

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

#dateFormatIn = "%d/%m/%Y"
#choose data between these dates 
startDate = datetime.datetime(1999, 12, 31)
finishDate = datetime.datetime(2012, 04, 28)

dataPlotLabels = 0
dataWorksheet = 1
dataCode = 2
dataInputs = []

# choice of data added to list given by information below
# ['data label name', 'worksheet name of data' , 'code string']
# i.e. ['FX AUD/USD 1MO($)','EXCHANGE RATES','FXRUSD']
# 'FX AUD/USD 1MO($)' = the data label for AUD USD FX rates, to be used for plots 
# 'EXCHANGE RATES' = the name of the worksheet where the data is found
# 'FXRUSD' = the code string of the data 
dataInputs.append(['FX AUD/USD 1MO($)','EXCHANGE RATES','FXRUSD'])
dataInputs.append(['FX AUD/JY 1MO($)','EXCHANGE RATES','FXRJY'])
dataInputs.append(['FX AUD/CNY 1MO($)','EXCHANGE RATES','FXRCR'])
dataInputs.append(['FX AUD/EUR 1MO($)','EXCHANGE RATES','FXREUR'])
dataInputs.append(['USD GOLD','USD EXRATES AND GOLD','FUSXRGP'])
dataInputs.append(['AU GOVBONDS INDEX','CM YIELDS GB','FCMYGBAGI'])
dataInputs.append(['AU GOVBONDS 2Y','CM YIELDS GB','FCMYGBAG2'])
dataInputs.append(['IB ON CASHRATE','IR AND YIELDS - MONEY MARKET','FIRMMCRI'])
dataInputs.append(['BANKBILLS 3MO','IR AND YIELDS - MONEY MARKET','FIRMMBAB90'])
dataInputs.append(['COMMOD INDEX ALL','COMMODITY PRICES','GRCPAIAD'])
dataInputs.append(['US TBILL 3MO','US T-Bill 3MO','RIFLGFCM03_N.M'])
dataInputs.append(['US TBILL 1Y','US T-Bill 1Y','RIFLGFCY01_N.M'])
dataInputs.append(['ASX200_DAILY_PX_OPEN','Indices','AS51_PX_OPEN'])
dataInputs.append(['S&P500_DAILY_PX_LAST','Indices','SPX_PX_LAST'])

choice = []
pricesData = []
datesData = []

for inputChoice in dataInputs:
    choice.append(inputChoice[dataPlotLabels])
    code = inputChoice[dataCode]
    dataLocation = inputChoice[dataWorksheet]

    requestedData = getData.returnColDataInPeriod(startDate,finishDate,dataLocation,code,ASX_ForecastNN_SheetNames,ASX_ForecastNN_Dates,ASX_ForecastNN_Unicodes,ASX_ForecastNN_Numeric)
    pricesData.append(requestedData[numericDataIndex])
    datesData.append(requestedData[datesIndex])
    
               
plt.figure(1)
plt.subplot(221)
plt.plot(datesData[choice.index('FX AUD/USD 1MO($)')], pricesData[choice.index('FX AUD/USD 1MO($)')], 'b')
plt.xlabel('Time')
plt.ylabel(dataInputs[choice.index('FX AUD/USD 1MO($)')][dataPlotLabels])
plt.grid(True)

plt.subplot(222)
plt.plot(datesData[choice.index('BANKBILLS 3MO')], pricesData[choice.index('BANKBILLS 3MO')], 'r')
plt.xlabel('Time')
plt.ylabel(dataInputs[choice.index('BANKBILLS 3MO')][dataPlotLabels])
plt.grid(True)

plt.subplot(223)
plt.plot(datesData[choice.index('USD GOLD')], pricesData[choice.index('USD GOLD')], 'r')
plt.xlabel('Time')
plt.ylabel(dataInputs[choice.index('USD GOLD')][dataPlotLabels])
plt.grid(True)

plt.subplot(224)
plt.plot(datesData[choice.index('ASX200_DAILY_PX_OPEN')], pricesData[choice.index('ASX200_DAILY_PX_OPEN')], 'r')
plt.xlabel('Time')
plt.ylabel(dataInputs[choice.index('ASX200_DAILY_PX_OPEN')][dataPlotLabels])
plt.grid(True)

plt.show()
    
     