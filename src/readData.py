'''
Created on 11 May 2016

@author: Joel
'''

import openpyxl
try :
    import cPickle as pickle
except:
    import pickle

# Store data (serialize)
def save_obj(obj, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# Load data (deserialize)
def load_obj(filename ):
    with open(filename + '.pickle', 'rb') as f:
        return pickle.load(f)

def  readExcelSpreadsheet_SaveInputs(filePath,workbookSheetNames,firstRowOfCodesPerSheet,dateFormatIn):

    workbookCodes = {}
    workbookNumericData = {}
    workbookDates = {}

    xl_workbook = openpyxl.load_workbook(filePath)

    for workSheetName in workbookSheetNames:
        sheet = xl_workbook.get_sheet_by_name(workSheetName)
        data = []
        codes = []
        dates = []

        i = 1
        
        for row in sheet.iter_rows():
            rowData = []
            if i == firstRowOfCodesPerSheet[workbookSheetNames.index(workSheetName)]:
                for cell in row:
                    if cell.column > 'A':
                        codes.append(cell.value)
            elif i > firstRowOfCodesPerSheet[workbookSheetNames.index(workSheetName)]:
                for cell in row:
                    if cell.column > 'A':
                        rowData.append(cell.value)
                    else:
                        dates.append(cell.value)
                        
                data.append(rowData)
                
            i = i+1
                      
        workbookNumericData[workSheetName] = data
        workbookCodes[workSheetName] = codes
        workbookDates[workSheetName] = dates   
    
    save_obj(workbookNumericData, 'ASX_ForecastNN_Numeric')
    save_obj(workbookCodes, 'ASX_ForecastNN_Unicodes')
    save_obj(workbookDates, 'ASX_ForecastNN_Dates')
    
if __name__ == "__main__":
    
    fileName = "ASX_ForecastNN_Data.xlsx"
    
    valuationDate_dateFormat = "1/04/2016";

    
    workbookSheetNames = ['EXCHANGE RATES',
                          'USD EXRATES AND GOLD',
                          'CM YIELDS GB',
                          'IR AND YIELDS - MONEY MARKET',
                          'COMMODITY PRICES',
                          'GDP AND INCOME',
                          'CPI',
                          'US T-Bill 3MO',
                          'US T-Bill 1Y',
                          'S&P ASX 200 (XJO)',
                          'Indices'
                          ]   
    firstRowOfCodesPerSheet = [11,
                               11,
                               11,
                               11,
                               11,
                               11,
                               11,
                               6,
                               6,
                               6,
                               2,
                                ]
    readExcelSpreadsheet_SaveInputs(fileName,workbookSheetNames,firstRowOfCodesPerSheet,valuationDate_dateFormat)

    ASX_ForecastNN_Numeric = load_obj('ASX_ForecastNN_Numeric')
    ASX_ForecastNN_Unicodes = load_obj('ASX_ForecastNN_Unicodes')
    ASX_ForecastNN_Dates = load_obj('ASX_ForecastNN_Dates')
    
    