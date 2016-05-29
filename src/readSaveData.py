from datetime import datetime
import openpyxl
import dill
try :
    import cPickle as pickle
except:
    import pickle

class dataContainer:
    def __init__(self,pricesData,pricesDates,pricesLabels,returnsData,returnsDates,returnsLabels):
        self.pricesData = pricesData
        self.pricesDates = pricesDates
        self.pricesLabels = pricesLabels
        self.returnsData = returnsData
        self.returnsDates = returnsDates
        self.returnsLabels = returnsLabels   
        
# Store network (serialize)
def save_network(network, filename):
    with open(filename + '.dill', 'wb') as f:
        dill.dump(network, f)

# Load network (deserialize)
def load_network(filename):
    with open(filename+ '.dill', 'rb') as f:
        return dill.load(f)

# Store data (serialize)
def save_obj(obj, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# Load data (deserialize)
def load_obj(filename ):
    with open(filename + '.pickle', 'rb') as f:
        return pickle.load(f)

def checkIncreasingDates(lsDates):
    for i in range(len(lsDates)-1):
        if(lsDates[i] >=  lsDates[i+1]):
            return False        
    return True
    
def returnColDataInPeriod(startDate,finishDate,workSheetName,codeString,workbookSheetNames,workbookDates,workbookCodes,workbookNumericData):
    sheetDates = workbookDates[workSheetName]
    colIndex = workbookCodes[workSheetName].index(codeString)
      
    assert(startDate < finishDate),"Start date is after finish date"
    assert((sheetDates[0] <= startDate) & (sheetDates[len(sheetDates)-1] >= startDate)),"Start date is not in the data list"
    assert((sheetDates[0] <= finishDate) & (sheetDates[len(sheetDates)-1] >= finishDate)),"Finish date is not in the data list"

    rows = [i for i in range(len(sheetDates)) if ((sheetDates[i] >= startDate) & (sheetDates[i] <= finishDate))]
    
    colNumeric = []
    colDates = []
    
    for idx, i in enumerate(rows):   
        colDates.append(sheetDates[i])
        colNumeric.append(workbookNumericData[workSheetName][i][colIndex])
        #get the data entry in that row and check if it is missing
        # if its not the first row, replace with previous otherwise set to 0.0
        if (colNumeric[idx] is None):
            if(idx > 0):
                colNumeric[idx] = colNumeric[idx-1]
            else:
                colNumeric[idx] = 0.0
                     
    requestedColData = []
    requestedColData.append(colDates)
    requestedColData.append(colNumeric)

    return requestedColData


def  readExcelSpreadsheet_SaveInputs(filePath,workbookSheetNames,firstRowOfCodesPerSheet):

    #assumes the workbook (xls spreadsheet) has worksheet with one column of dates in column A
    #that aligns with all data in that worksheet
    #each code above a column of data is a descriptive label of the data column  
    #all data is assumed to be entered column wise with no column spaces in between
    workbookCodes = {}
    workbookNumericData = {}
    workbookDates = {}

    xl_workbook = openpyxl.load_workbook(filePath)

    for workSheetName in workbookSheetNames:
        sheet = xl_workbook.get_sheet_by_name(workSheetName)
        data = []
        codes = []
        dates = []
       
        for row in sheet.iter_rows():
            #check if first cell in row is empty, then its not a valid data entry
            firstCellInRow = row[0]
            if firstCellInRow.value is not None:
                
                if firstCellInRow.row == firstRowOfCodesPerSheet[workbookSheetNames.index(workSheetName)]: 
                    for cell in row:
                        #does the current row for the current worksheet contain the unicodes
                        if cell.row == firstRowOfCodesPerSheet[workbookSheetNames.index(workSheetName)]:
                                #not inclusive of the first column
                                if cell.column > 'A':
                                    codes.append(cell.value)
                #all rows after the codes contain numeric data
                elif firstCellInRow.row > firstRowOfCodesPerSheet[workbookSheetNames.index(workSheetName)]:
                    validRowData = []
                    for cell in row:
                        #not inclusive of the first column
                        if cell.column > 'A':
                            validRowData.append(cell.value)
                        else:
                            #dates are contained in the first column 
                            #format dates 
                            if(isinstance(cell.value,basestring)):
                                dates.append(datetime.strptime(cell.value,'%d/%m/%Y'))
                            else:
                                dates.append(cell.value)
                            
                    data.append(validRowData)
                           
        assert(checkIncreasingDates(dates)),"Worksheet: '" + workSheetName + "' contains non-increasing dates"
                     
        workbookNumericData[workSheetName] = data
        workbookCodes[workSheetName] = codes
        workbookDates[workSheetName] = dates   
    
    save_obj(workbookNumericData, 'ASX_ForecastNN_Numeric')
    save_obj(workbookCodes, 'ASX_ForecastNN_Unicodes')
    save_obj(workbookDates, 'ASX_ForecastNN_Dates')
    save_obj(workbookSheetNames, 'ASX_ForecastNN_SheetNames')
    
if __name__ == "__main__":
    
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
    
    readExcelSpreadsheet_SaveInputs(fileName,workbookSheetNames,firstRowOfCodesPerSheet)

    ASX_ForecastNN_Numeric = load_obj('ASX_ForecastNN_Numeric')
    ASX_ForecastNN_Unicodes = load_obj('ASX_ForecastNN_Unicodes')
    ASX_ForecastNN_Dates = load_obj('ASX_ForecastNN_Dates')
    ASX_ForecastNN_SheetNames = load_obj('ASX_ForecastNN_SheetNames')
    
    