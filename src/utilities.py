import math
import numpy as np
import datetime

def appendData(newDataEntry,newDateEntry, newLabel, data,dates,labels):
    
    if newLabel not in labels:
        data.append(newDataEntry)
        dates.append(newDateEntry)
        labels.append(newLabel)
    else:
        #replace existing
        idx = labels.index(newLabel)
        data.pop(idx), data.insert(idx,newDataEntry)
        dates.pop(idx), dates.insert(idx,newDateEntry)
        labels.pop(idx), labels.insert(idx,newLabel)
        
    return data, dates, labels
    
def convertTargetListToNumpyArray(targetList):
    #check first element, if list the scale array dimensions appropriately
    if(isinstance(targetList[0],list)):
        outputNum = len(targetList[0])
    else:
        outputNum = 1
        
    npArrayTarget = np.zeros(shape=(len(targetList),outputNum))
    for idx in range(0,len(targetList)):
        npArrayTarget[idx] = np.array(targetList[idx])
      
    return npArrayTarget

def classifyReturnDirection(signReturn,option):
    positiveReturn = [1.0, -1.0]
    negativeReturn = [-1.0, 1.0]
    
    if(option == 'DUAL_VEC'):
        if(signReturn > 0):
            output = positiveReturn
        else:
            output = negativeReturn
    elif(option == 'SINGLE_VEC'):
        output = signReturn
        
    return output
    
def transformPrices(firstPrice,returns,returnsDates,option):
    prices = []
    prices.append(firstPrice)
    for idx in range(1,len(returns)+1):
        if(option == 'LOG_DIFF'):
                prices.append(prices[idx-1]*math.exp(returns[idx-1]))
        elif(option == 'REL_DIFF'):
                prices.append(prices[idx-1] + returns[idx-1]*prices[idx-1])
                              
    firstPriceDate = returnsDates[0]+ datetime.timedelta(days=(-1))        
    if(firstPriceDate.isoweekday() > 5):
        firstPriceDate = firstPriceDate + datetime.timedelta(days=-2) 
    
    returnsDates.insert(0, firstPriceDate)   
    pricesDates  = returnsDates
    
    return prices, pricesDates

def transformIntraDayPrices(openPrices,returns,returnsDates,option):
    closePrices = []
    for idx in range(len(returns)):
        if(option == 'LOG_DIFF'):
                closePrices.append(openPrices[idx]*math.exp(returns[idx]))
        elif(option == 'REL_DIFF'):
                closePrices.append(openPrices[idx] + returns[idx]*openPrices[idx])
                           
    return closePrices, returnsDates

def getReturns(prices,dates,option):
    returns = []
    for idx in range(1,len(prices)):
        if(option == 'LOG_DIFF'):
            if((prices[idx] != 0) & (prices[idx-1] != 0)):
                returns.append(math.log(prices[idx])-math.log(prices[idx-1]))
            else:
                returns.append(0.0)
        elif(option == 'REL_DIFF'):
            if(prices[idx-1] != 0):
                returns.append((prices[idx]-prices[idx-1])/prices[idx-1])
            else:
                returns.append(0.0)
                
    return returns,[dates[i] for i in range(1,len(prices))]

def getIntraDayReturns(openPrices,closePrices,dates,option):
    returns = []
    #remove the first return to align with dates of interest
    #due to Rt = Pt_later - Pt_early
    assert(len(openPrices) == len(closePrices)), "Size of open prices != size of close prices"
    
    for idx in range(1,len(openPrices)):
        if(option == 'LOG_DIFF'):
            if((openPrices[idx] != 0) & (closePrices[idx] != 0)):
                returns.append(math.log(closePrices[idx])-math.log(openPrices[idx]))
            else:
                returns.append(0.0)
        elif(option == 'REL_DIFF'):
            if(openPrices[idx] != 0):
                returns.append((closePrices[idx]-openPrices[idx])/openPrices[idx])
            else:
                returns.append(0.0)
                
    return returns,[dates[i] for i in range(1,len(openPrices))]