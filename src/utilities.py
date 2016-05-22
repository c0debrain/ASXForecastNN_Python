import math

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