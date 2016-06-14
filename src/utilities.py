import math
import numpy as np
import datetime
import plotting as customPlot

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
    
    pricesDates  = returnsDates[:]#create copy
    pricesDates.insert(0, firstPriceDate)   

    return prices, pricesDates

def transformIntraDayPrices(openPrices,returns,returnsDates,option):
    closePrices = []
    for idx in range(len(returns)):
        if(option == 'LOG_DIFF'):
                closePrices.append(openPrices[idx]*math.exp(returns[idx]))
        elif(option == 'REL_DIFF'):
                closePrices.append(openPrices[idx] + returns[idx]*openPrices[idx])
                           
    return closePrices, returnsDates

def transformIntraDayAdjPrices(openPrices,trueClosePrices,returns,returnsDates,option):
    estClosePrices = [openPrices[0]*math.exp(returns[0])]
    for idx in range(1,len(returns)):
        adjOpen = (estClosePrices[idx-1]- trueClosePrices[idx-1] + openPrices[idx])
        if(option == 'LOG_DIFF'):
                estClosePrices.append(adjOpen*math.exp(returns[idx]))
        elif(option == 'REL_DIFF'):
                estClosePrices.append(adjOpen + returns[idx]*adjOpen)
                           
    return estClosePrices, returnsDates

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

def applyNetworkReturnOriginalScale(inputTrainedScale,target_scaler,network):    
    estTarget = target_scaler.inverse_transform(network.predict(inputTrainedScale))
    return estTarget

def getIntraDayPNL(accumIndexLastPrices,estReturns,openPrices,lastPrices,dailyInterestRate,option):
         
    startPrice = lastPrices[0]     
    strategyPNL = [startPrice]
    buyHoldPNL = [startPrice]
    accumulation = [startPrice]
    accumulationUnits = startPrice / accumIndexLastPrices[0]
    
    for idx in range(1,len(estReturns)):
        
        buyHoldPNL.append(lastPrices[idx]-openPrices[idx] + buyHoldPNL[idx-1])
        accumulation.append(accumIndexLastPrices[idx] * accumulationUnits)
        
        if(estReturns[idx] > 0):
            #buy at open and sell at close 
            strategyPNL.append(lastPrices[idx]-openPrices[idx] + strategyPNL[idx-1])
        else:
            if(option == 'LONG_SHORT'):
                #short at open and buy at close
                strategyPNL.append(openPrices[idx]-lastPrices[idx] + strategyPNL[idx-1])
            elif(option == 'LONG'):
                if(dailyInterestRate is not None):
                    strategyPNL.append((1+dailyInterestRate)*strategyPNL[idx-1])      
                else:
                    strategyPNL.append(strategyPNL[idx-1])          
            else:
                    strategyPNL.append(strategyPNL[idx-1])  
                            
    return strategyPNL,buyHoldPNL,accumulation

def getIntraDayPNL2(estReturns,openPrices,lastPrices,dailyInterestRate,option):
    startPrice = lastPrices[0]     
    strategyPNL = [startPrice]
    buyHoldPNL = [startPrice]
    
    for idx in xrange(1, len(estReturns)):
        
        buyHoldPNL.append(lastPrices[idx])
        buySell = math.copysign(1,estReturns[idx])
        strategyPNL.append((lastPrices[idx]-openPrices[idx])*buySell + strategyPNL[idx-1])
    
    return strategyPNL,buyHoldPNL     
    
def predictorResultsSummary(binCount,histTitle,plotTitle,targetLabel,dates,y_target,y_predict):
    errs = y_target - y_predict
    customPlot.plotHist([errs.tolist()],binCount,[targetLabel +'_ERROR'],histTitle)
    customPlot.plotPerformance(dates,y_target,y_predict,'True','Est',targetLabel,plotTitle)                  
    
    print targetLabel + ' ' + plotTitle
    print 'Mean error = ',np.mean(errs)
    print 'RMSE = ', np.sqrt(np.mean(errs**2))
    print 'Corr = ', np.corrcoef(y_target,y_predict)[0,1]
    
    #use level estimated returns vectors to infer target direction 
    targetDirection = np.array([math.copysign(1,y_target[i]) for i in range(len(y_target))])
    estTargetDirection = np.array([math.copysign(1,y_predict[i]) for i in range(len(y_predict))])
    
    print targetLabel + ' ' + plotTitle +' - Classification Results'    
    print "From {} to {}".format(dates[0],dates[-1])                                                               
    print("Guessed {} out of {} = {}% correct".format(
        np.sum(targetDirection == estTargetDirection), targetDirection.size, 100*np.sum(targetDirection == estTargetDirection)/targetDirection.size
    ))     

def performTimeSeriesCV(X_train, y_train, number_folds, algorithm, parameters):
    """
    Given X_train and y_train (the test set is excluded from the Cross Validation),
    number of folds, the ML algorithm to implement and the parameters to test,
    the function acts based on the following logic: it splits X_train and y_train in a
    number of folds equal to number_folds. Then train on one fold and tests accuracy
    on the consecutive as follows:
    - Train on fold 1, test on 2
    - Train on fold 1-2, test on 3
    - Train on fold 1-2-3, test on 4
    ....
    Returns mean of test accuracies.
    """

    print 'Parameters --------------------------------> ', parameters
    print 'Size train set: ', X_train.shape
    
    # k is the size of each fold. It is computed dividing the number of 
    # rows in X_train by number_folds. This number is floored and coerced to int
    k = int(np.floor(float(X_train.shape[0]) / number_folds))
    print 'Size of each fold: ', k
    
    # initialize to zero the accuracies array. It is important to stress that
    # in the CV of Time Series if I have n folds I test n-1 folds as the first
    # one is always needed to train
    accuracies = np.zeros(number_folds-1)

    # loop from the first 2 folds to the total number of folds    
    for i in range(2, number_folds + 1):
        print ''
        
        # the split is the percentage at which to split the folds into train
        # and test. For example when i = 2 we are taking the first 2 folds out 
        # of the total available. In this specific case we have to split the
        # two of them in half (train on the first, test on the second), 
        # so split = 1/2 = 0.5 = 50%. When i = 3 we are taking the first 3 folds 
        # out of the total available, meaning that we have to split the three of them
        # in two at split = 2/3 = 0.66 = 66% (train on the first 2 and test on the
        # following)
        split = float(i-1)/i
        
        # example with i = 4 (first 4 folds):
        #      Splitting the first       4        chunks at          3      /        4
        print 'Splitting the first ' + str(i) + ' chunks at ' + str(i-1) + '/' + str(i) 
        
        # as we loop over the folds X and y are updated and increase in size.
        # This is the data that is going to be split and it increases in size 
        # in the loop as we account for more folds. If k = 300, with i starting from 2
        # the result is the following in the loop
        # i = 2
        # X = X_train[:(600)]
        # y = y_train[:(600)]
        #
        # i = 3
        # X = X_train[:(900)]
        # y = y_train[:(900)]
        # .... 
        X = X_train[:(k*i)]
        y = y_train[:(k*i)]
        print 'Size of train + test: ', X.shape # the size of the dataframe is going to be k*i

        # X and y contain both the folds to train and the fold to test.
        # index is the integer telling us where to split, according to the
        # split percentage we have set above
        index = int(np.floor(X.shape[0] * split))
        
        # folds used to train the model        
        X_trainFolds = X[:index]        
        y_trainFolds = y[:index]
        
        # fold used to test the model
        X_testFolds = X[(index + 1):]
        y_testFolds = y[(index + 1):]
        
        # i starts from 2 so the zeroth element in accuracies array is i-2. performClassification() is a function which takes care of a classification problem. This is only an example and you can replace this function with whatever ML approach you need.
        #accuracies[i-2] = performClassification(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds, algorithm, parameters)
        
        # example with i = 4:
        #      Accuracy on fold         4     :    0.85423
        print 'Accuracy on fold ' + str(i) + ': ', accuracies[i-2]
    
    # the function returns the mean of the accuracy on the n-1 folds    
    return accuracies.mean()