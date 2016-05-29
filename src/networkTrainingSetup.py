
import utilities as util
import math      
       
def createPredictorVariables(dataContainer,predictorInputs,returnsCalcOption):
    useReturn = 0
    usePrice = 1    
    
    predictorInputData = []
    predictorLabels = []

    for predictor in predictorInputs:
        if(predictor[1] == usePrice):
            dataIdx = dataContainer.pricesLabels.index(predictor[0])
            predictorLabels.append(dataContainer.pricesLabels[dataIdx])
            #get copy of prices data and
            #remove first price to match number of inputs from returns vectors
            copyPricesData = dataContainer.pricesData[dataIdx][:]
            copyPricesData.pop(0)
            predictorInputData.append(copyPricesData)
        elif(predictor[1] == useReturn):
            dataIdx = dataContainer.returnsLabels.index(predictor[0])
            predictorLabels.append(dataContainer.returnsLabels[dataIdx]+'_'+ returnsCalcOption+'_RETS')
            predictorInputData.append(dataContainer.returnsData[dataIdx])

    return predictorInputData, predictorLabels
    
def createIntraDayTarget(dataContainer,targetOpenPriceLabel,targetLastPriceLabel,targetLabel,targetChoice,classifyDirection,returnsCalcOption):
    
    targetReturns,targetDates = util.getIntraDayReturns(dataContainer.pricesData[dataContainer.pricesLabels.index(targetOpenPriceLabel)],dataContainer.pricesData[dataContainer.pricesLabels.index(targetLastPriceLabel)],dataContainer.pricesDates[dataContainer.pricesLabels.index(targetLastPriceLabel)],returnsCalcOption)
    if(targetChoice == 'DIRECTION'):
        target = [util.classifyReturnDirection(math.copysign(1,targetReturns[i]),classifyDirection) for i in range(len(targetReturns))]
    else:
        target = targetReturns
    
    dataContainer.returnsData, dataContainer.returnsDates, dataContainer.returnsLabels = util.appendData(targetReturns,
                                                   targetDates, 
                                                   targetLabel, 
                                                   dataContainer.returnsData,
                                                   dataContainer.returnsDates,
                                                   dataContainer.returnsLabels)                

    #remove the first price for intraday price targets 
    dataContainer.pricesData, dataContainer.pricesDates, dataContainer.pricesLabels = util.appendData(dataContainer.pricesData[dataContainer.pricesLabels.index(targetOpenPriceLabel)][1:],
                                                   dataContainer.pricesDates[dataContainer.pricesLabels.index(targetOpenPriceLabel)][1:], 
                                                   targetOpenPriceLabel, 
                                                   dataContainer.pricesData,
                                                   dataContainer.pricesDates,
                                                   dataContainer.pricesLabels)
        
    dataContainer.pricesData, dataContainer.pricesDates, dataContainer.pricesLabels = util.appendData(dataContainer.pricesData[dataContainer.pricesLabels.index(targetLastPriceLabel)][1:],
                                                   dataContainer.pricesDates[dataContainer.pricesLabels.index(targetLastPriceLabel)][1:], 
                                                   targetLastPriceLabel, 
                                                   dataContainer.pricesData,
                                                   dataContainer.pricesDates,
                                                   dataContainer.pricesLabels)
    
    targetOpenPrices = dataContainer.pricesData[dataContainer.pricesLabels.index(targetOpenPriceLabel)]
    targetPrices = dataContainer.pricesData[dataContainer.pricesLabels.index(targetLastPriceLabel)]
    targetPricesDates = dataContainer.pricesDates[dataContainer.pricesLabels.index(targetLastPriceLabel)]

    return target, targetDates, targetOpenPrices, targetPrices, targetPricesDates

def createTarget(dataContainer,targetLabel,targetChoice,classifyDirection,returnsCalcOption):    
    
    targetReturns = dataContainer.returnsData[dataContainer.returnsLabels.index(targetLabel)]
    if(targetChoice == 'DIRECTION'):
        target = [util.classifyReturnDirection(math.copysign(1,targetReturns[i]),classifyDirection) for i in range(len(targetReturns))]
    else:
        target = targetReturns
    targetDates = dataContainer.returnsDates[dataContainer.returnsLabels.index(targetLabel)]  
    
    targetPrices = dataContainer.pricesData[dataContainer.pricesLabels.index(targetLabel)] 
    targetPricesDates = dataContainer.pricesDates[dataContainer.pricesLabels.index(targetLabel)]
    
    return target, targetDates, targetPrices, targetPricesDates
         