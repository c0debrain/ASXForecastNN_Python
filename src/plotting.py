import math
import matplotlib.pyplot as plt
import numpy as np

def norm(x,binCount,n,SD,mean):
    rangeX = x.max() - x.min()
    binwidth = rangeX/binCount
    area = n*binwidth
    lowLim = mean - 3*SD
    upLim = mean + 3*SD
    x = np.arange(lowLim,upLim,(upLim - lowLim)/100.0)
    return x,area*np.exp(-(x-mean)*(x-mean)/(2*SD*SD))/(SD*math.sqrt(2*math.pi))

def correlationHeatMap(data,labels,title):
    
    array = np.array(data)
    corrMat = np.abs(np.corrcoef(array))

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(corrMat,cmap='RdBu')
    plt.colorbar(heatmap)
    plt.title(title)
    
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(array.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(array.shape[0])+0.5, minor=False)
    
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    ax.set_xticklabels(labels, minor=False,rotation = 'vertical')
    ax.set_yticklabels(labels, minor=False)
    plt.show()

def plotHist(data,binCount,plotLabels,title):
    
    numInputs = len(data)
    numSubPlotsPerFig = 4    
    div = int(math.floor(numInputs/numSubPlotsPerFig))
    rem = numInputs%numSubPlotsPerFig
    
    if(div != 0):
        if(rem == 0):
            numFigs = div
        else :
            numFigs = div + 1
    else:
        numFigs = 1
    
    for i in range(1,numFigs+1):
        plt.figure(i)
        plt.title(title+' - Figure ' + str(i))
        for j in range(1,numSubPlotsPerFig+1):
            dataIdx = numSubPlotsPerFig*(i-1) +j-1
            if(dataIdx < numInputs):
                x = np.array(data[dataIdx])
                subplotFig = 200+20+j
                plt.subplot(subplotFig)
                plt.hist(x,binCount)
                z,pdf = norm(x,binCount,x.shape[0],np.std(x),np.mean(x))
                plt.plot(z,pdf,'r')
                plt.xlabel(plotLabels[dataIdx])
                plt.ylabel('Instances')
                plt.grid(True)       

    plt.show()
    
def subPlotData(datesData,pricesData,plotLabels,title):
    
    numSubPlotsPerFig = 4    
    assert(len(datesData) == len(pricesData)),"Number of dates not equal to number of prices"   
    div = int(math.floor(len(datesData)/numSubPlotsPerFig))
    rem = len(datesData)%numSubPlotsPerFig
    
    if(div != 0):
        if(rem == 0):
            numFigs = div
        else :
            numFigs = div + 1
    else:
        numFigs = 1
    
    for i in range(1,numFigs+1):
        plt.figure(i)
        plt.title(title+' - Figure ' + str(i))
        for j in range(1,numSubPlotsPerFig+1):
            dataIdx = numSubPlotsPerFig*(i-1) +j-1
            if(dataIdx < len(datesData)):
                subplotFig = 200+20+j
                plt.subplot(subplotFig)
                plt.plot(datesData[dataIdx], pricesData[dataIdx], 'b')
                plt.xlabel('Time')
                plt.ylabel(plotLabels[dataIdx])
                plt.grid(True)       

    plt.show()