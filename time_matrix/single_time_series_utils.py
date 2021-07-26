import pandas as pd

from scipy.stats import boxcox

import scipy.special as sc

import numpy as np






#### Utilities for transforming a time series of given length
### into an approximately square matrix


def allPairs(theLength): ### Note: this is an O(sqrt(N)) algorithm for seeing if a number is prime!
    
    allPairings = []
    
    rootLength = np.floor(np.sqrt(theLength)).astype('int')
    
    allNums = np.linspace(2, rootLength, rootLength-1)
    
    for aNum in allNums:
        
        thePartner = theLength/aNum
        
        isItAWholeNumber = np.round(thePartner-np.floor(thePartner), 7) == 0
        
        if isItAWholeNumber:
            
            allPairings.append([aNum, thePartner])
            
            
    return(np.array(allPairings))
    
def SquareLoss(sideLengths): ### Measures how non-square a matrix is. 
    
    return(np.diff(sideLengths)**2)    


def squareify(aLength):
    ### Produces the dimensions that will make a 1D array of length N
    ### a 2D array of length N as square as possible 
    potentialPairings = allPairs(aLength)
    
    if potentialPairings.size==0:
        
        raise ValueError('Length of Series is Prime!')
    
    allPairingLosses = pd.Series(np.apply_along_axis(SquareLoss, 1, potentialPairings).flatten())

    bestPairingIndex = allPairingLosses.idxmin()

    bestDimensions = potentialPairings[bestPairingIndex,: ]
    
    return(bestDimensions)



################################
#################################




class timeSeries: ### A class for exploring and analyzing individual time series
    
    def __init__(self, timeSeriesData, frequency=None):
        
        if not isinstance(timeSeriesData, pd.Series):
            
            raise TypeError('Data must be of type Series.')
            
        if frequency is None: 
            
            raise ValueError('Must set a frequency.')
            
        self.__series = timeSeriesData
        
        self.__frequency = frequency
        
        self.__is_transformed = False
        
        self.__transform = int(1)
        
        self.__matrixForm = None
        
    def getSeries(self):
        
        return(self.__series)
    
    def getFrequency(self):
        
        return(self.__frequency)
    
    def expandToImpute(self):
        
        ### Optional utility for expanding an irregularly spaced series to prepare it for imputation. 
        ### Requires first and last values of the series be observed. 
        
        startTime = np.min(self.__series.index)
        
        endTime = np.max(self.__series.index)
        
        fullIndex = np.linspace(startTime, endTime, int((endTime-startTime+1/self.__frequency)*self.__frequency))
        
        indexAsSeries = pd.Series(fullIndex)
        
        
        return indexAsSeries, self.__series
    
    def forecastingPrep(self, h):
        
        nextTimePoints = np.max(self.getSeries().index)\
        +(1/self.getFrequency())*np.linspace(1, h, h)
        
        nextValues = pd.Series(dict(zip(nextTimePoints, np.zeros(h))))
        
        return(self.getSeries().append(nextValues))
    
    def
    
    
    def transform(self, lmbda = None):
        #### Applies a BoxCox transformation to your series.
        
        ### Transform parameter can either be specified, 
        ### or set to None and calculated via MLE. 
        
        if (self.__transform != int(1)):

            raise ValueError('Series is already transformed!')
        
        if lmbda is not None:

            if int(lmbda) == int(1):

                return None
            
            boxcoxTransformInfo = sc.boxcox(self.__series, lmbda)
            
            self.__transform = lmbda
            
        else:
            
            ### If the time series has missing values, they are removed 
            ### when the power transform is calculated.
            
            ### Removes missing values before power transformation
        
            _, estimLmbda = boxcox(self.__series.dropna())
            
            boxcoxTransformInfo = sc.boxcox(self.__series, estimLmbda)
            
            self.__transform = estimLmbda
        
        self.__series = pd.Series(boxcoxTransformInfo, index=self.__series.index)

  
    def getTransform(self): ### gives value of transformation
        
        return(self.__transform)
        
    def invTransform(self): ### Back-transforms a Box-Cox transformed series
        
        if (self.__transform == int(1)):
            
            return None
        
        backtransformedSeries = sc.inv_boxcox(self.__series, self.getTransform())
        
        self.__series = pd.Series(backtransformedSeries, index=self.__series.index)
        
        self.__transform = int(1)
        
    def isTransformed(self):
        
        if (self.__transform == int(1)):
            
            return(False)
        
        else: 
            
            return(True)
        
        
def toMatrix(aTimeSeries, L, squareifySeries=False): #### Transforms a time series to matrix format. 
        
    if squareifySeries:
            
        L = squareify(aTimeSeries.shape[0])[0].astype('int')
            
    else:
            
        if (aTimeSeries.shape[0] % L !=0): 

            raise ValueError("Cannot create complete matrix with L rows.")

    seriesAsMatrix = np.array(aTimeSeries).reshape((L, int(aTimeSeries.shape[0]/L)), order ='F')

    return(seriesAsMatrix)
    
def matrixToSeries(timeSeriesInMatrixForm): ### This will prove useful in transforming a matrix back into a series
    ### And concattenating the new data onto the orignal series. 
        
    return(pd.Series(timeSeriesInMatrixForm.flatten('F')))
        
def forecastBackTransform(forecastSeries, theLambda): 
    
    backtransformedSeries = sc.inv_boxcox(forecastSeries, theLambda)
    
    return(backtransformedSeries)

def matrixForecast(aTimeSeriesObject, completionMethod, h, squareify):
    
    timeSeriesPreppedForForcast = aTimeSeriesObject.forecastingPrep(h)
    
    trainingSeries = np.max(aTimeSeriesObject.getSeries().index)
    
    theTimeSeriesAsAMatrix = matrixToSeries(timeSeriesInMatrixForm, squareify=squareify)
    