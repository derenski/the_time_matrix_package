import pandas as pd

from scipy.stats import boxcox

import scipy.special as sc


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
    
    
    def transform(self, lmbda = None):
        
        if (self.__transform != int(1)):

            raise ValueError('Series is already transformed!')
        
        if lmbda is not None:

            if int(lmbda) == int(1):

                return None
            
            boxcoxTransformInfo = sc.boxcox(self.__series, lmbda)
            
            self.__transform = lmbda
            
        else:
            
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

    def toMatrix(self, L): #### Transforms a time series to matrix format. 
    
        if (self.__series.shape[0] % L !=0): 

            raise ValueError("Cannot create complete matrix with L rows.")

        seriesAsMatrix = np.array(self.__series).reshape((L, int(self.__series.shape[0]/L)), order ='F')

        return(seriesAsMatrix)
        
        
        
#def expandToImpute(pandasTimeSeries, startTime=None, endTime=None, frequency= None):
    
        ### Utility for expanding an irregularly spaced series to prepare it for imputation. 
        ### Requires first and last time points to be specified, in decimal format. 
        ### The frequency of the time series must also be specified
    
#    if startTime is None:
        
#        raise ValueError('Must Specify a Start Time.')
        
#    if endTime is None:
        
#        raise ValueError('Must Specify an end Time.')
    
#    if frequency is None: 
        
 #       raise ValueError('Must Specify a Frequency.')
        
#    fullIndex = np.linspace(startTime, endTime, int((endTime-startTime+1/frequency)*frequency))
#        
 #   indexAsSeries = pd.Series(fullIndex)
    
 #   indexAsSeries = indexAsSeries.to_frame('real_index')
    
 #   indexAsSeries = indexAsSeries.set_index('real_index', drop=True)
        
 #   fullIndexJoinedWithObservedData = indexAsSeries.join(pandasTimeSeries.to_frame('time_series'), how='left')
    
 #   fullIndexJoinedWithObservedData.index.name = None
        
#    return fullIndexJoinedWithObservedData['time_series']



class panelDataSet:
    
    def __init__(self, panelSet, idVariable, timeVariable, responseVariable):
        ### Creates an object of class 'panel data set'
        
        ### Arguments: 
        ### panelset: the panel data IN LONG FORMAT
        ## idVariable: variable that identifies the units
        ## timeVariable: the variable defining the time domain
        ## responseVariable: the variable serving as the response
        
        self.__data = panelSet[[idVariable, timeVariable, responseVariable]]
        
        self.__idVariable = idVariable
        
        self.__timeVariable = timeVariable
        
        self.__responseVariable = responseVariable
    
    def getData(self): ## Returns the data IN LONG FORMAT
        
        return(self.__data)
    
    def getidVariable(self): ### Returns the name of the variable used as the unit identifier
        
        return(self.__idVariable)
    
    def gettimeVariable(self): ### Returns the name of the variable used for the time domain
        
        return(self.__timeVariable)
    
    def getresponseVariable(self): ## Returns the name of the variable used for the response
        
        return(self.__responseVariable)
        
    def getWideFormat(self): ### Produces table in wide format, including NA's 
        
        return(self.__data.pivot_table(index=self.__idVariable, 
                                                columns=self.__timeVariable, 
                                                values=self.__responseVariable))
    
    def parseUnits(self, dropNA = True): ## Parses the units in the data set into a list of pandas series objects
                                         ## Set dropNA=True to recover the unit on all the time points it is observed 
            
                                         ## This is good for exploring a panel data set, and analyzing invividual time series within that set. 
        
        dataMadeWideAndTransposed = self.getWideFormat().transpose()
        
        unitsParsed =  dict(dataMadeWideAndTransposed)
        
        if dropNA:
            
            unitsParsed = {i: unitsParsed[i].dropna() for i in unitsParsed.keys()}
            
        return(unitsParsed)
        
    def getMissingnessMatrix(self): ### Returns a missingness indicator matrix
        
        dataMadeWide = self.getWideFormat()
        
        dataMadeWide = dataMadeWide.apply(lambda x: pd.isnull(x))*1
        
        return(dataMadeWide)