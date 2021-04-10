import pandas as pd

import numpy as np

class PanelDataSet:
    
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
    
    def parseUnits(self, dropNA = True): 
        
        ## Parses the units in the data set into a list of pandas series objects
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
    
    
    
    
#def sieveOfEratosthenes(N):
    
 #   allNums = np.linspace(2, N, N-1)
    
  #  numsToMult = allNums[np.where(allNums < np.sqrt(N))[0]]
    
   # for theNum in numsToMult: 
        
    #    if ~np.isin(theNum, allNums):

     #       next
        
      #  howFarToGo = np.floor(N/theNum).astype('int')
        
       # numsToRemove = theNum*np.linspace(2, howFarToGo, howFarToGo-1)
        
        #indexForRemovalAllNums = np.argwhere(np.isin(allNums, numsToRemove))
        
        #allNums = np.delete(allNums, indexForRemovalAllNums)
    
   # return(allNums)