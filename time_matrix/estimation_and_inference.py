import numpy as np
import pandas as pd


def makeTreatmentDataFrame(Y, LHat, D): 
    
    ### Creates a data frame that tracks treatment effect for each unit, relative to when 
    ### the unit adopted the intervention.
    ### Only units that adopt the intervention are in the returned data frame, 
    ### and only the points after adoption are returned. 
    
    ###  Y should be a DataFrame with unit identifiers as index, time domain across columns
    
    differenceMatrix = Y-LHat ### Matrix of differences, reformatting into long format
   
    differenceMatrix.reset_index(inplace=True) 

    differenceMatrix = pd.melt(differenceMatrix, id_vars=differenceMatrix.columns[0], value_name='estimated_effect')
    
    T = np.matmul(D, np.triu(np.ones(np.repeat(D.shape[1], 2)))) 
    ### Keeps track of how long each unit has adopted the intervention for at each time point, also converted to long format
    
    T = pd.DataFrame(T, index=Y.index, columns=Y.columns)
    
    T.reset_index(inplace=True) 

    T = pd.melt(T, id_vars=T.columns[0], value_name='time_since_intervention_adoption')
    
    joinedData = differenceMatrix.merge(T) ### Join treatment effect data with treatment duration data
    
    joinedData = joinedData.loc[joinedData['time_since_intervention_adoption'] > 0, :]
    
    return(joinedData.reset_index(drop=True))



def bootstrappedDataGenerator(Y, fullD, treatedD, bootstrapIterations):
    #### Generates bootstrapped treatment effect data sets. 
    #### These data sets can be aggregated together in different ways to generate confidence intervals
    ### for various treatment effects. 
    
    allYs = {}
    
    allFullDs = {}
    
    allTreatedDs = {}
    
    ### All array arguments are pandas DataFrames so that the treatment effect DataFrame can be constructed
    
    allDurations = pd.Series(np.apply_along_axis(lambda x: x.sum(), 1, treatedD))

    treatmentDistribution = allDurations.value_counts().sort_index()
    
  #  if weightMatrix is None: ### If weight matrix not supplied, sets matrix to a matrix of 1's
        
  #      weightMatrix = pd.DataFrame(np.ones(treatedD.shape), index=Y.index, columns=Y.columns)

    for b in range(bootstrapIterations): ### Generates bootstrapIterations number of data sets

        for d in treatmentDistribution.index: ### Stratified sampling of units based on duration of intervention

            numberToSamp = treatmentDistribution[d]

            unitsThisDuration = np.where(allDurations ==d)[0]

            theSample = np.random.choice( unitsThisDuration, numberToSamp, replace=True)
            

            ### Sampling from all the DataFrames
            theSampledY = Y.iloc[theSample,:]

            theSampledFullD = fullD.iloc[theSample,:]

            theSampledTreatedD = treatedD.iloc[theSample,:]

       #     theSampledWeightMatrix = weightMatrix.iloc[theSample,:]

            if d==0:

                bootstrappedY = theSampledY

                bootstrappedFullD = theSampledFullD

                bootstrappedTreatedD = theSampledTreatedD

       #         bootstrappedWeightMatrix = theSampledWeightMatrix

            else: 

                bootstrappedY = pd.concat((bootstrappedY, theSampledY), axis=0)

                bootstrappedFullD = pd.concat((bootstrappedFullD, theSampledFullD), axis=0)

                bootstrappedTreatedD = pd.concat((bootstrappedTreatedD, theSampledTreatedD), axis=0)

       #         bootstrappedWeightMatrix = pd.concat((bootstrappedTreatedD, theSampledWeightMatrix), axis=0)
                
        allYs[b+1] = bootstrappedY
        
        allFullDs[b+1] = bootstrappedFullD

        allTreatedDs[b+1] = bootstrappedTreatedD
    
    return allYs, allFullDs, allTreatedDs