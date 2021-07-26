import numpy as np
import pandas as pd


def makeTreatmentDataFrame(Y, LHat, DTreated): 
    
    ### Creates a data frame that tracks treatment effect for each unit, relative to when 
    ### the unit adopted the intervention.
    ### Only units that adopt the intervention are in the returned data frame, 
    ### and only the points after adoption are returned. 
    
    ###  Y should be a DataFrame with unit identifiers as index, time domain across columns
    
    differenceMatrix = Y-LHat ### Matrix of differences, reformatting into long format
   
    differenceMatrix.reset_index(inplace=True) 

    differenceMatrix = pd.melt(differenceMatrix, id_vars=differenceMatrix.columns[0], value_name='estimated_effect')
    
    Tmatrix = np.matmul(DTreated, np.triu(np.ones(np.repeat(DTreated.shape[1], 2)))) 
    ### Keeps track of how long each unit has adopted the intervention for at each time point, also converted to long format
    
    Tmatrix = pd.DataFrame(Tmatrix, index=Y.index, columns=Y.columns)
    
    Tmatrix.reset_index(inplace=True) 

    Tmatrix = pd.melt(Tmatrix, id_vars=Tmatrix.columns[0], value_name='time_since_intervention_adoption')
    
    joinedData = differenceMatrix.merge(Tmatrix) ### Join treatment effect data with treatment duration data
    
    joinedData = joinedData.loc[joinedData['time_since_intervention_adoption'] > 0, :]
    
    return(joinedData.reset_index(drop=True))



def bootstrappedDataGenerator(Y, DMissingness, DTreated, bootstrapIterations):
    #### Generates bootstrapped treatment effect data sets. 
    #### These data sets can be aggregated together in different ways to generate confidence intervals
    ### for various treatment effects. 
    
    ### As there are two types of missingness present in panel data 
    ### (actual missing observations, and observations that are missing 
    ### because they correspond to treated cells), we differentiate
    ### the missingness types with two arguments
    
    ### DMissingness is the full missingness matrix that will be used 
    ### for matrix completion. 
    
    ### DTreated is a matrix representing ONLY the treatment adoption design,
    ### Meaning the only 1's in this matrix correspond to cells that are treated,
    ### and not to cells missing for another reason. 

    
    allYs = {}
    
    allDMissingnesses = {}
    
    allDTreateds = {}
    
    ### All array arguments are pandas DataFrames so that the treatment effect DataFrame can be constructed
    
    allDurations = pd.Series(np.apply_along_axis(lambda x: x.sum(), 1, DTreated))

    treatmentDistribution = allDurations.value_counts().sort_index()
    
  #  if weightMatrix is None: ### If weight matrix not supplied, sets matrix to a matrix of 1's
        
  #      weightMatrix = pd.DataFrame(np.ones(DTreated.shape), index=Y.index, columns=Y.columns)

    for b in range(bootstrapIterations): ### Generates bootstrapIterations number of data sets

        for d in treatmentDistribution.index: ### Stratified sampling of units based on duration of intervention

            numberToSamp = treatmentDistribution[d]

            unitsThisDuration = np.where(allDurations ==d)[0]

            theSample = np.random.choice( unitsThisDuration, numberToSamp, replace=True)
            

            ### Sampling from all the DataFrames
            theSampledY = Y.iloc[theSample,:]

            theSampledDMissingness = DMissingness.iloc[theSample,:]

            theSampledDTreated = DTreated.iloc[theSample,:]

       #     theSampledWeightMatrix = weightMatrix.iloc[theSample,:]

            if d==0:

                bootstrappedY = theSampledY

                bootstrappedDMissingness = theSampledDMissingness

                bootstrappedDTreated = theSampledDTreated

       #         bootstrappedWeightMatrix = theSampledWeightMatrix

            else: 

                bootstrappedY = pd.concat((bootstrappedY, theSampledY), axis=0)

                bootstrappedDMissingness = pd.concat((bootstrappedDMissingness, theSampledDMissingness), axis=0)

                bootstrappedDTreated = pd.concat((bootstrappedDTreated, theSampledDTreated), axis=0)

       #         bootstrappedWeightMatrix = pd.concat((bootstrappedDTreated, theSampledWeightMatrix), axis=0)
                
        allYs[b+1] = bootstrappedY
        
        allDMissingnesses[b+1] = bootstrappedDMissingness

        allDTreateds[b+1] = bootstrappedDTreated
    
    return allYs, allDMissingnesses, allDTreateds