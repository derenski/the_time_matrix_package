import pandas as pd

import numpy as np

from numpy.linalg import svd, matrix_rank, lstsq, inv

from scipy.stats import bernoulli

from multiprocessing import Pool

from functools import partial

def P_Omega(Y, D): ## Projects data onto set of observed entries,
                   ## Where D_ij=0 means observed, and D_ij=1 mean missing
    
    return(Y*(1-D))


def propensityScoreToWeight(propScoreMat):
    ### Converts a matrix of propensity scores to the appropriate weight matrix
    
    ###  A propensity score of 1 means the cell
    ### of the corresponding response matrix is guaranteed to be missing.
    ### To prevent numerical issues, these values are reset to 0,
    ### because they will be eliminated when the weight matrix is 
    ### multiplied by one minus the missingness matrix. 
    propScoreMat[propScoreMat==1] = 0
    
    return(1/(1-propScoreMat))
    


def r1CompDescentWeighted(Y, D, propScoreMat, lambda_penalty=None, r_init=40, tolerance=1e-04, max_iterations=1000):
    
    weightMatrix = propensityScoreToWeight(propScoreMat)
    
    weightTimesMissingness = (1-D)*weightMatrix
    
    def shrink_operator(x, lambda_penalty): ## A helper function, the shrinkage operator
    
        if (x > 1*lambda_penalty):
      
            return(x-lambda_penalty)
      
        elif abs(x)< lambda_penalty:
      
            return(0)
      
        else:
      
            return(x+lambda_penalty)
  
    N = Y.shape[0]
  
    Time = Y.shape[1]
  
    Us = np.random.multivariate_normal(np.zeros(r_init), np.diag(np.ones(r_init)), size=N)
    
    Us = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 0 ,Us)
  
    sigma_vec = np.random.normal(loc=0.0, scale=1.0, size=r_init)
  
    Vs = np.random.multivariate_normal(np.zeros(r_init), np.diag(np.ones(r_init)), size=Time)
  
    Vs = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 0 ,Vs)
  
    r = r_init
  
    L_k = P_Omega(Y, D)
  
    iterating = True
  
    iter_number = 1
   
    while iterating:
    
        L_k_r = L_k
    
        if iter_number==1:
      
          parts_to_update = range(len(sigma_vec))
      
        else:
      
          parts_to_update = np.where(sigma_vec > 0)[0]
      
    
        for r_number in parts_to_update: ## Note: weights don't influence update of u,v (because we normalize)
      
      ## Two algorithms should be identical when matrix is just 1's (unless initialized differently)
            firstTermUUpdate = np.apply_along_axis(lambda x: 1/np.sum(x), 1, weightTimesMissingness)
            
            Us[:, r_number] = firstTermUUpdate*np.matmul((weightTimesMissingness*L_k_r), Vs[:, r_number])
      
            Us[:, r_number] = Us[:, r_number]/np.linalg.norm(Us[:, r_number])
        
            firstTermVUpdate = np.apply_along_axis(lambda x: 1/np.sum(x), 0, weightTimesMissingness)
      
            Vs[:, r_number] = firstTermVUpdate*np.matmul((weightTimesMissingness*L_k_r).transpose(), Us[:, r_number])
      
            Vs[:, r_number] = Vs[:, r_number]/np.linalg.norm(Vs[:, r_number])
      
            uvOuterProduct = np.matmul(np.atleast_2d(Us[:, r_number]).transpose(), 
                                   np.atleast_2d(Vs[:, r_number]))

            special_number = np.sum(weightTimesMissingness*(uvOuterProduct**2))
      
            sigma_vec[r_number] = max(0, shrink_operator(x=np.sum(uvOuterProduct * L_k_r*weightTimesMissingness)\
                                                         /special_number, lambda_penalty=lambda_penalty))

            L_k_r = L_k_r - sigma_vec[r_number] * uvOuterProduct 
            
        L_k_plus_1 = L_k.copy()
    
        Z = L_k-L_k_r
    
        L_k_plus_1[D!=0] = Z[D!=0]

       # convergenceMissingEntries = (np.linalg.norm((L_k_plus_1-L_k)*D, 'fro')\
       #               /max(1, np.linalg.norm(1-D, 'fro')))
        
        
        convergenceMissingEntries = (np.linalg.norm((L_k_plus_1-L_k)*D, 'fro')\
                      /max(1, np.linalg.norm(1-D, 'fro')))
        
        condition1 = (convergenceMissingEntries  < tolerance)
    
        condition2 = (iter_number >= max_iterations)
        
        if condition1 | condition2:
      
            L_k_final = Z
      
            iterating = False
      
            break
      
        else:
      
            L_k = L_k_plus_1
      
            iter_number = iter_number+1
      
            continue
                    
    ## Is updating changing much? 

    return(Z)


def factorModelCompletionWeighted(Y, D, propScoreMat, numFactors=3):
    
    ### Helper function for calculating and storing
    ### the outer products of the rows of a matrix with themselves
    def outerProductOfRows(aMatrix):
    
        listOfMatrices = []
    
        for row in range(aMatrix.shape[0]):
    
            rowAsArray = np.atleast_2d(aMatrix[row ,:])
    
            arrayToAppend = np.matmul(rowAsArray.transpose(), rowAsArray)
    
            listOfMatrices.append(arrayToAppend)
    
        tensorOfOuterProducts = np.stack(listOfMatrices, axis=2)
    
        return(tensorOfOuterProducts)
    
    ### Calculate the weighted sum of several matrices
    def weightedMatrixComp(numpy3DArray, matrixWeights):
        
        initialArray = np.zeros(numpy3DArray.shape[0:2])
        
        for matrixSlice in range(numpy3DArray.shape[2]):
            
            ### If weight is very small, skip
            if round(matrixWeights[matrixSlice], 4)==0:
                
                next
            
            initialArray = initialArray+matrixWeights[matrixSlice]*numpy3DArray[:,:, matrixSlice]
            
        return(initialArray)    
        
    ### Set unobserved entries to 0.    
    Y_proj = P_Omega(Y, D)
    
    ### Calculate numerator for covariance matrix of Y_proj
    varianceCovariance = np.matmul(Y_proj, Y_proj.transpose())
    
    ### Adjusts demoninator because we only use times where 
    ### both units are observed
    times_where_both_observed = np.matmul(1-D, (1-D).transpose()) 
  
    altered_cov_est = varianceCovariance/times_where_both_observed
  
    ### Preparing to estimate the loadings
    cov_est_for_PCA = (1/D.shape[0])*altered_cov_est
  
    ### The loadings are stored in the variable 'u'
    _, u = eig(cov_est_for_PCA)
    
    if numFactors is None: ### If K is not specified, default to full rank SVD
        
        numFactors = sDiag.shape[0]
    
    ### Taking the correct number of factors
    lambda_i = np.sqrt(D.shape[0])*u[:, 0:numFactors]
    
    ### Prevents divisibility warning
    propScoreMat[propScoreMat==1] = 0
    
    ### Needed propensity score weights
    missingnessDivPropScore = (1-D)/(1-propScoreMat)
    
    ### Calculating te outer products of the loading vectors with themselves,
    ### used for estimating factors via weighted regression.
    tensorOfOuterProducts = outerProductOfRows(lambda_i)
    
    ### The second term needed for estimating the factors via regression
    secondPartToRegression = np.matmul((missingnessDivPropScore*Y).transpose(), 
                                       lambda_i)
    
    ### Stores estimated factors in a list
    listOfFts = []
    
    for timePoint in range(Y.shape[1]):
        ### Calculate F_t, for each t
        
        matrixWeights = missingnessDivPropScore[:, timePoint]
        
        ### Computes design matrix for regression
        designMatrix = weightedMatrixComp(tensorOfOuterProducts, matrixWeights)
        
        firstRegressionPart = inv(designMatrix)
        
        F_t = np.matmul(firstRegressionPart, secondPartToRegression[timePoint,:])
        
        listOfFts.append(F_t)
        
    F_tArray = np.stack(listOfFts, axis=1)
  
    ### Calculates complete, imputed matrix.
    L_hat = np.matmul(lambda_i, F_tArray)
    
    return(L_hat)


###################### Validation Methods

def validationGivenTuningWeighted(Y, D, propScoreMat, tuningValue, method, numFolds=5, max_iterations=1000, tolerance=1e-02,
                         r_init=None):
    ### Does K-fold cross-validation, for a given penalty term
    
        if isinstance(method(1), R1Comp):
        
            if r_init is None: 
        
                raise ValueError('Must Specify the Rank Parameter!')
        
            elif np.round(np.floor(r_init)-r_init, 4) != 0:
        
                raise TypeError('Rank is not an Integer!')
        
            elif r_init <= 0: 
        
                raise ValueError('Rank must be positive!')
        
            builtMethod = method(tuningValue, max_iterations=max_iterations, 
                                         r_init=r_init)  
    
        else: 
            
            builtMethod = method(tuningValue, max_iterations=max_iterations, 
                                         tolerance=tolerance)
        
        errorList = []
        
        np.random.seed() ### Guarantees each new process gets a differnt seed

        for thisFold in range(numFolds):
            
            ### Cross validation, by creating holdout set
            probChosen = np.sum(D)/np.product(D.shape)

            ### Hold out same proportion missing in full matrix
            DResample = bernoulli.rvs(probChosen, size=np.product(D.shape))

            DResample = DResample.reshape(D.shape)
            
            DThisFold = D + (1-D)*DResample ### Index matrix for this fold
            
            estimateThisFold = builtMethod.fit(Y, DThisFold, propScoreMat)
            
            ### Validation error
            errorThisFold = np.sum(((estimateThisFold-Y)**2)*(1-D)*DResample)\
            /np.maximum(1, np.sum((1-D)*DResample))
            
            errorList.append(errorThisFold)

        return(np.mean(errorList))    
    
    
def validationWeighted(Y, D, propScoreMat, method, tuningGrid=None, numFolds=3, max_iterations=1000, tolerance=1e-02,
              r_init=None):

    ### The R1Comp method has one different parameter, so it must be initialized separately
    if isinstance(method(1), R1Comp):
        
        if r_init is None: 
        
            raise ValueError('Must Specify the Rank Parameter!')
        
        elif np.round(np.floor(r_init)-r_init, 4) != 0:
        
            raise TypeError('Rank is not an Integer!')
        
        elif r_init <= 0: 
        
            raise ValueError('Rank must be positive!')
        
        partialFunction= partial(validationGivenTuningWeighted, Y, D, propScoreMat, method=method, numFolds=numFolds,
                                 max_iterations=max_iterations, tolerance=tolerance, r_init=r_init)    
        
    else:
        
        partialFunction= partial(validationGivenTuningWeighted, Y, D, propScoreMat, 
                                 method=method, numFolds=numFolds, 
                                 max_iterations=max_iterations, tolerance=tolerance)
            
    with Pool(4) as p: 

        errorMatrix = p.map(partialFunction, tuningGrid)

    return(pd.Series(dict(zip(tuningGrid, errorMatrix))))
