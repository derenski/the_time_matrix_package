import pandas as pd

import numpy as np

from numpy.linalg import svd, eig, matrix_rank, lstsq, inv

from scipy.stats import bernoulli

from multiprocessing import Pool

from functools import partial

### UNWEIGHTED Methods for matrix completion.
### Document the unweighted methods separately. 


def P_Omega(Y, D): ## Projects data onto set of observed entries,
                   ## Where D_ij=0 means observed, and D_ij=1 mean missing
    
    return(Y*(1-D))


## A utility for calculating the rank K SVD of a matrix, Y.
## Allows for optional shrinkage of the singular values.
def svdReconstruction(Y, K=None, shrink=False, lambda_penalty=None):
    ### Allows one to calculate the rank K svd of a matrix, Y. 
    ### Includes optional parameters for soft-thresholding. 
    
    u, sDiag, vh = svd(Y, full_matrices=True)
    
    if K is None: ### If K is not specified, default to full rank SVD
        
        K = sDiag.shape[0]
    
    if (shrink): ### Shrinkage with soft-thresholding
        
        if lambda_penalty is None: 
            
            raise ValueError("Specify a value (lambda_penalty) for the shrinkage penalty.")
            
        sDiag = np.maximum(0, sDiag-lambda_penalty)
            
    S = np.diag(sDiag)
        
    if (K > sDiag.shape[0]): ### Can't make rank larger than rank of Y
        
        raise ValueError("K is larger than rank of Y.")
    
    ### Rank K reconstruction
    
    secondPart = np.matmul(np.atleast_2d(S[0:K, 0:K]),
                           np.atleast_2d(vh[0:K, :]))
    
    SVDReconstruction = np.matmul(np.atleast_2d(u[:, 0:K]), secondPart)
    
    return(SVDReconstruction)


### A factor model estimation approach to matrix completion
def factorModelCompletion(Y, D, numFactors=3):
    
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
            
            ## Skips the entries that have 0 weight
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
    
    oneMinusD = 1-D
    
    ### Calculating te outer products of the loading vectors with themselves,
    ### used for estimating factors via weighted regression.
    tensorOfOuterProducts = outerProductOfRows(lambda_i)
    
    ### The second term needed for estimating the factors via regression
    secondPartToRegression = np.matmul((oneMinusD*Y).transpose(), 
                                       lambda_i)
    
    ### Stores estimated factors in a list
    listOfFts = []
    
    for timePoint in range(Y.shape[1]):
        ### Calculate F_t, for each t
        
        matrixWeights = oneMinusD[:, timePoint]
        
        ### Computes design matrix for regression
        designMatrix = weightedMatrixComp(tensorOfOuterProducts, matrixWeights)
        
        firstRegressionPart = inv(designMatrix)
        
        F_t = np.matmul(firstRegressionPart, secondPartToRegression[timePoint,:])
        
        listOfFts.append(F_t)
        
    F_tArray = np.stack(listOfFts, axis=1)
  
    ### Calculates complete, imputed matrix.
    L_hat = np.matmul(lambda_i, F_tArray)
    
    return(L_hat)


### The LAPIS algorithm
def LAPISDescent(Y, D, rank=None, max_iterations=1000, tolerance=1e-02):
    
    if rank is None: 
        
        raise ValueError('Must Specify the Rank Parameter!')
        
    elif np.round(np.floor(rank)-rank, 4) != 0:
        
        raise TypeError('Rank is not an Integer!')
        
    elif rank <= 0: 
        
        raise ValueError('Rank must be positive!')
    
    L_k = P_Omega(Y, D) ### Initialization
    
    for iteration in range(max_iterations):
        
        softThresholdSVD = svdReconstruction(L_k, K=rank, shrink=False)

        L_kPlus1 = P_Omega(Y, D) +  P_Omega(softThresholdSVD, 1-D) ### Impute missing values with shrunken SVD estimates
        
        if np.sum(((L_kPlus1-L_k)*D)**2)/np.sum(D) < tolerance: ### Stopping condition
            
            break
            
        L_k  = L_kPlus1 
        
    L_hat = svdReconstruction(L_kPlus1, K=rank, shrink=False)    ### Output 
    
    return L_hat

### The soft impute algorithm
def SoftImputeDescent(Y, D, lambda_penalty=None, max_iterations=1000, tolerance=1e-02):
    ### The soft impute algorithm
    
    L_k = P_Omega(Y, D) ### Initialization
    
    for iteration in range(max_iterations):
        
        softThresholdSVD = svdReconstruction(L_k, shrink=True, lambda_penalty=lambda_penalty)

        L_kPlus1 = P_Omega(Y, D) +  P_Omega(softThresholdSVD, 1-D) ### Impute missing values with shrunken SVD estimates
        
        if np.sum(((L_kPlus1-L_k)*D)**2)/np.sum(D) < tolerance: ### Stopping condition
            
            break
            
        L_k = L_kPlus1
        
    L_hat = svdReconstruction(L_kPlus1, shrink=True, lambda_penalty=lambda_penalty)    ### Output 
    
    return L_hat

### The r1Comp algorithm
def R1CompDescent(Y, D, lambda_penalty=None, r_init=40, tolerance=1e-04, max_iterations=1000):
    
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
    
        L_k_r = L_k.copy()
    
        if iter_number==1:
      
          parts_to_update = range(len(sigma_vec))
      
        else:
      
          parts_to_update = np.where(sigma_vec > 0)[0]
      
    
        for r_number in parts_to_update: ## Note: weights don't influence update of u,v (because we normalize)
      
      ## Two algorithms should be identical when matrix is just 1's (unless initialized differently)
      
            Us[:, r_number] = np.matmul(L_k_r, Vs[:, r_number])
      
            Us[:, r_number] = Us[:, r_number]/np.linalg.norm(Us[:, r_number])
      
            Vs[:, r_number] = np.matmul(L_k_r.transpose(), Us[:, r_number])
      
            Vs[:, r_number] = Vs[:, r_number]/np.linalg.norm(Vs[:, r_number])
      
            mult_thing = np.matmul(np.atleast_2d(Us[:, r_number]).transpose(), 
                                   np.atleast_2d(Vs[:, r_number]))

            sigma_vec[r_number] = max(0, shrink_operator(x=np.sum(mult_thing * L_k_r), 
                                                         lambda_penalty=lambda_penalty))
      
    #  print(mean(sigma_vec))

            L_k_r = L_k_r - sigma_vec[r_number] *  mult_thing
      
        L_k_plus_1 = L_k.copy()
    
        Z = L_k-L_k_r
    
        L_k_plus_1[D!=0] = Z[D!=0]

     #   first_num = (np.linalg.norm((L_k_plus_1-Z)*(1-D), 'fro')
     #       /np.linalg.norm(1-D, 'fro'))
    
    
        second_num = (np.linalg.norm((L_k_plus_1-L_k)*D, 'fro')\
                      /max(1, np.linalg.norm(1-D, 'fro')))
        
        condition1 = (second_num  < tolerance)
    
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


### SoftImpute class.
class SoftImpute:  
    
    def __init__(self, tuning_lambda=None, max_iterations=100, tolerance=1e-02):
        
        self.__tuning_lambda = np.unique(tuning_lambda)
        
        self.__max_iterations = max_iterations
        
        self.__tolerance = tolerance
    
    def getTuningLambda(self): ## Obtaining and changing grid of penalty terms
        
        return(self.__tuning_lambda)
    
    
    def setTuningLambda(self, new_lambda):
        
        self.__tuning_lambda = np.unique(new_lambda)
        
        
    def getMaxIterations(self): ### Getting and changing number of iterations for SoftImpute
        
        return(self.__max_iterations)
    
    
    def setMaxIterations(self, new_max_iterations):
        
        self.__max_iterations = new_max_iterations
        
   
    def getTolerance(self): ### Getting and changing minimum required step size for SoftImpute
        
        return(self.__tolerance)
    
    
    def setTolerance(self, new_tolerance):

        self.__tolerance = tolerance

    def fit(self, Y, D): ### Fit model, taking best penalty as the default penalty parameter
        
        YHat = SoftImputeDescent(Y, D, lambda_penalty=self.__tuning_lambda, max_iterations=self.__max_iterations, 
                                 tolerance=self.__tolerance)
        
        return(YHat)
        
        
### LAPIS class.
class LAPIS:  
    
    def __init__(self, rank=None, max_iterations=1000, tolerance=1e-04):
        
        if rank is None: 
            
            raise ValueError('Must specify rank for SVD!')
        
        self.__rank=rank
        
        self.__max_iterations = max_iterations
        
        self.__tolerance = tolerance
    
    def getRank(self): ## Obtaining and changing grid of penalty terms
        
        return(self.__rank)
    
    def setRank(self, new_rank):
        
        self.__rank = np.unique(new_lambda_grid)
        
    def getMaxIterations(self): ### Getting and changing number of iterations for SoftImpute
        
        return(self.__max_iterations)
    
    
    def setMaxIterations(self, new_max_iterations):
        
        self.__max_iterations = new_max_iterations
        
   
    def getTolerance(self): ### Getting and changing minimum required step size for SoftImpute
        
        return(self.__tolerance)
    
    
    def setTolerance(self, new_tolerance):

        self.__tolerance = tolerance

    def fit(self, Y, D): ### Fit model, taking best penalty as the default penalty parameter
        
        YHat = LAPISDescent(Y, D, rank=self.__rank, max_iterations=self.__max_iterations, 
                                 tolerance=self.__tolerance)
        
        return(YHat)        
    
### The r1Comp class. 
class R1Comp: 
    
    def __init__(self, tuning_lambda=None, r_init=40, tolerance=1e-04, max_iterations=1000):
        
        self.__tuning_lambda = tuning_lambda
        
        self.__r_init = r_init
        
        self.__max_iterations = max_iterations
        
        self.__tolerance = tolerance
        
        self.__r_init = r_init
    
    def getTuningLambda(self): ## Obtaining and changing grid of penalty terms
        
        return(self.__tuning_lambda)
    
    
    def setTuningLambda(self, new_lambda):
        
        self.__tuning_lambda = np.unique(new_lambda)
        
        
    def getMaxIterations(self): ### Getting and changing number of iterations for SoftImpute
        
        return(self.__max_iterations)
    
    
    def setMaxIterations(self, new_max_iterations):
        
        self.__max_iterations = new_max_iterations
        
   
    def getTolerance(self): ### Getting and changing minimum required step size for SoftImpute
        
        return(self.__tolerance)
    
    
    def setTolerance(self, new_tolerance):

        self.__tolerance = tolerance

    def fit(self, Y, D): ### Fit model, taking best penalty as the default penalty parameter

        YHat = R1CompDescent(Y, D, r_init=self.__r_init, tolerance=self.__tolerance, 
                             max_iterations=self.__max_iterations, 
                             lambda_penalty=self.__tuning_lambda)
        
        return(YHat)       
    
    
### A factor model approach to matrix completion
class FactorModel: 
    
    def __init__(self, numFactors):
        
        self.__numFactors = numFactors
        
    def fit(self, Y, D): ### Fit model, taking best penalty as the default penalty parameter

        YHat = factorModelCompletion(Y, D, numFactors = self.__numFactors)
        
        return(YHat)
        
        
##### Utilities for validating ANY matrix completion method    (NEEDS TESTING)
def validationGivenTuning(Y, D, tuningValue, method, numFolds=5, max_iterations=1000, tolerance=1e-02,
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
            
        elif isinstance(method(1), FactorModel):
            
            builtMethod = method(tuningValue)  
    
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
            
            estimateThisFold = builtMethod.fit(Y, DThisFold)
            
            ### Validation error
            errorThisFold = np.sum(((estimateThisFold-Y)**2)*(1-D)*DResample)\
            /np.maximum(1, np.sum((1-D)*DResample))
            
            errorList.append(errorThisFold)

        return(np.mean(errorList))    
    

### Parameter tuning via cross-validation
def validation(Y, D, method, tuningGrid=None, numFolds=3, max_iterations=1000, tolerance=1e-02,
              r_init=None):

    ### The R1Comp method has one different parameter, so it must be initialized separately
    if isinstance(method(1), R1Comp):
        
        if r_init is None: 
        
            raise ValueError('Must Specify the Rank Parameter!')
        
        elif np.round(np.floor(r_init)-r_init, 4) != 0:
        
            raise TypeError('Rank is not an Integer!')
        
        elif r_init <= 0: 
        
            raise ValueError('Rank must be positive!')
            
        partialFunction= partial(validationGivenTuning, Y, D, method=method, numFolds=numFolds, max_iterations=max_iterations, 
                               tolerance=tolerance, r_init=r_init)
            
    elif isinstance(method(1), FactorModel):
        
        partialFunction= partial(validationGivenTuning, Y, D, method=method)    
        
    else:
        
        partialFunction= partial(validationGivenTuning, Y, D, method=method, numFolds=numFolds, max_iterations=max_iterations, 
                               tolerance=tolerance)
            
    with Pool(4) as p: 

        errorMatrix = p.map(partialFunction, tuningGrid)

    return(pd.Series(dict(zip(tuningGrid, errorMatrix))))









