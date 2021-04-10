import pandas as pd

import numpy as np

from numpy.linalg import svd, matrix_rank, lstsq

from scipy.stats import bernoulli

from multiprocessing import Pool

from functools import partial

### UNWEIGHTED Methods for matrix completion.
### Document the unweighted methods separately. 


def P_Omega(Y, D): ## Projects data onto set of observed entries,
                   ## Where D_ij=0 means observed, and D_ij=1 mean missing
    
    return(Y*(1-D))

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
    SVDReconstruction = np.matmul(np.atleast_2d(u[:, 0:K]), np.matmul(np.atleast_2d(S[0:K, 0:K]),np.atleast_2d(vh[0:K, :])))
    
    return(SVDReconstruction)



def LAPISDescent(Y, D, rank=None, max_iterations=1000, tolerance=1e-02):
    ### The LAPIS algorithm
    
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


### Remove weight matrix for now, until you've implemented weighted SoftImpute
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

    
    
    
    
    
    
    
    
    
    
    
class SoftImpute: ### SoftImpute class. 
    
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
        
        
class LAPIS: ### LAPIS class. 
    
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
    
    
class R1Comp: ### SoftImpute class. 
    
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
        
    else:
        
        partialFunction= partial(validationGivenTuning, Y, D, method=method, numFolds=numFolds, max_iterations=max_iterations, 
                               tolerance=tolerance)
            
    with Pool(4) as p: 

        errorMatrix = p.map(partialFunction, tuningGrid)

    return(pd.Series(dict(zip(tuningGrid, errorMatrix))))
