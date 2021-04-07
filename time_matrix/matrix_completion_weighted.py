def r1CompDescentWeighted(Y, D, weightMatrix, lambda_penalty=None, r_init=40, tolerance=1e-04, max_iterations=1000):
    
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
      
            Us[:, r_number] = np.apply_along_axis(lambda x: 1/np.sum(x), 1, weightMatrix)\
        *np.matmul((weightMatrix*L_k_r), Vs[:, r_number])
      
            Us[:, r_number] = Us[:, r_number]/np.linalg.norm(Us[:, r_number])
      
            Vs[:, r_number] = np.apply_along_axis(lambda x: 1/np.sum(x), 0, weightMatrix)\
        *np.matmul((weightMatrix*L_k_r).transpose(), Us[:, r_number])
      
            Vs[:, r_number] = Vs[:, r_number]/np.linalg.norm(Vs[:, r_number])
      
            mult_thing = np.matmul(np.atleast_2d(Us[:, r_number]).transpose(), 
                                   np.atleast_2d(Vs[:, r_number]))

            special_number = np.sum(weightMatrix*(mult_thing**2))
      
      # mu/special_number
      
            sigma_vec[r_number] = max(0, shrink_operator(x=np.sum(mult_thing * L_k_r*weightMatrix)\
                                                         /special_number, lambda_penalty=lambda_penalty))
      
    #  print(mean(sigma_vec))

            L_k_r = L_k_r - sigma_vec[r_number] *  mult_thing
      
        L_k_plus_1 = L_k
    
        Z = L_k-L_k_r
    
        L_k_plus_1[D!=0] = Z[D!=0]

        #first_num = (np.linalg.norm(P_Omega(weightMatrix*(L_k_plus_1-Z), D), 'fro')
        #        /np.linalg.norm(P_Omega(weightMatrix, D), 'fro'))
    
    
        #second_num = (np.linalg.norm(weightMatrix*(L_k_plus_1-L_k), 'fro')\
         #             /np.linalg.norm(weightMatrix, 'fro'))
        
        theNum = np.linalg.norm((L_k_plus_1-L_k)*(1-D), 'fro')\
                      /max(1, np.linalg.norm(1-D, 'fro'))
    
        condition1 = (theNum  < tolerance)
    
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
