# -*- coding: utf-8 -*-
import numpy as np
from sklearn import covariance
def randCov(x, lambda1, lambda2, lambda3 = 0,delta = 0.001, max_iters = 100):
    def spcov_bcd(samp_cov, rho, lambda2, lambda3, initial = None):
        p = samp_cov.shape[0]
        if initial is None:
            Sigma = samp_cov + 0.01*np.diag(p)
        else:
            Sigma = initial
        delta = 1e-04
        Sigma_old = np.zeros((p,p))
        count2 = 0
        while (max(abs(Sigma - Sigma_old)) > delta):
            # loop 1: Sigma convergence
            Sigma_old = Sigma
            for i in range(p):
                # loop 2: update each row/column of Sigma
                Omega11 = np.linalg.inv(Sigma[-i, -i])
                beta = Sigma[-i, i]
        
                S11 = samp_cov[-i, -i]
                s12 = samp_cov[-i, i]
                s22 = samp_cov[i, i]
                a = beta.T * Omega11 * S11 * Omega11 * beta - 2 * s12.T * Omega11 * beta + s22
                if rho == 0:
                    gamma = a
                elif a == 10^(-10): #comeback to this
                    gamma = a
                else:
                    gamma = (-1 / (2 * rho) + (1 / (4 * rho^2) + a / rho)^0.5)
                V = Omega11 * S11 * Omega11 / gamma + rho * Omega11
                u = s12.T * Omega11 / gamma
                beta_old = 0
                while (max(abs(beta - beta.old)) > delta):
                    # loop 3: off-diagonals convergence
                    beta_old = beta
                    for j in range(p-1):
                        # loop 4: each element
                        temp = u[j] - V[j, -j] * beta[-j]
                        beta[j] = np.sign(temp) * max(0, abs(temp) - rho) / V[j, j]
                Sigma[i, -i] = beta.T
                Sigma[-i, i] = beta.T
                Sigma[i, i] = gamma + beta.T * Omega11 * beta
            count2 = count2 + 1
            if count2 > 100:
                print("Omega0 fails to converge for lam1 =", rho, "lam2 =", lambda2 / K, "lam3 =", lambda3)
                break
        return(Sigma)
    # Inputs:
    K = len(x)
    p = x[0].shape[1]
    Sa = map(np.cov,x, simplify = "array")
    Sl = map(np.cov,x)
    # Initial values
    Omega0 = np.linalg.inv(map(np.mean,Sa) + np.diag(1, p) * 0.01)

    Omegas = map(lambda x1: np.linalg.inv(x1 + np.diag(1, p) * 0.01),Sl, simplify = "array")
    Omega0_old = np.zeros((p,p))
    Omegas_old = np.zeros((p,p))
    count = 0
    rho = lambda1 / (1 + lambda2 / K)
    # Start BCD algorithm
    while (max(abs(Omega0 - Omega0_old)) > delta or max(abs(Omegas - Omegas_old)) > delta):
        # Exit if exceeds max.iters
       if (count > max_iters):
                 print("Failed to converge for lambda1 =", lambda1, ", lambda2 =", lambda2,\
                       ", lambda3 =", lambda3,"delta0 =", max(abs(Omega0 - Omega0.old)),\
                               ", deltaK =", max(abs(Omegas - Omegas.old)))
                 # Returning results
                 res = list(Omega0, Omegas)
                 #names_res=["Omega0", "Omegas"]
                 return(res)
        # Record current Omega0 & Omegas
       Omega0_old = Omega0
       Omegas.old = Omegas
       # 1st step:
       sk = map(lambda x1: (np.linalg.inv(Omega0) * lambda2 / K + x1) / (1 + lambda2 / K),Sl, simplify = False)
       for k in range(K):
           covariance.graphical_lasso(sk[k], rho)
           # 2nd step:
       if lambda3 == 0:
         Omega0 = map(np.mean, Omegas)
       else:
           s0 = map(np.mean,Omegas)
           log = subprocess.check_output(Omega0 = spcov_bcd(s0, lambda3, lambda2 = lambda2, lambda3 = lambda3))
       count+=1 
    res = list(Omega0, Omegas)
   # names(res) <- c("Omega0", "Omegas")
    return(res)
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                