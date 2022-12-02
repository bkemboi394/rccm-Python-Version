# -*- coding: utf-8 -*-


import numpy as np
import math
from scipy import stats
from rccSim import rccsim


p=10
x = np.random.random((p,p))
M = np.random.random((p,p))
nu=20
def dwishart (x, M, nu, logged = True):
  x = (x + np.transpose(x)) / 2
  M = (M + np.transpose(M)) / 2
  p = x.shape[0]
  lnumr = (nu - p - 1) /( 2 * math.log(np.linalg.det(x))) - (nu / (2 * sum(np.diag(np. linalg. inv(x) * x))))
  ldenom = (nu * p / 2) * math.log(2) + (nu / 2) * math.log(np.linalg.det(1 / nu * M)) + (p * (p - 1) / 4)*\
                     math.log(math.pi)+ sum([math.lgamma(nu / 2 + (1 - j) / 2) for j in range(1,p+1)])
  return (lnumr - ldenom) if logged else (math.exp(lnumr - ldenom))
    
              
def adj(mat, thresh = 0.001):
    return((abs(mat) > thresh) + 0)


def zToA(z):
    K = len(z)
    A = np.zeros((K,K))
    #print(A)
    for r in range(K):
        for s in range(K):
           if z[r] != 0 and z[s] != 0:
               A[r, s] = int(z[r] == z[s])
           else:
               A[r, s] = 0
    return A



#Defined a function to get indices corresponding to the 4 types of triangular matrix
#There's an existing similar function (np.tril_indices_from(np_array, k= -1)) but its behavior is not exactly the same\
    
def triu_or_tril_indices(np_array,tri_type, diagonal_type  ):
    if tri_type == "triu" and diagonal_type == "true":
        ud_boolean = np.triu(np_array, k = 0).astype(bool)
        nrows,ncols = ud_boolean.shape
        ud_indices = []
        for x in range(nrows):
            for y in range(ncols):
                if ud_boolean[x,y] == True:
                    ud_indices.append((x,y))

        rows_ud=[]
        cols_ud=[]
        for i in ud_indices:
            rows_ud.append(i[0])
            cols_ud.append(i[1])
        return (rows_ud,cols_ud)
    elif tri_type == "triu" and diagonal_type == "false":
        u_boolean = np.triu(np_array, k = 1).astype(bool)
        nrows,ncols = u_boolean.shape
        u_indices = []
        for x in range(nrows):
            for y in range(ncols):
                if u_boolean[x,y] == True:
                    u_indices.append((x,y))

        rows_u=[]
        cols_u=[]
        for i in u_indices:
            rows_u.append(i[0])
            cols_u.append(i[1])
        return (rows_u,cols_u)
    elif tri_type == "tril" and diagonal_type == "true":
        ld_boolean = np.tril(np_array, k = 0).astype(bool)
        nrows,ncols = ld_boolean.shape
        ld_indices = []
        for x in range(nrows):
            for y in range(ncols):
                if ld_boolean[x,y] == True:
                    ld_indices.append((x,y))

        rows_ld=[]
        cols_ld=[]
        for i in ld_indices:
            rows_ld.append(i[0])
            cols_ld.append(i[1])
        return (rows_ld,cols_ld)
    elif tri_type == "tril" and diagonal_type == "false":
        l_boolean = np.tril(np_array, k = -1).astype(bool)
        nrows,ncols = l_boolean.shape
        l_indices = []
        for x in range(nrows):
            for y in range(ncols):
                if l_boolean[x,y] == True:
                    l_indices.append((x,y))

        rows_l=[]
        cols_l=[]
        for i in l_indices:
            rows_l.append(i[0])
            cols_l.append(i[1])
        return (rows_l,cols_l)
    
              
x=[1,1,1,1,1,2,2,2,2,2]
y=[1,1,1,1,1,2,2,2,2,2]


def randCalc(x,y):
        
        Ahat = zToA(x)[np.tril_indices_from(zToA(x ), k= -1)]
        #print(Ahat)
        A0 = zToA(y)[np.tril_indices_from(zToA(y), k= -1)]
        #print(A0)
        return((sum((Ahat - A0) == 2) + sum((Ahat - A0) == 0)) / math.comb(len(x), 2))
        
       
    
    
f = randCalc(x,y)


G,clustSize,p,n, overlap = 2,(67,37),10,177,0.5
rho,esd, gtype, eprob = 0.10,0.05,"hub",0.5


sim = rccsim(G,clustSize,p,n,overlap,rho,esd,gtype,eprob)
#print(sim)


x = sim[0]
g0s = sim[1]
omega0s = sim[2]
gks = sim[3]
omegaks = sim[4]
ws= sim[5]

 #Uncomment below to assess each of the outputs
       
# print(len(g0s))
#print(len(simData))
#print(len(simData[0])) 
#print(len(omega0s))
# print(len(gks))
# print(len(Omegaks))
#print(ws)

lambda2 = 135


# Will need clarification on this function
def rccmLogLike(omegaks,omega0s,x,ws,lambda2):
    G = len(omega0s)
    K = len(omegaks)
    
    mll = 0
    for k in range(K):
        lk1 = stats.multivariate_normal.logpdf(x[k],mean = np.mean(x[k], axis =0) , cov = np.linalg.inv(omegaks[k]))
        if ws[k] == 1:
          lk2 = dwishart(omegaks[k], M = omega0s[np.where(ws[k] == 1)],logged = True, nu = lambda2)
          mll+= lk1 + lk2
        else:
           list_g = [g for g in range(G)]
           lk2 = math.log(sum(list(map(lambda g: ws[g,k]*dwishart(omegaks[k], \
                                                M = omega0s[g],logged = False, nu = lambda2),list_g))))
           mll+= lk1 + lk2  
    return mll
     
    
    
    
rll = rccmLogLike(omegaks, omega0s, x, ws, lambda2)
#rll


def aic(omegaks, omega0s, ws, x, lambda2):
    K = len(omegaks)
    G = len(omega0s)
    
    X =[k for k in range(K)]
    dfks = map(lambda k: sum(adj(omegaks[k][np.tril_indices_from(omegaks[k])])),X)
    
    list_G = [g for g in range(G)]
    dfgs = map(lambda g: sum(adj(omega0s[g][np.tril_indices_from(omega0s[g])])),list_G)
    
    
    modelDim = sum(list(dfks), list(dfgs))
    mll = rccmLogLike(omegaks = omegaks, omega0s = omega0s, ws = ws, x = x, lambda2 = lambda2)

    aic = 2*modelDim - 2*mll
    return(aic)

    
    
    
    
    
    
    
    
    