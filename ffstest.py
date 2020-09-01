import numpy as np

from math import pi  
import itertools


import multiprocessing
import functools
import time
import cProfile

import GillespieTST_0501_pyrgrow as TST

import rgrow

# Load all maps
JmatE,JmatN,J_friends,smaps = TST.load_interaction_matrices();

# common initialization
# max grid size
n=62
init_conf = [[0]* (n+2)] * (n+2) # one layer margin on all sides

# Load interaction matrix data
q = len(JmatE)  # number of species+1. 0 = solvent

# User defined parameters here:
beta = 1
params ={"JmatE": JmatE,"JmatN": JmatN, "J_friends": J_friends, "n":n, "q":q, \
         "save_config_every":1, "kf": 1e6, "a":7.2} #, "Gmc": 10*beta}

## Code to process patterns

# Run FFS from lambda_2 to a high lambda_n. 
    # Target fixed variance for each lambda_i
    # Compute Z factor and report actual nucleation rate in units of M/s
def prop_forward_constvar(Gse,M,cvar,prams):
    
    params = prams.copy()    
    params["Gse"]=Gse
    
    sys = rgrow.PyStaticKTAM.from_raw(params["conc_vec"],JmatN.T*Gse, JmatE*Gse)

    # maintain constant variance.
    # var/mean^2 ~ (1-p)/(pM).. which decreases as p \to 1.
    # p_next > p_now .. so lower variance in the future. 
    # Use p_{now} to compute the desired variance. 
    
    # L2 -> L3 and then Li -> Li+1
    L_configs = [0] * 500
    pf_vec=[0]*500

    # Set up the special initial ratchet L2 -> L3 (dimers to trimers)
    L_configs[3],fin_size,z = TST.special_prop_L2_2_Lp_target(M,[3],sys,params);  pf_vec[3] = M/len(fin_size)
    print('z =',z)
    pf_vec[2] = (params["kf"]*z*np.exp(-2*params["a"]))


    Mnew = M
    
    lambda_sizes = list(range(3,100))

    overall_start_time = time.time()
    pf_vec_coll=[]
    for idx,init_size in enumerate(lambda_sizes[:-1]):
        target_size = lambda_sizes[idx+1]
        start_time = time.time()

        L_configs[target_size],fin_sze = TST.prop_Li_2_Lip1_targetC(L_configs[init_size],Mnew,[target_size],sys,params)        
        pf_vec[target_size] = Mnew/len(fin_sze)
                
        
        # compute Mnew to have a fixed variance
        Mnew = max(int((1-pf_vec[target_size])/(cvar)),50) 
        #print(Mnew)
        
        #print('pf(',init_size,'|',target_size,') = ',pf_vec[target_size], ' : Time = ', np.ceil(time.time()-start_time),' s')        

         # while the mean of last 4 values were < 0.97
        if np.mean(pf_vec[target_size-3:target_size]) > 0.98:
            break            

    print('total time = ',time.time()-overall_start_time)
    
    return L_configs,pf_vec,target_size,np.exp((np.cumsum(np.log(pf_vec[2:target_size])))[-1])


# Given a conc_pattern, run FFS code for each structure and report nucleation rate
def analyze_pattern(pattern_filename,Gse):
    cvar = 0.001  # Sets the target accuracy. 0.001 might be ~ 1.2x accurate when runs are ~15 secs long. Less accurate for longer runs.

    hcv,acv,mcv,cv = TST.load_concentration_pattern(pattern_filename,smaps)
    
    params["conc_vec"] = hcv
    L_configs,pf_vec,target_size,nuc_rate_h = prop_forward_constvar(Gse,1000,cvar,params)
    print(nuc_rate_h,' M/s')
    
    params["conc_vec"] = acv
    L_configs,pf_vec,target_size,nuc_rate_a = prop_forward_constvar(Gse,1000,cvar,params)
    print(nuc_rate_a,' M/s')
    
    params["conc_vec"] = mcv
    L_configs,pf_vec,target_size,nuc_rate_m = prop_forward_constvar(Gse,1000,cvar,params)
    print(nuc_rate_m,' M/s')
    
    return [nuc_rate_h,nuc_rate_a,nuc_rate_m]
    


#a = prop_forward_constvar(10.5, 3, 0.4, p)

# Actual runs

np.load('myApat.npy');

#TST.show_conc_patterns('myApat.npy',smaps)

analyze_pattern('myApat.npy',6.5)
