#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy.spatial.distance import pdist, squareform

from math import pi  
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import collections
from scipy.optimize import least_squares
import itertools
import collections
import rgrow
from typing import List, Iterable, Collection

def load_concentration_pattern(filename,smaps):
    muvec=np.load(filename) # DNA tiles are numbered 0-916
    cv = np.concatenate(([0], np.exp(-muvec)))

    hcv = np.concatenate(([0], 0*np.exp(-muvec)))
    acv = np.concatenate(([0], 0*np.exp(-muvec)))
    mcv = np.concatenate(([0], 0*np.exp(-muvec)))

    for i in smaps["htiles"]:
        hcv[i] = np.exp(-muvec[i-1])  # muvec numbered from 0 - 916 for DNA tiles. 

    for i in smaps["mtiles"]:
        mcv[i] = np.exp(-muvec[i-1])    

    for i in smaps["atiles"]:
        acv[i] = np.exp(-muvec[i-1])   
        
    return hcv,acv,mcv, cv

def show_conc_patterns(filename,smaps):
    
    hcv,acv,mcv,cv = load_concentration_pattern(filename,smaps)
        
    lcv = np.log(cv)
    lcv[0] = np.mean(lcv)
    
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,6))

    heatmap = [[ lcv[int(tle)+1] for tle in rw] for rw in smaps['hmap']]
    im = ax1.imshow(heatmap)
    fig.colorbar(im,ax=ax1,fraction=0.046, pad=0.04)    
    ax1.axis('off')
    
    heatmap = [[ lcv[int(tle)+1] for tle in rw] for rw in smaps['amap']]
    im = ax2.imshow(heatmap)
    fig.colorbar(im,ax=ax2,fraction=0.046, pad=0.04)
    ax2.axis('off')


    heatmap = [[ lcv[int(tle)+1] for tle in rw] for rw in smaps['mmap']]
    im = ax3.imshow(heatmap)
    fig.colorbar(im,ax=ax3,fraction=0.046, pad=0.04)
    ax3.axis('off')

    #plt.show()
    
    #plt.colorbar()
    

def load_interaction_matrices():
    raw_JmatE = np.load('contactMatE.npy')
    raw_JmatN = np.load('contactMatN.npy')
    
    smaps ={}
    # Load all maps
    smaps["hmap"]=np.loadtxt(open("locmap-h.txt", "rb"))  # DNA tiles: 0 - 916, Solvent = -1
    temp_htiles= smaps["hmap"].flatten()
    smaps["htiles"] = [int(temp_htiles[i])+1 for i in np.where(temp_htiles!= -1)[0]] # Solvent = 0, DNA = 1 - 917

    smaps["amap"]=np.loadtxt(open("locmap-a.txt", "rb"))
    temp_atiles= smaps["amap"].flatten()
    smaps["atiles"] = [int(temp_atiles[i])+1 for i in np.where(temp_atiles!= -1)[0]]

    smaps["mmap"]=np.loadtxt(open("locmap-m.txt", "rb"))
    temp_mtiles= smaps["mmap"].flatten()
    smaps["mtiles"] = [int(temp_mtiles[i])+1 for i in np.where(temp_mtiles!= -1)[0]]



    # pad with a row and col of zeros to represent solvent
    JmatE = np.vstack(( np.zeros(len(raw_JmatE)+1), np.column_stack((np.zeros(len(raw_JmatE)),raw_JmatE))))
    JmatN = np.vstack(( np.zeros(len(raw_JmatN)+1), np.column_stack((np.zeros(len(raw_JmatN)),raw_JmatN))))
    
    J_friends={}
    J_friends["W_friends"] = [list(np.where(JmatE[::,s])[0]) for s in range(len(JmatE))]
    J_friends["E_friends"] = [list(np.where(JmatE[s,::])[0]) for s in range(len(JmatE))]

    J_friends["S_friends"] = [list(np.where(JmatN[::,s])[0]) for s in range(len(JmatN))]
    J_friends["N_friends"] = [list(np.where(JmatN[s,::])[0]) for s in range(len(JmatN))]

    return JmatE,JmatN,J_friends,smaps


def participation_heatmap(list_of_pieces,structure_map):
    tally=collections.Counter(np.array([conf.flatten() for conf in list_of_pieces]).flatten())
    tally[0]=0
    heatmap = [[ tally[tle+1] for tle in rw] for rw in structure_map]
    
    return heatmap


def shifted_size(spinsys):
    shifted_spinsys = np.multiply(spinsys[0:-2,1:-1], spinsys[1:-1,1:-1]) + np.multiply(spinsys[2:,1:-1], spinsys[1:-1,1:-1]) + \
                np.multiply(spinsys[1:-1,0:-2], spinsys[1:-1,1:-1]) + np.multiply(spinsys[1:-1,2:], spinsys[1:-1,1:-1]) 

    return np.count_nonzero(shifted_spinsys)


def halting_func(spinsys,halting_data,params):    
    # compute number of bonds
    
    #energy,nbnds = total_energy_bonds(spinsys,params)    
    #nbnds = fast_total_bonds(spinsys,params)    
    #haltnow = (nbnds in halting_data["state_A"]) or (nbnds in halting_data["state_B"])
    
    sze = np.count_nonzero(spinsys)
    haltnow = (sze in halting_data["state_A"]) or (sze in halting_data["state_B"])

    return haltnow


def remove_zeros(struct_conf):
    
    newr = struct_conf[~np.all(struct_conf == 0, axis=1)]
    return newr[:, ~np.all(newr == 0, axis=0)]

def extract_A2C1(struct_history,state_A, state_L1):  # Locate all

    size_arr = [np.count_nonzero(confs) for confs in struct_history]

    C1_configs = []
    A2C1_traj =[]

    for idx,cur_state in enumerate(size_arr):    
        if cur_state in state_A:
            visited_A_but_not_L1 = True
            last_time_in_A = idx

        if cur_state in state_L1:
            if visited_A_but_not_L1 == True:
                # save this state
                C1_configs.append(idx)
                A2C1_traj.append((last_time_in_A,idx))
            visited_A_but_not_L1 = False

    return C1_configs, A2C1_traj



def fast_total_bonds(spinsys,params):
    
    nb = 0
    JmatN = params["JmatN"]
    JmatE = params["JmatE"]

    useful_spots = np.where(spinsys!=0) # need to convert to regular indices? 
    
    # run over structure and count bonds
    for (x,y) in zip(useful_spots[0],useful_spots[1]):        
        old_tile = spinsys[x,y]    
        tile_N = spinsys[x-1,y]
        tile_E = spinsys[x,y+1]
        tile_W = spinsys[x,y-1]
        tile_S = spinsys[x+1,y]

        #For row i and column j of contactMatN.npy, tile j is north of tile i.
        nbnds = JmatN[old_tile,tile_N] + JmatN[tile_S,old_tile] + JmatE[old_tile,tile_E] +JmatE[tile_W,old_tile]        
        nb =nb + nbnds
                  
    return nb/2



def prop_Li_2_Lip1(Li_configs,Mi,state_Lip1,params,state_A = [0]):
        
    # Pick Mi random configs from Li_configs
    rnd_idx_list = np.random.choice(len(Li_configs),Mi)
    
    # Halting conditions -  n_bonds criterion
    halting_data = {"monte_carlo_steps": 5000, "state_B":state_Lip1, "state_A": state_A} # details of when to halt

    fin_size =[]
    Lip1_configs =[]

    for rndL1_idx in rnd_idx_list:

        # set random Li config as initial config
        init_conf=Li_configs[rndL1_idx]

        struct_history = propagate_Gillespie(init_conf,params,halting_data)
        
        # store nbonds of final structure
        fin_size.append(np.count_nonzero(struct_history[-1]))
                
        if fin_size[-1] in state_Lip1: # if it reached the desired set of states, save this config
            Lip1_configs.append(struct_history[-1].copy())    
    
    return Lip1_configs,fin_size


def prop_Li_2_Lip1_targetC(Li_configs: List[rgrow.PyStateKTAM], CMi,state_Lip1,sys,params,state_A = [0]):
        
    # Pick Mi random configs from Li_configs  # need to keep sampling until we have enough.. 
    rnd_idx_list = np.random.choice(len(Li_configs),CMi)  # an initial batch of CMi.. but might need to keep sampling
    
    fin_size =[]
    Lip1_configs =[]

    #for rndL1_idx in rnd_idx_list:
        
    idx = 0 # index
        
    while len(Lip1_configs) < CMi:
     
        # set random Li config as initial config
        init_conf=Li_configs[rnd_idx_list[idx]].copy()

        init_conf.evolve_in_size_range(sys, state_A[0], state_Lip1[0], 20000)
        
        # store nbonds of final structure
        fin_size.append(init_conf.ntiles())
                
        if fin_size[-1] in state_Lip1: # if it reached the desired set of states, save this config
            Lip1_configs.append(init_conf)    
            
        idx = idx + 1
        
        if idx % (CMi-1) == 0:  # time to expand the list
            rnd_idx_list = np.concatenate((rnd_idx_list,np.random.choice(len(Li_configs),CMi)))
    
    return Lip1_configs,fin_size



# create Lp ensemble from L2 ensemble of dimers 
def special_prop_L2_2_Lp(M2,target_state,sys,prms): 
    
    params = prms.copy()        
    cv= params["conc_vec"].copy()

    # go through each species, find their E partner and N partner. 
    pair_set_WE = [[ (i,x) for x in np.where(params["JmatE"][i,::])[0]] for i in range(len(params["JmatE"]))]
    WE_pairs = [item for sublist in pair_set_WE for item in sublist]
    WE_cv=[cv[x]*cv[y] for (x,y) in WE_pairs]

    pair_set_NS = [[ (i,x) for x in np.where(params["JmatN"][::,i])[0]] for i in range(len(params["JmatN"]))]
    NS_pairs = [item for sublist in pair_set_NS for item in sublist]
    NS_cv =[cv[x]*cv[y] for (x,y) in NS_pairs]
    
    NSWE_cv = NS_cv+ WE_cv   
    rnd_idx_list_total = np.random.choice(len(NSWE_cv),M2,p = NSWE_cv/np.sum(NSWE_cv))
        
    fin_size =[]
    
    Lp_configs = []
 
    create = True
    for rnd_dimer_idx in rnd_idx_list_total:

        if create:         
            if rnd_dimer_idx >= len(NS_cv):  # this is an WE pair        
                a,b = WE_pairs[rnd_dimer_idx-len(NS_cv)]
                init_conf = rgrow.PyStateKTAM.create_we_pair(sys, a, b)

            else:  # this is an NS pair        
                a,b = NS_pairs[rnd_dimer_idx]
                init_conf = rgrow.PyStateKTAM.create_ns_pair(sys, a, b)
        else:
            if rnd_dimer_idx >= len(NS_cv):  # this is an WE pair        
                a,b = WE_pairs[rnd_dimer_idx-len(NS_cv)]
                init_conf.set_point(sys, size//2, size//2)
            else:  # this is an NS pair        
                a,b = NS_pairs[rnd_dimer_idx]
                init_conf = rgrow.PyStateKTAM.create_ns_pair(sys, a, b)

        struct_history = init_conf.evolve_in_size_range(sys, [0], target_state[0], 1000)        

        # store nbonds of final structure
        fin_size.append(struct_history.ntiles())
                
        if fin_size[-1] in target_state: # if it reached the desired set of states, save this config
            Lp_configs.append(struct_history[-1].copy())
            create = True
            
        
    return Lp_configs,fin_size




# create Lp ensemble from L2 ensemble of dimers 
def special_prop_L2_2_Lp_target(CM2,target_state,sys,prms): 
    
    params = prms.copy()        
    cv= params["conc_vec"].copy()

    # go through each speices, find their E partner and N partner. 
    pair_set_WE = [[ (i,x) for x in np.where(params["JmatE"][i,::])[0]] for i in range(len(params["JmatE"]))]
    WE_pairs = [item for sublist in pair_set_WE for item in sublist]
    WE_cv=[cv[x]*cv[y] for (x,y) in WE_pairs]

    pair_set_NS = [[ (i,x) for x in np.where(params["JmatN"][::,i])[0]] for i in range(len(params["JmatN"]))]
    NS_pairs = [item for sublist in pair_set_NS for item in sublist]
    NS_cv =[cv[x]*cv[y] for (x,y) in NS_pairs]
    
    NSWE_cv = NS_cv+ WE_cv   
    z = np.sum(NSWE_cv) # normalization for dimers
    
    rnd_idx_list_total = np.random.choice(len(NSWE_cv),CM2,p = NSWE_cv/np.sum(NSWE_cv))
    #print(rnd_idx_list_total)
    
    fin_size =[]
        
    Lp_configs = []
 
    idx = 0
    #for rnd_dimer_idx in rnd_idx_list_total:
    while len(Lp_configs) < CM2:
                            
        rnd_dimer_idx = rnd_idx_list_total[idx]

        s = params['n']+2
        s2 = s//2

        create = True
        
            # set up initial condition with dimer
        if create:    
            create = False
            if rnd_dimer_idx >= len(NS_cv):  # this is an WE pair        
                a,b = WE_pairs[rnd_dimer_idx-len(NS_cv)]
                init_conf = rgrow.PyStateKTAM.create_we_pair(sys, a, b, s)

            else:  # this is an NS pair        
                a,b = NS_pairs[rnd_dimer_idx]
                init_conf = rgrow.PyStateKTAM.create_ns_pair(sys, a, b, s)
        else:
            if rnd_dimer_idx >= len(NS_cv):  # this is an WE pair        
                a,b = WE_pairs[rnd_dimer_idx-len(NS_cv)]
                init_conf.set_point(sys, s2, s2, a)
                init_conf.set_point(sys, s2, s2+1, b)
                assert init_conf.ntiles() == 2

            else:  # this is an NS pair        
                a,b = NS_pairs[rnd_dimer_idx]
                init_conf.set_point(sys, s2, s2, a)
                init_conf.set_point(sys, s2+1, s2, b)
                assert init_conf.ntiles() == 2

        
        init_conf.evolve_in_size_range(sys, 0, target_state[0], 1000)        

        # store nbonds of final structure
        fin_size.append(init_conf.ntiles())
                
        if fin_size[-1] in target_state: # if it reached the desired set of states, save this config
            Lp_configs.append(init_conf) 
            create = True
            
        idx = idx + 1
        
        if idx % (CM2-1) == 0:  # time to expand the list
            #print('expanding')
            rnd_idx_list_total = np.concatenate((rnd_idx_list_total,np.random.choice(len(NSWE_cv),CM2,p = NSWE_cv/np.sum(NSWE_cv))))             
    return Lp_configs,fin_size, z


# For given conc vector, interaction matrix in prams, 
    # run FFS from lambda_2 to a high lambda_n. 
    # Target fixed variance for each lambda_i
    # Compute Z factor and report actual nucleation rate in units of M/s
def report_nucleation_rate(Gse,M,variance_target,prams,only_rate=True):
    
    cvar = variance_target
    params = prams.copy()    
    params["Gse"]=Gse

    # maintain constant variance.
    # var/mean^2 ~ (1-p)/(pM).. which decreases as p \to 1.
    # p_next > p_now .. so lower variance in the future. 
    # Use p_{now} to compute the desired variance. 
    
    # L2 -> L3 and then Li -> Li+1
    L_configs = [0] * 500
    pf_vec=[0]*500

    # Set up the special initial ratchet L2 -> L3 (dimers to trimers)
    L_configs[3],fin_size,z = special_prop_L2_2_Lp_target(M,[3],params);  pf_vec[3] = M/len(fin_size)
    #print('z =',z)
    pf_vec[2] = (params["kf"]*z*np.exp(-2*params["a"]))


    Mnew = M
    
    lambda_sizes = list(range(3,100))

    overall_start_time = time.time()
    pf_vec_coll=[]
    for idx,init_size in enumerate(lambda_sizes[:-1]):
        target_size = lambda_sizes[idx+1]
        start_time = time.time()

        L_configs[target_size],fin_sze = prop_Li_2_Lip1_targetC(L_configs[init_size],Mnew,[target_size],params)        
        pf_vec[target_size] = Mnew/len(fin_sze)
                
        
        # compute Mnew to have a fixed variance
        Mnew = max(int((1-pf_vec[target_size])/(cvar)),50) 
        #print(Mnew)
        
        #print('pf(',init_size,'|',target_size,') = ',pf_vec[target_size], ' : Time = ', np.ceil(time.time()-start_time),' s')        

         # while the mean of last 4 values were < 0.97
        if np.mean(pf_vec[target_size-3:target_size]) > 0.98:
            break            

    print('total time = ',time.time()-overall_start_time)
    
    if only_rate:
        return np.exp((np.cumsum(np.log(pf_vec[2:target_size])))[-1])
    else:
        return np.exp((np.cumsum(np.log(pf_vec[2:target_size])))[-1]),L_configs,pf_vec,target_size
