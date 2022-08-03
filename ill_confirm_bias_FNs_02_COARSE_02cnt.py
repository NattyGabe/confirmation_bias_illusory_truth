#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import numba
from math import lgamma, floor
np.set_printoptions(suppress=True)



# define a function that generates random numbers nsteps many timesteps
def rand_nsteps(memstore0, memshare0, mem0, pulls0, payprobs0, numagents0, nsteps0, rng0):
    narms0 = len(payprobs0)
    payouts0 = rng0.binomial(pulls0, payprobs0, (nsteps0, numagents0, narms0))
    
    randunif00 = rng0.random((nsteps0, numagents0, numagents0, memshare0+1))
    
    perm00 = [np.arange(mem0)]
    perm00 = np.repeat(perm00, numagents0*nsteps0, axis=0)
    perm00 = rng0.permuted(perm00, axis=1)
    perm00 = np.reshape(perm00, [nsteps0, numagents0, mem0])
    
    store00 = [np.arange((numagents0*(memshare0+1)))]
    store00 = np.repeat(store00, numagents0*nsteps0, axis=0)
    store00 = rng0.permuted(store00, axis=1)
    store00 = (store00[:, :memstore0]).copy()
    if memshare0 != 0:
        store00 = np.reshape(store00, [nsteps0, numagents0, memstore0])
    else:
        store00 = np.zeros([nsteps0, numagents0, 1])
    return payouts0, randunif00, perm00, store00
    

# beta-binomial pmf that works with numba
@numba.jit(nogil=True, cache=True)
def bbpmfNG(vk, vn, va, vb):
    va1t = lgamma(vn+1)
    va2t = lgamma(vk+va)
    va3t = lgamma(vn-vk+vb)
    va4t = lgamma(va+vb)
    va1b = lgamma(vk+1)
    va2b = lgamma(vn-vk+1)
    va3b = lgamma(vn+va+vb)
    va4b = lgamma(va)
    va5b = lgamma(vb)
    
    vans = np.exp((va1t+va2t+va3t+va4t)-(va1b+va2b+va3b+va4b+va5b))
    return vans

# calculate a single agent's probability that Y,the arm thought to be worse, is better than X, the arm thought to be better
@numba.jit(nogil=True, cache=True)
def bbpmfCOARSEng(pulls6, aX, bX, aY, bY):
    
    sum6 = 0
    pmfYsums = np.zeros(pulls6+2)
    pmfYsums[0] = 1
    for idx60 in range(0, pulls6+1):
        pmfXi = bbpmfNG(idx60, pulls6, aX, bX)
        pmfYsums[idx60+1] = pmfYsums[idx60] - bbpmfNG(idx60, pulls6, aY, bY)
        sum6 = sum6 + (pmfXi*pmfYsums[idx60+1])
    return sum6

# build array that gives accept prob (WITH tollerance adjustment) of each agent for each arm
@numba.jit(nogil=True, cache=True)
def bbpmfCORarray(pulls7, params7, expect7, nag7, narms7, tol7):
    agpmfCORtols = np.zeros((nag7, narms7))
    for idx70 in numba.prange(0, nag7):
        for idx71 in range(0, narms7):
            arm7 = idx71
            arm7mod = (idx71+1)%2
            if expect7[idx70][arm7] > expect7[idx70][arm7mod]:
                agpmfCORtols[idx70][idx71] = 1
            else:
                aX7 = params7[idx70][arm7mod][0]
                bX7 = params7[idx70][arm7mod][1]
                aY7 = params7[idx70][arm7][0]
                bY7 = params7[idx70][arm7][1]
                agpmfCORtols[idx70][idx71] = (bbpmfCOARSEng(pulls7, aX7, bX7, aY7, bY7))**tol7
    return agpmfCORtols

# function that gets expected value for alpha beta parameters
@numba.jit(nogil=True, cache=True)
def abexpect(a1, b1):
    exp1 = (a1/(a1+b1))
    return exp1

# function that gets all expected values given parameters array
@numba.jit(nogil=True, cache=True)
def agentexp(param1):
    nag1 = len(param1)
    narms1 = len(param1[0])
    agexp1 = np.zeros((nag1, narms1))
    for idx10 in numba.prange(0, nag1):
        for idx11 in range(0, narms1):
            a11 = param1[idx10][idx11][0]
            b11 = param1[idx10][idx11][1]
            exp11 = abexpect(a11, b11)
            agexp1[idx10][idx11] = exp11
    return agexp1


# from payouts and expectations, get evidence
@numba.jit(nogil=True, cache=True)
def evid(pay2, exp2, idxa, idxb):
    nag2 = len(pay2)
    evid2 = np.zeros(nag2)
    evidarms2 = np.zeros(nag2)
    evidplay2 = np.zeros(nag2) +idxa +idxb
    for idx20 in numba.prange(0, nag2):
        evididx = np.argmax(exp2[idx20])
        evidarms2[idx20] = evididx
        evid2[idx20] = pay2[idx20][evididx]
        evid2[idx20] = pay2[idx20][evididx]
    return evid2, evidarms2, evidplay2


# calculate the probability each agent assigns to each piece of evidence given their priors if they are connected
# and why not throw in the tollerance parameter here as well
# @numba.jit
# def agentstolpmfs(memshare3, pulls3, nag3, params3, memevid3, memevidarms3, con3, tol3):
#     tolpmfs3 = np.zeros((nag3, nag3, memshare3+1))
#     for idx30 in range(0, nag3):
#         for idx31 in range(0, nag3):
#             if idx30 == idx31:
#                 tolpmfs3[idx30][idx31][0] = 1
#             elif con3[idx30][idx31] == 0:
#                 tolpmfs3[idx30][idx31][0] = 0
#             else:
#                 for idx32 in range(0, memshare3+1):
#                     if memevid3[idx32][idx31] == -1:
#                         tolpmfs3[idx30][idx31][idx32] = 0
#                     else:
#                         sharearm3 = floor(memevidarms3[idx32][idx31])
#                         aga3 = params3[idx30][sharearm3][0]
#                         agb3 = params3[idx30][sharearm3][1]
#                         shareevid3 = memevid3[idx32][idx31]
#                         shareprob3 = bbpmfNG(shareevid3, pulls3, aga3, agb3)
#                         shareprob3 = shareprob3**tol3
#                         tolpmfs3[idx30][idx31][idx32] = shareprob3
#     return tolpmfs3

@numba.jit(nogil=True, cache=True)
def agentstolpmfsCOARSE(agpmfCORtols3, memshare3, pulls3, nag3, params3, memevid3, memevidarms3, con3, tol3):
    tolpmfs3 = np.zeros((nag3, nag3, memshare3+1))
    for idx30 in numba.prange(0, nag3):
        for idx31 in range(0, nag3):
            if idx30 == idx31:
                tolpmfs3[idx30][idx31][0] = 1
            elif con3[idx30][idx31] == 0:
                tolpmfs3[idx30][idx31][0] = 0
            else:
                for idx32 in range(0, memshare3+1):
                    if memevid3[idx32][idx31] == -1:
                        tolpmfs3[idx30][idx31][idx32] = 0
                    else:
                        sharearm3 = floor(memevidarms3[idx32][idx31])
                        shareprob3 = agpmfCORtols3[idx30][sharearm3]
#                         aga3 = params3[idx30][sharearm3][0]
#                         agb3 = params3[idx30][sharearm3][1]
#                         shareevid3 = memevid3[idx32][idx31]
#                         shareprob3 = bbpmfNG(shareevid3, pulls3, aga3, agb3)
#                         shareprob3 = shareprob3**tol3
                        tolpmfs3[idx30][idx31][idx32] = shareprob3
    return tolpmfs3


# given random uniforms and tolpmfs determine whether shared evidence is accepted
# also given params and evidence update params
@numba.jit(nogil=True, cache=True)
def paramsupdate(memplay4, memevidplay4, perms4, memsuc4, memarms4, store4, memstore4, memshare4, pulls4, nag4, randunif4, tolpmfs4, memevid4, memevidarms4, params4):
    for idx40 in numba.prange(0, nag4):
        storecount = -1
        for idx41 in range(0, nag4):
            for idx42 in range(0, memshare4+1):
                if randunif4[idx40][idx41][idx42] < tolpmfs4[idx40][idx41][idx42]:
                    memevididx4 = floor(memevidarms4[idx42][idx41])
                    params4[idx40][memevididx4][0] += memevid4[idx42][idx41]
                    params4[idx40][memevididx4][1] += (pulls4- memevid4[idx42][idx41])
                    storeidx = idx42+idx42*idx41
                    if storeidx in store4[idx40]:
                        replaceidx = floor(perms4[idx40][storecount])
                        memsuc4[idx40][replaceidx] = memevid4[idx42][idx41]
                        memarms4[idx40][replaceidx] = memevidarms4[idx42][idx41]
                        memplay4[idx40][replaceidx] = memevidplay4[idx42][idx41]
                        storecount -=1
                        
    return params4, memsuc4, memarms4, memplay4

# function to take evid, memory, permutations, memshare and return evidence array
@numba.jit(nogil=True, cache=True)
def memevidence(evidplay5, memplay5, evid5, evidarms5, memsuc5, memarms5, perms5, nag5, memshare5):
    memevid5 = np.zeros((memshare5+1, nag5))-1
    memevidarms5 = np.zeros((memshare5+1, nag5))
    memevid5[0] = evid5
    memevidarms5[0] = evidarms5
    memevidplay5 = np.zeros((memshare5+1, nag5))
    memevidplay5[0] = evidplay5
    for idx50 in range(1, memshare5+1):   
        for idx51 in range(0, nag5):
            memidx = perms5[idx51][idx50-1]
            memevid5[idx50][idx51] = memsuc5[idx51][memidx]
            memevidarms5[idx50][idx51] = memarms5[idx51][memidx]
            memevidplay5[idx50][idx51] = memplay5[idx51][memidx]
    return memevid5, memevidarms5, memevidplay5

# okay lets go through everything for nsteps many timesteps
@numba.jit(nogil=True, cache=True)
def params_nsteps(memplay9, idxf9, narms9, store9, perms9, memsuc9, memarms9, memshare9, memstore9, pulls9, nag9, con9, randunif90, pay9, params9, tol9, nsteps9):
    for idx90 in range(0, nsteps9):
        expect9 = agentexp(params9)
        evid9, evidarms9, evidplay9 = evid(pay9[idx90], expect9, idxf9, idx90)
        memevid9, memevidarms9, memevidplay9 = memevidence(evidplay9, memplay9, evid9, evidarms9, memsuc9, memarms9, perms9[idx90], nag9, memshare9)
        agpmfCORtols9 = bbpmfCORarray(pulls9, params9, expect9, nag9, narms9, tol9)
        tolpmfs9 = agentstolpmfsCOARSE(agpmfCORtols9, memshare9, pulls9, nag9, params9, memevid9, memevidarms9, con9, tol9)
        params9, memsuc9, memarms9, memplay9 = paramsupdate(memplay9, memevidplay9, perms9[idx90], memsuc9, memarms9, store9[idx90], memstore9, memshare9, pulls9, nag9, randunif90[idx90], tolpmfs9, memevid9, memevidarms9, params9)
    return params9, evidarms9, memsuc9, memarms9, memplay9

# convert memplay into the desired formatt
@numba.jit
def persistence(idxf8, mem_depth8, mem_snap8, memplay8, mem_ds8, md_len8, ms_len8, nagf8, memf8):
    persis = np.zeros(md_len8)
    for idx80 in range(0, md_len8):
        cut8 = mem_depth8[idx80]
        cut81 = idxf8 - cut8
        check8 = 0.
        for idx81 in range(0, nagf8):
            for idx82 in range(0, memf8):
                if memplay8[idx81][idx82] < cut81:
                    check8 += 1.
        persis[idx80] = check8
        mem_ds8[0][idx80] += check8
        
    for idx83 in range(0, ms_len8-1):
        if mem_snap8[idx83] == idxf8:
            mem_ds8[idx83+1] = persis
            
    return mem_ds8

# putting it all together
def confirm_bias_full(mem_snapf, mem_depthf, erpf, memf, memsharef, memstoref, networktypef, maxalphabetaf, pullsf, nagf, narmsf, tolf, payprobsf, nstepsf, tstepsf, rngs, n0):
    
    rngf = rngs[n0]
    
    if networktypef == 'cycle':
        connections = np.identity(nagf,dtype=int) + np.eye(nagf,k=1,dtype=int) + np.eye(nagf,k=-1,dtype=int)
        connections[0,-1] = 1
        connections[-1,0] = 1

    elif networktypef == 'wheel':
        connections = np.identity(nagf,dtype=int) + np.eye(nagf,k=1,dtype=int) + np.eye(nagf,k=-1,dtype=int)
        connections[0,:] = 1
        connections[:,0] = 1
        connections[1,-1] = 1
        connections[-1,1] = 1

    elif networktypef == 'complete':
        connections = np.ones([nagf,nagf],dtype=int)

    elif networktypef == 'clique':
        connections = np.zeros([nagf,nagf],dtype=int)
        clique1 = connections[:-nagf//2,:-nagf//2]
        clique2 = connections[-nagf//2:,-nagf//2:]
        clique1[:,:] = 1
        clique2[:,:] = 1
        connections[-nagf//2-1,-nagf//2] = 1
        connections[-nagf//2,nagf//2-1] = 1 
        
    elif networktypef == 'errandom':
        erprob = erpf
        half_erprob = erprob/2
        connections_count = 0
        while connections_count < 1:

            connections = np.identity(nagf,dtype=int)
            randcon = np.random.choice(2, nagf**2, p=[1-half_erprob, half_erprob])
            randcon = np.reshape(randcon, [nagf, nagf])
            connections = connections + randcon + np.transpose(randcon)
            connections[connections > 1] = 1

            #check connectedness
            adj_check = np.linalg.matrix_power(connections, nagf)
            adj_check[adj_check > 1] = 1
            adj_check = np.prod(adj_check, axis = 1)
            adj_check = np.sum(adj_check)


            if adj_check > 0:
                connections_count = connections_count +1
    
    paramsf = rngf.random((nagf, narmsf, 2))*(maxalphabetaf-1)+1.00001
    # array to store memory of number of successes
    memsucf = np.zeros([nagf, memf])-1
    # array to store whch arm successes were from
    memarmsf = np.zeros([nagf, memf])
    # array to store on which play the results were generated
    memplayf = np.ones([nagf, memf]) +tstepsf
    
    # mem_depth is going to be how many pieces of data in memory are from plays further back than the values in mem_depth
    # mem_snap is the plays where we are going to take a snapshot of mem_dept
    md_len = len(mem_depthf)
    ms_len = len(mem_snapf) +1
    mem_ds = np.zeros([ms_len, md_len]) 
    
    for idxf in range(0, tstepsf, nstepsf):
        payf, randunif0f, permsf, storef = rand_nsteps(memstoref, memsharef, memf, pullsf, payprobsf, nagf, nstepsf, rngf)
        paramsf, evidarmsf, memsucf, memarmsf, memplayf = params_nsteps(memplayf, idxf, narmsf, storef, permsf, memsucf, memarmsf, memsharef, memstoref, pullsf, nagf, connections, randunif0f, payf, paramsf, tolf, nstepsf)
        mem_ds = persistence(idxf, mem_depthf, mem_snapf, memplayf, mem_ds, md_len, ms_len, nagf, memf)
    
    if np.sum(evidarmsf) == nagf:
        success = 1
        consensus = 1
    elif np.sum(evidarmsf) == 0:
        success = 0
        consensus = 1
    else:
        success = 0
        consensus = 0
        
    mem_ds[0] = mem_ds[0]/(tstepsf/nstepsf)
    
    return [success, consensus, mem_ds, n0]

