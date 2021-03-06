# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 20:11:57 2022

@author: aoust
"""

import os


import instance
import numpy as np
from testing.gbBenchmark import GurobiACOPFmodel
import time

import EnhancedSdpRelaxer
import BTSDPRelaxer
from localSolver import  localACOPFsolver

from progress.bar import Bar

#Instance parameters
lineconstraints = 'S'

#Main algo parameters
BTtimeLimit = 36000
MILPtimeLimit = 5*3600
reltol = 1E-4
ubcuts = True
with_lazy_random_sdp_cuts = False

######################### ###############################

def load_instance(name_instance):
    np.random.seed(10)
    instance_config = {"lineconstraints" : lineconstraints,  "cliques_strategy":"ASP"}
    Instance = instance.ACOPFinstance("data/pglib-opf/{0}.m".format(name_instance),name_instance,instance_config)
    return Instance

# def compute_sdp_cuts(I):
#     B2 = EnhancedSdpRelaxer.EnhancedSdpRelaxer(I)
#     value,X1,X2 = B2.computeDuals()
#     return value,X1,X2

def load_instance_and_bound_tightening(name_instance):
    I = load_instance(name_instance)
    if lineconstraints=='I':
        folder = 'mips_outputs_lc'
    else:
        folder = 'mips_outputs_S'
    localOptParser = localACOPFsolver(I)
    ########################## Checking #########################################
    
    if I.n<=100:
        niter=3
    else:
        niter=1
    deadline = BTtimeLimit + time.time()
    print('FBBT/OBBT started')
    for counter in range(niter):
        print('FBBT/OBBT Round {0}/{1}'.format(counter+1, niter))
        if deadline<time.time():
            break
        B = BTSDPRelaxer.BTSDPRelaxer(I)
        obj = localOptParser.value
        B2 = BTSDPRelaxer.BTSDPRelaxer(I)
        bar = Bar('Angle tightening', max=len(I.SymEdgesNoDiag))
        for (i,j) in I.SymEdgesNoDiag:
            if deadline<time.time():
                break
            if (i<j) and I.ThetaMaxByEdge[(i,j)]<=np.pi/2 and I.ThetaMinByEdge[(i,j)]>=-0.5*np.pi:
                idx_clique = I.SymEdgesNoDiag_to_clique[(i,j)]
                bound_lower = B2.computeBTminAngle(obj,idx_clique,i,j)
                if (bound_lower>np.imag(localOptParser.V[i]*np.conj(localOptParser.V[j]))):
                      print('Alert !! {0}'.format(bound_lower-np.imag(localOptParser.V[i]*np.conj(localOptParser.V[j]))))
                # print(bound_lower)
                # print(counter,'lower',np.arcsin(min(bound_lower/(I.Vmin[i]*I.Vmin[j]),bound_lower/(I.Vmax[i]*I.Vmax[j]))),I.ThetaMinByEdge[(i,j)])
                I.ThetaMinByEdge[(i,j)] = B2.ThetaMinByEdge[(i,j)] = max(I.ThetaMinByEdge[(i,j)],np.arcsin(min(bound_lower/(I.Vmin[i]*I.Vmin[j]),bound_lower/(I.Vmax[i]*I.Vmax[j]))))
                
                upper_bound = B2.computeBTmaxAngle(obj,idx_clique,i,j)
                #print(counter,'upper',np.arcsin(max(upper_bound/(I.Vmin[i]*I.Vmin[j]),upper_bound/(I.Vmax[i]*I.Vmax[j]))),I.ThetaMaxByEdge[(i,j)])
                if (upper_bound<np.imag(localOptParser.V[i]*np.conj(localOptParser.V[j]))):
                      print('Alert !! {0}'.format(-upper_bound+np.imag(localOptParser.V[i]*np.conj(localOptParser.V[j]))))
                I.ThetaMaxByEdge[(i,j)] = B2.ThetaMaxByEdge[(i,j)] = min(I.ThetaMaxByEdge[(i,j)],np.arcsin(max(upper_bound/(I.Vmin[i]*I.Vmin[j]),upper_bound/(I.Vmax[i]*I.Vmax[j]))))
            bar.next()
        bar.finish()
        bar = Bar('Magnitude tightening', max=I.n)
        for i in range(I.n):
            if deadline<time.time():
                break
            idx_clique = I.globalBusIdx_to_cliques[i][0]
            lower_bound = B.computeBTminMag(obj,idx_clique,i)
            if (np.sqrt(lower_bound)>abs(localOptParser.V[i])):
                      print('Alert !! {0}'.format(lower_bound-abs(localOptParser.V[i])))
                
            #print(counter,'lower_mag',np.sqrt(lower_bound),I.Vmin[i])
            I.Vmin[i] = B.Vmin[i] = max(I.Vmin[i],np.sqrt(lower_bound))
            upper_bound = B.computeBTmaxMag(obj,idx_clique,i)
    
            #print(counter,'upper_mag',np.sqrt(upper_bound),I.Vmax[i])
            I.Vmax[i] = B.Vmax[i] = min(I.Vmax[i],np.sqrt(upper_bound))
            if (np.sqrt(upper_bound)<abs(localOptParser.V[i])):
                      print('Alert !! {0}'.format(-upper_bound+abs(localOptParser.V[i])))
        
            bar.next()
        bar.finish()
        for i,j in I.SymEdgesNoDiag:
            I.ThetaMinByEdge[(i,j)] = max(I.ThetaMinByEdge[(i,j)], -I.ThetaMaxByEdge[(j,i)])
            I.ThetaMaxByEdge[(i,j)] = min(I.ThetaMaxByEdge[(i,j)], -I.ThetaMinByEdge[(j,i)])
        
        for idx_clique in range(len(I.cliques)):
            I.FloydWarshallOnClique(idx_clique)
        
        for i,j in I.SymEdgesNoDiag:
            I.ThetaMinByEdge[(i,j)] = max(I.ThetaMinByEdge[(i,j)], -I.ThetaMaxByEdge[(j,i)])
            I.ThetaMaxByEdge[(i,j)] = min(I.ThetaMaxByEdge[(i,j)], -I.ThetaMinByEdge[(j,i)])
    
    print('FBBT/OBBT ended')
    return I


def global_algo(name_instance):
    print('############################################')
    print("Start loading instance " +name_instance)
    t0 = time.time()
    I = load_instance(name_instance)
    if lineconstraints=='I':
        folder = 'mips_outputs_lc'
    else:
        folder = 'mips_outputs_S'
    localOptParser = localACOPFsolver(I)
    localOptParser.solve()
    # I = load_instance_and_bound_tightening(name_instance)
    # ########################## Checking #########################################
    # for cl in I.cliques:
    #     for index_bus_b in cl:
    #         for index_bus_a in cl:
    #             if index_bus_b!=index_bus_a:
    #                 assert(I.ThetaMinByEdge[(index_bus_b,index_bus_a)]<=1e-5+np.pi*(localOptParser.VA[index_bus_b]-localOptParser.VA[index_bus_a])/180)
    #                 assert(I.ThetaMaxByEdge[(index_bus_b,index_bus_a)]+1e-5>=np.pi*(localOptParser.VA[index_bus_b]-localOptParser.VA[index_bus_a])/180)
    # localOptParser.test_validity(I)

    # value,X1,X2 = compute_sdp_cuts(I)
    # timeBTSDP = time.time() - t0
    
    # t1= time.time()
    
    
    
    
    R=GurobiACOPFmodel(I,{'reinforcement':True},localOptParser)
    R.build_model()
    #R.add_sdp_duals_W(X1)
    R.solve(36000)
       
        
        
instances = [        # 'pglib_opf_case3_lmbd.m',
                      'pglib_opf_case5_pjm.m', 
                      # 'pglib_opf_case14_ieee.m',
    #                     'pglib_opf_case24_ieee_rts.m',
    #                   'pglib_opf_case30_as.m',
    #             'pglib_opf_case30_ieee.m',
    #               'pglib_opf_case39_epri.m',
    #               'pglib_opf_case57_ieee.m',
    #               'pglib_opf_case73_ieee_rts.m',
        'pglib_opf_case89_pegase.m',    
    #       'pglib_opf_case118_ieee.m',  
    #  'pglib_opf_case162_ieee_dtc.m', 
    # 'pglib_opf_case179_goc.m', 
    #    'pglib_opf_case200_activ.m', 
    # 'pglib_opf_case240_pserc.m', 
    # 'pglib_opf_case300_ieee.m', 
]


for name_instance in instances:
    global_algo(name_instance.replace('.m',''))