# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:57:17 2021

@author: aoust

Special purpose file to test the Bukhsh instances.

Comparison with H. Godard's PhD thesis results => same total time limit (1h) and same accuracy (0.1\%)
"""



import instance
import numpy as np
import piecewiseRelaxer
import time
import sdpRelaxer
import EnhancedSdpRelaxer
from mipsParser import mipsResultParser
import BTSDPRelaxer
from progress.bar import Bar



#Instance parameters
lineconstraints = 'S'


#Main algo parameters
maxit = 1e4
BTtimeLimit = 1800
MILPtimeLimit = 3600
reltol = 1E-3 
ubcuts = True
with_lazy_random_sdp_cuts = False

######################### ###############################

def load_instance(name_instance):
    np.random.seed(10)
    instance_config = {"lineconstraints" : lineconstraints,  "cliques_strategy":"ASP"}
    Instance = instance.ACOPFinstance("data/benchmark_godard/{0}.m".format(name_instance),name_instance,instance_config)
    
    """handle the modified cases (bu Bukhsh et al)"""    
    if name_instance=='case9mod':
        Instance.Qmin = [-5.0]*Instance.gn
        Instance.Pload = 0.6 * np.array(Instance.Pload)
        Instance.Qload = 0.6 * np.array(Instance.Qload)
    
    if name_instance == 'case22loop':
        Instance.Pload = 2.15 * np.array(Instance.Pload)
        Instance.Qload = 2.15 * np.array(Instance.Qload)
        
    if name_instance == 'case39mod1':
        Instance.Pload = 0.5 * np.array(Instance.Pload)
        Instance.Qload = 0.5 * np.array(Instance.Qload)
        Instance.Vmin = 0.95 * np.ones(Instance.n)
        Instance.Vmax = 1.05 * np.ones(Instance.n)
        
    if name_instance == 'case39mod2':
        Instance.Pload = 0.5 * np.array(Instance.Pload)
        Instance.Qload = 0.5 * np.array(Instance.Qload)
        Instance.Vmin = 0.95 * np.ones(Instance.n)
        Instance.Vmax = 1.05 * np.ones(Instance.n)
        Instance.quadcost = [0.0]*Instance.gn
        
    if name_instance == 'case118mod':
        Instance.Qmin = 7 * np.array(Instance.Qmin)
        Instance.Qmax = 7 * np.array(Instance.Qmax)
        Instance.Pmax = 7 * np.array(Instance.Pmax)

    if name_instance =='case300mod':
        Instance.Qmin = [-5.0]*Instance.gn
        Instance.Pload = 0.6 * np.array(Instance.Pload)
        Instance.Qload = 0.6 * np.array(Instance.Qload)
        for i in range(Instance.n):
            if Instance.Pload[i]<0:
                Instance.Pload[i]=0
                Instance.Qload[i] = 0    
    
    return Instance

def compute_sdp_cuts(I):
    B2 = EnhancedSdpRelaxer.EnhancedSdpRelaxer(I)
    value,X1,X2 = B2.computeDuals()
    return value,X1,X2

def load_instance_and_bound_tightening(name_instance):
    I = load_instance(name_instance)
    if lineconstraints=='I':
        folder = 'data/mips_outputs_lc'
    else:
        folder = 'data/mips_outputs_S'
    localOptParser = mipsResultParser(folder,name_instance,I.baseMVA)
    
    
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
        bar = Bar('Angle tightening', max=len(I.SymEdgesNoDiag))
        for (i,j) in I.SymEdgesNoDiag:
            if deadline<time.time():
                break
            if (i<j) and I.ThetaMaxByEdge[(i,j)]<=np.pi/2 and I.ThetaMinByEdge[(i,j)]>=-0.5*np.pi:          
                idx_clique = I.SymEdgesNoDiag_to_clique[(i,j)]
                bound_lower = B.computeBTminAngle(obj,idx_clique,i,j)
                if (bound_lower>np.imag(localOptParser.V[i]*np.conj(localOptParser.V[j]))):
                      print('Angle tightening Alert !! {0}'.format(bound_lower-np.imag(localOptParser.V[i]*np.conj(localOptParser.V[j]))))
                # print(bound_lower)
                # print(counter,'lower',np.arcsin(min(bound_lower/(I.Vmin[i]*I.Vmin[j]),bound_lower/(I.Vmax[i]*I.Vmax[j]))),I.ThetaMinByEdge[(i,j)])
                I.ThetaMinByEdge[(i,j)] = B.ThetaMinByEdge[(i,j)] = max(I.ThetaMinByEdge[(i,j)],np.arcsin(min(bound_lower/(I.Vmin[i]*I.Vmin[j]),bound_lower/(I.Vmax[i]*I.Vmax[j]))))
                
                upper_bound = B.computeBTmaxAngle(obj,idx_clique,i,j)
                #print(counter,'upper',np.arcsin(max(upper_bound/(I.Vmin[i]*I.Vmin[j]),upper_bound/(I.Vmax[i]*I.Vmax[j]))),I.ThetaMaxByEdge[(i,j)])
                if (upper_bound<np.imag(localOptParser.V[i]*np.conj(localOptParser.V[j]))):
                      print('Angle tightening Alert !! {0}'.format(-upper_bound+np.imag(localOptParser.V[i]*np.conj(localOptParser.V[j]))))
                I.ThetaMaxByEdge[(i,j)] = B.ThetaMaxByEdge[(i,j)] = min(I.ThetaMaxByEdge[(i,j)],np.arcsin(max(upper_bound/(I.Vmin[i]*I.Vmin[j]),upper_bound/(I.Vmax[i]*I.Vmax[j]))))
            bar.next()
        bar.finish()
        bar = Bar('Magnitude tightening', max=I.n)
        for i in range(I.n):
            if deadline<time.time():
                break
            idx_clique = I.globalBusIdx_to_cliques[i][0]
            lower_bound = B.computeBTminMag(obj,idx_clique,i)
            if (np.sqrt(lower_bound)>abs(localOptParser.V[i])):
                      print('Magnitude tightening Alert !! {0}'.format(lower_bound-abs(localOptParser.V[i])))
                
            #print(counter,'lower_mag',np.sqrt(lower_bound),I.Vmin[i])
            I.Vmin[i] = B.Vmin[i] = min(max(I.Vmin[i],np.sqrt(lower_bound)),I.Vmax[i])
            upper_bound = B.computeBTmaxMag(obj,idx_clique,i)
    
            #print(counter,'upper_mag',np.sqrt(upper_bound),I.Vmax[i])
            I.Vmax[i] = B.Vmax[i] = max(min(I.Vmax[i],np.sqrt(upper_bound)),I.Vmin[i])
            if (np.sqrt(upper_bound)<abs(localOptParser.V[i])):
                      print('Magnitude tightening Alert !! {0}'.format(-upper_bound+abs(localOptParser.V[i])))
        
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

def basicsdp_relaxation_value(I,ub):
    B = sdpRelaxer.sdpRelaxer(I)
    val = B.computeSDPvalue()
    if lineconstraints=='I':
        with open('output_I/'+name_instance+'_sdp_val.txt','w') as f:
            f.write('SDP value = {0} \n'.format(val))
            f.write('SDP gap = {0} \n'.format(abs(val-ub)/(ub)))
            f.close()
    else:
        with open('output_S/'+name_instance+'_sdp_val.txt','w') as f:
            f.write('SDP value = {0} \n'.format(val))
            f.write('SDP gap = {0} \n'.format(abs(val-ub)/(ub)))
            f.close()  
    return val

def global_algo(name_instance):
    print('############################################')
    print("Start loading instance " +name_instance)
    t0 = time.time()
    I = load_instance(name_instance)
    if lineconstraints=='I':
        folder = 'data/mips_outputs_lc'
    else:
        folder = 'data/mips_outputs_S'
    localOptParser = mipsResultParser(folder,name_instance,I.baseMVA)
    valSDP,X1,X2 = compute_sdp_cuts(I)
    if abs(localOptParser.value - valSDP)/localOptParser.value < reltol:
        value = valSDP
        timeBTSDP = time.time() - t0
        timeMILP = 0
        bestLB = valSDP
        status = 'Strengthened SDP relaxation has no gap '
    else:
        if I.n<300:
            I = load_instance_and_bound_tightening(name_instance)    
        value,X1,X2 = compute_sdp_cuts(I)
        timeBTSDP = time.time() - t0
        
        t1= time.time()
        R=piecewiseRelaxer.piecewiseRelaxer(I,{'reinforcement':True},localOptParser)
        R.build_model()
        R.add_sdp_duals_W(X1)
        R.add_sdp_duals_R(X2)
        status = R.solve(MILPtimeLimit-timeBTSDP,maxit,reltol,ubcuts,with_lazy_random_sdp_cuts)
        timeMILP = time.time()-t1
        bestLB = R.bestLB
    print(status)
    if lineconstraints=='I':
        with open('output_I/'+name_instance+'_global_status.txt','w') as f:
            f.write('Time BT-SDP = {0} \n'.format(timeBTSDP))
            f.write('BT-SDP value = {0} \n'.format(value))
            f.write('Time MILP = {0}\n'.format(timeMILP))
            f.write('Best-LB = {0}\n'.format(bestLB))
            f.write(status)
            f.close()
    else:
        with open('output_S/'+name_instance+'_global_status.txt','w') as f:
            f.write('Time BT-SDP = {0}\n'.format(timeBTSDP))
            f.write('BT-SDP value = {0} \n'.format(value))
            f.write('Time MILP = {0} \n'.format(timeMILP))
            f.write('Best-LB = {0}\n'.format(bestLB))
            f.write(status)
            f.close()
        
        
instances = [
       'WB2',
       'WB3',
       'WB5',
   'WB5mod',
   'case9mod',
  'case22loop',
 'case30loopmod',
'case39mod1',
 'case39mod2',
 'case118mod',
 'case5',
  'case6ww',
 'case9',
 'case14',
 'case30',
 'case39',
 'case57',
 'case89pegase',
 'case118',
   'case300',
 'case300mod'
 ]


for name_instance in instances:
    global_algo(name_instance.replace('.m',''))