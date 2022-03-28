# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:08:56 2022

@author: aoust
"""

from localSolver import  localACOPFsolver
import instance
import numpy as np
import piecewiseRelaxer
import time
import sdpRelaxer
import EnhancedSdpRelaxer
import mutableBTSDPRelaxer
import pandas as pd
from progress.bar import Bar

#Paths
folder_dict = {'S':'output_S','I':'output_I',False:'output_no_lim'}
mips_folder_dict = {'S':'data/mips_outputs_S','I':'data/mips_outputs_lc',False:'data/mips_output_no_lim'}


def load_instance(name_instance,lineconstraints,case_datafolder):
    assert(lineconstraints in folder_dict)
    np.random.seed(10)
    instance_config = {"lineconstraints" : lineconstraints,  "cliques_strategy":"ASP"}
    Instance = instance.ACOPFinstance(case_datafolder+"/{0}.m".format(name_instance),name_instance,instance_config)
    
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
    sdprelaxer = EnhancedSdpRelaxer.EnhancedSdpRelaxer(I)
    value,X1,X2, PgenVal, LVal, ReWVal,ImWVal = sdprelaxer.computeDuals()
    return value,X1,X2, PgenVal, LVal, ReWVal,ImWVal

def bound_tightening(I,localOptParser,BTtimeLimit,reltol):
    assert(localOptParser.success ==1)
    niter = 4
    deadline = BTtimeLimit + time.time()
    log_sdp_values = []
    print('FBBT/OBBT started')
    for counter in range(niter):
        tol = 1e-6
        print('FBBT/OBBT Round {0}/{1}'.format(counter+1, niter))
        if deadline<time.time():
            break
        obj = localOptParser.value + 1e-5
        Bound_tightener = mutableBTSDPRelaxer.BTSDPRelaxer(I)
        Bound_tightener.buildModel(obj)
        bar = Bar('Angle tightening', max=len(I.SymEdgesNoDiag))
        for (i,j) in I.SymEdgesNoDiag:
            assert(I.ThetaMinByEdge[(i,j)]== Bound_tightener.ThetaMinByEdge[(i,j)])
            if deadline<time.time():
                break
            if (i<j) and I.ThetaMaxByEdge[(i,j)]<=np.pi/2 and I.ThetaMinByEdge[(i,j)]>=-0.5*np.pi:
                idx_clique = I.SymEdgesNoDiag_to_clique[(i,j)]
                bound_lower = Bound_tightener.computeBTminAngle(idx_clique,i,j)
                if (bound_lower>tol+np.imag(localOptParser.V[i]*np.conj(localOptParser.V[j]))):
                     print('Angle alert !! {0}'.format(bound_lower-np.imag(localOptParser.V[i]*np.conj(localOptParser.V[j]))))
                newThetamin =  max(I.ThetaMinByEdge[(i,j)],np.arcsin(min(bound_lower/(I.Vmin[i]*I.Vmin[j]),bound_lower/(I.Vmax[i]*I.Vmax[j]))))
                I.ThetaMinByEdge[(i,j)]= Bound_tightener.ThetaMinByEdge[(i,j)] = newThetamin
                Bound_tightener.update_angle(idx_clique,i,j,newThetamin,I.ThetaMaxByEdge[(i,j)])
                
                
                upper_bound = Bound_tightener.computeBTmaxAngle(idx_clique,i,j)
                if (upper_bound+tol<np.imag(localOptParser.V[i]*np.conj(localOptParser.V[j]))):
                     print('Angle alert !! {0}'.format(-upper_bound+np.imag(localOptParser.V[i]*np.conj(localOptParser.V[j]))))
                newThetaMax = min(I.ThetaMaxByEdge[(i,j)],np.arcsin(max(upper_bound/(I.Vmin[i]*I.Vmin[j]),upper_bound/(I.Vmax[i]*I.Vmax[j]))))
                I.ThetaMaxByEdge[(i,j)] = Bound_tightener.ThetaMaxByEdge[(i,j)] = newThetaMax
                Bound_tightener.update_angle(idx_clique,i,j,I.ThetaMinByEdge[(i,j)],newThetaMax)
            bar.next()
        bar.finish()
        valSDP,X1,X2, PgenVal, LVal, ReWVal,ImWVal = compute_sdp_cuts(I)
        log_sdp_values.append(valSDP)
        if abs(localOptParser.value - valSDP)/localOptParser.value < reltol:
            return I, log_sdp_values
        bar = Bar('Magnitude tightening', max=I.n)
        assert(I.Vmin[i] == Bound_tightener.Vmin[i])
        for i in range(I.n):
            if deadline<time.time():
                break
            idx_clique = I.globalBusIdx_to_cliques[i][0]
            lower_bound = Bound_tightener.computeBTminMag(idx_clique,i)
            if (np.sqrt(lower_bound)>tol+abs(localOptParser.V[i])):
                     print('LB Magnitude alert !! {0}'.format(np.sqrt(lower_bound)-abs(localOptParser.V[i])))
            newVmin = max(I.Vmin[i],np.sqrt(lower_bound))
            I.Vmin[i] = Bound_tightener.Vmin[i] = newVmin
            Bound_tightener.updateV(newVmin,Bound_tightener.Vmax[i],i)
            
            upper_bound = Bound_tightener.computeBTmaxMag(idx_clique,i)
            if (np.sqrt(upper_bound)+tol<abs(localOptParser.V[i])):
                     print('UB Magnitude alert !! {0}'.format(-np.sqrt(upper_bound)+abs(localOptParser.V[i])))
            newVmax = min(I.Vmax[i],np.sqrt(upper_bound))
            I.Vmax[i] = Bound_tightener.Vmax[i] = newVmax
            Bound_tightener.updateV(Bound_tightener.Vmin[i],newVmax,i)
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
            
        Bound_tightener.ThetaMinByEdge, Bound_tightener.ThetaMaxByEdge = I.ThetaMinByEdge, I.ThetaMaxByEdge
        for (i,j) in I.SymEdgesNoDiag:
            idx_clique = I.SymEdgesNoDiag_to_clique[(i,j)]
            Bound_tightener.update_angle(idx_clique,i,j,Bound_tightener.ThetaMinByEdge[(i,j)],Bound_tightener.ThetaMaxByEdge[(i,j)])
            
        valSDP,X1,X2, PgenVal, LVal, ReWVal,ImWVal = compute_sdp_cuts(I)
        log_sdp_values.append(valSDP)
        if abs(localOptParser.value - valSDP)/localOptParser.value < reltol:
            return I, log_sdp_values
    print('FBBT/OBBT ended')
    return I, log_sdp_values

def basicsdp_relaxation_value(name_instance,I,ub):
    B = sdpRelaxer.sdpRelaxer(I)
    lineconstraints = I.config['lineconstraints']
    val = B.computeSDPvalue()
    with open(folder_dict[lineconstraints]+'/'+name_instance+'_sdp_val.txt','w') as f:
        f.write('SDP value = {0} \n'.format(val))
        f.write('SDP gap = {0} \n'.format(abs(val-ub)/(ub)))
        f.close()
    return val

def saveBTbound_evolution(name_instance,lineconstraints,log_sdp_values):
    if len(log_sdp_values):
        df = pd.DataFrame()
        df['LB'] = log_sdp_values
        df.to_csv(folder_dict[lineconstraints]+'/'+name_instance+'_BT_progress.csv')
        

def global_algo(name_instance,lineconstraints,case_datafolder,BTtimeLimit,MILPtimeLimit,reltol):
    maxit = 1e5
    ubcuts = True
    with_lazy_random_sdp_cuts = False
    print('############################################')
    print("Start loading instance " +name_instance)
    t0 = time.time()
    I = load_instance(name_instance,lineconstraints,case_datafolder)
    lineconstraints = I.config['lineconstraints']
    localOptParser = localACOPFsolver(I)
    localOptParser.solve()
    valSDP,X1,X2, PgenVal, LVal, ReWVal,ImWVal = compute_sdp_cuts(I)
    total_it_number = 0
    if abs(localOptParser.value - valSDP)/localOptParser.value < reltol:
        value = valSDP
        timeBTSDP = time.time() - t0
        timeMILP = 0
        bestLB = valSDP
        bestGap = abs(localOptParser.value - valSDP)/localOptParser.value
        status = 'Strengthened SDP relaxation has no gap '
    else:
        I,log_sdp_values = bound_tightening(I,localOptParser,BTtimeLimit,reltol)
        saveBTbound_evolution(name_instance,lineconstraints,log_sdp_values)
        value,X1,X2, PgenVal, LVal, ReWVal,ImWVal = compute_sdp_cuts(I)
        timeBTSDP = time.time() - t0
        print('Time BTSDP = {0}'.format(timeBTSDP))
        t1= time.time()
        R=piecewiseRelaxer.piecewiseRelaxer(I,{'reinforcement':True},localOptParser)
        R.build_model()
        R.add_sdp_duals_W(X1)
        R.add_sdp_duals_R(X2)
        R.add_quad_cuts(PgenVal, LVal, ReWVal,ImWVal)
        status = R.solve(MILPtimeLimit,maxit,reltol,ubcuts,with_lazy_random_sdp_cuts)
        timeMILP = time.time()-t1
        bestLB = R.bestLB
        bestGap = (R.UB - bestLB)/R.UB
        total_it_number = R.total_it_number
    print(status)

    with open(folder_dict[lineconstraints]+'/'+name_instance+'_global_status.txt','w') as f:
        f.write('Time BT-SDP = {0} \n'.format(timeBTSDP))
        f.write('BT-SDP value = {0} \n'.format(value))
        f.write('Time MILP = {0}\n'.format(timeMILP))
        f.write('Total iteration number = {0}\n'.format(total_it_number))
        f.write('Best-LB = {0}\n'.format(bestLB))
        f.write('Best-gap = {0}\n'.format(bestGap))
        f.write(status)
        f.close()
    
        