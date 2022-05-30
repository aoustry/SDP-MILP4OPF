# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 19:33:04 2022

@author: aoust
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import operator
import pandas as pd
import tools
import time

from numpy import linalg as LA



class GurobiACOPFmodel():
    
    def __init__(self, ACOPF, config, local_optimizer_results):
      
        """
              
        """
        print("------ Testing the solver Gurobi on the (S,W) formulation -----------")
        self.name = ACOPF.name
        self.config = config
        self.local_optimizer_results = local_optimizer_results
        
        #Sizes
        self.baseMVA = ACOPF.baseMVA
        self.n, self.m, self.gn = ACOPF.n, ACOPF.m, ACOPF.gn
        self.N = ACOPF.N
        #Generator quantities
        self.C = ACOPF.C
        self.offset = ACOPF.offset
        self.lincost = ACOPF.lincost
        self.quadcost = ACOPF.quadcost
        self.genlist = ACOPF.genlist
        assert(len(self.genlist)==self.gn)
        self.Pmin, self.Qmin, self.Pmax, self.Qmax = ACOPF.Pmin, ACOPF.Qmin, ACOPF.Pmax, ACOPF.Qmax
        self.inactive_generators = []
        
        #Bus quantities
        self.buslist = ACOPF.buslist
        self.buslistinv = ACOPF.buslistinv
        self.bus_to_gen = {}
        for idx in range(self.n):
            self.bus_to_gen[idx] = []
        for idx_gen,gen in enumerate(self.genlist):
            bus,index = self.genlist[idx_gen]
            index_bus =  self.buslistinv[bus]
            self.bus_to_gen[index_bus].append(idx_gen)
       
        for i in range(self.n):
            assert(self.buslistinv[self.buslist[i]]==i)
        self.busType = ACOPF.busType
        self.Vmin, self.Vmax = ACOPF.Vmin, ACOPF.Vmax
        self.radius = max(self.Vmax)
        self.A = ACOPF.A
        self.Pload, self.Qload = ACOPF.Pload, ACOPF.Qload
        
        #Lines quantities
        self.status = ACOPF.status
        self.cl = ACOPF.cl
        self.clinelist, self.clinelistinv = ACOPF.clinelist, ACOPF.clinelistinv
        self.Imax = ACOPF.Imax
        
        self.ThetaMinByEdge, self.ThetaMaxByEdge = ACOPF.ThetaMinByEdge, ACOPF.ThetaMaxByEdge
        
        #Construct cliques
        self.cliques_nbr = ACOPF.cliques_nbr
        self.cliques, self.ncliques = ACOPF.cliques, ACOPF.ncliques
        self.cliques_parent, self.cliques_intersection = ACOPF.cliques_parent, ACOPF.cliques_intersection 
        self.localBusIdx = ACOPF.localBusIdx
        self.SVM = ACOPF.SVM
                
        # #Construct m_cb matrices
        self.HM, self.ZM = ACOPF.HM, ACOPF.ZM
        self.config['lineconstraints'] = ACOPF.config['lineconstraints']
        assert( ACOPF.config['lineconstraints']=='I' or ACOPF.config['lineconstraints']=='S')
        if self.config['lineconstraints']=='I':
            self.Nf, self.Nt = ACOPF.Nf, ACOPF.Nt
        else:
            self.Yff, self.Yft, self.Ytf, self.Ytt = ACOPF.Yff, ACOPF.Yft, ACOPF.Ytf, ACOPF.Ytt
        
        #Build edges
        self.edges = set()
        self.edgesNoDiag = set()
        self.symedges = set()
        for cl in self.cliques:
            for i in cl:
                for j in cl:
                    if i<j:
                        self.edges.add((i,j))
                        self.edgesNoDiag.add((i,j))
                        self.symedges.add((i,j))
                        self.symedges.add((j,i))
        for i in range(self.n):
            self.edges.add((i,i)) 
            self.symedges.add((i,i))
                
        self.neighbors = {}
        for (b,a) in self.edgesNoDiag:
            if not(b in self.neighbors):
                self.neighbors[b] = []
            if not(a in self.neighbors):
                self.neighbors[a] = []
            assert(not(a in self.neighbors[b] ))
            self.neighbors[b].append(a)
            assert(not(b in self.neighbors[a] ))
            self.neighbors[a].append(b)
      

    def build_model(self):
        self.binaries = False
        self.mdl = gp.Model("Master problem")
        ######################################### Variables ######################################
        #Gen variables
        self.Pgen = self.mdl.addVars(self.gn,lb = self.Pmin, ub = self.Pmax,name = "Pgen")
        self.Qgen = self.mdl.addVars(self.gn,lb = self.Qmin, ub = self.Qmax,name = "Qgen")
        #Bus variables
        self.ReV = self.mdl.addVars(self.n,lb = [-v for v in self.Vmax], ub = self.Vmax,name = "ReV")
        self.ImV = self.mdl.addVars(self.n,lb = [-v for v in self.Vmax], ub = self.Vmax,name = "ImV")
        #Bus and edges variables
        self.ReW = self.mdl.addVars(self.symedges,lb = -self.radius**2, ub = self.radius**2,name = "ReW")
        self.ImW = self.mdl.addVars(self.symedges,lb = -self.radius**2, ub = self.radius**2,name = "ImW")
        ######################################### Objective function ################################################
        self.mdl.setObjective(self.offset + gp.quicksum((self.Pgen[i]*self.lincost[i] for i in range(self.gn))) + gp.quicksum(((self.Pgen[i]**2)*self.quadcost[i] for i in range(self.gn))) , GRB.MINIMIZE)
        ######################################### Linear constraints defining F ######################################
        self.mdl.addConstrs((self.Pgen[i]<= self.Pmax[i] for i in range(self.gn)))
        self.mdl.addConstrs((self.Pgen[i]>= self.Pmin[i] for i in range(self.gn)))
        self.mdl.addConstrs((self.Qgen[i]<= self.Qmax[i] for i in range(self.gn)))
        self.mdl.addConstrs((self.Qgen[i]>= self.Qmin[i] for i in range(self.gn)))
        self.mdl.addConstrs((self.ReW[b,a]==self.ReW[a,b] for (b,a) in self.edges))
        self.mdl.addConstrs((self.ImW[b,a]==-self.ImW[a,b] for (b,a) in self.edges))
        self.mdl.addConstrs((self.ReW[b,b]<=self.Vmax[b]**2 for b in range(self.n)))
        self.mdl.addConstrs((self.ReW[b,b]>=self.Vmin[b]**2 for b in range(self.n)))
        #Power conservation constraints
        for idx_bus in range(self.n):
            row, col = self.HM[idx_bus].nonzero()
            dicoHmbRe = {(row[aux],col[aux]):np.real(self.HM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            dicoHmbIm = {(row[aux],col[aux]):np.imag(self.HM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            self.mdl.addConstr(gp.quicksum((self.Pgen[idx_gen] for idx_gen in self.bus_to_gen[idx_bus]))==self.Pload[idx_bus]+gp.quicksum((self.ReW[i,j]*dicoHmbRe[i,j] for i,j in dicoHmbRe)) + gp.quicksum((self.ImW[i,j]*dicoHmbIm[i,j] for i,j in dicoHmbIm)))
            row, col = self.ZM[idx_bus].nonzero()
            dicoZmbRe = {(row[aux],col[aux]):np.real(1j*self.ZM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            dicoZmbIm = {(row[aux],col[aux]):np.imag(1j*self.ZM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            self.mdl.addConstr(gp.quicksum((self.Qgen[idx_gen] for idx_gen in self.bus_to_gen[idx_bus]))==self.Qload[idx_bus]+gp.quicksum((self.ReW[i,j]*dicoZmbRe[i,j] for i,j in dicoZmbRe)) + gp.quicksum((self.ImW[i,j]*dicoZmbIm[i,j] for i,j in dicoZmbIm)))
        if self.config['lineconstraints']=='I':
            for idx_line in range(self.cl):
                row, col = self.Nf[idx_line].nonzero()
                dicoNflineRe = {(row[aux],col[aux]):np.real(self.Nf[idx_line][row[aux],col[aux]]) for aux in range(len(row))}
                dicoNflineIm = {(row[aux],col[aux]):np.imag(self.Nf[idx_line][row[aux],col[aux]]) for aux in range(len(row))}
                self.mdl.addConstr(gp.quicksum((self.ReW[i,j]*dicoNflineRe[i,j] for i,j in dicoNflineRe)) + gp.quicksum((self.ImW[i,j]*dicoNflineIm[i,j] for i,j in dicoNflineIm)) <=self.Imax[idx_line]**2)
                row, col = self.Nt[idx_line].nonzero()
                dicoNtlineRe = {(row[aux],col[aux]):np.real(self.Nt[idx_line][row[aux],col[aux]]) for aux in range(len(row))}
                dicoNtlineIm = {(row[aux],col[aux]):np.imag(self.Nt[idx_line][row[aux],col[aux]]) for aux in range(len(row))}
                self.mdl.addConstr(gp.quicksum((self.ReW[i,j]*dicoNtlineRe[i,j] for i,j in dicoNtlineRe)) + gp.quicksum((self.ImW[i,j]*dicoNtlineIm[i,j] for i,j in dicoNtlineIm)) <=self.Imax[idx_line]**2)
        else:
            assert(self.config['lineconstraints']=='S')
            for idx_line,line in enumerate(self.clinelistinv):
                b,a,h = line
                index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
                ##Version MIQCP
                rex = np.real(self.Yff[line]) * self.ReW[index_bus_b,index_bus_b] + np.real(self.Yft[line]) * self.ReW[index_bus_b,index_bus_a] + np.imag(self.Yft[line])* self.ImW[index_bus_b,index_bus_a]
                imx = -np.imag(self.Yff[line]) * self.ReW[index_bus_b,index_bus_b] -np.imag(self.Yft[line])* self.ReW[index_bus_b,index_bus_a] + np.real(self.Yft[line]) * self.ImW[index_bus_b,index_bus_a]
                self.mdl.addConstr(rex**2+imx**2<=self.Imax[idx_line]**2)
                ##Version MILP
                
                #Switching indices to have (a,b,h) \in L
                aux = index_bus_b
                index_bus_b = index_bus_a
                index_bus_a = aux
                # ##Version MIQCP
                rex =np.real(self.Ytt[line]) * self.ReW[index_bus_b,index_bus_b] + np.real(self.Ytf[line]) * self.ReW[index_bus_b,index_bus_a] + np.imag(self.Ytf[line])* self.ImW[index_bus_b,index_bus_a]
                imx = -np.imag(self.Ytt[line])* self.ReW[index_bus_b,index_bus_b] -np.imag(self.Ytf[line])* self.ReW[index_bus_b,index_bus_a] + np.real(self.Ytf[line]) * self.ImW[index_bus_b,index_bus_a]
                self.mdl.addConstr(rex**2+imx**2<=self.Imax[idx_line]**2)
                ########################################## Discretization constraints #######################################################################   
 
        for b,a in self.edgesNoDiag:
            if self.ThetaMaxByEdge[b,a]- self.ThetaMinByEdge[b,a]<=np.pi:
                    halfdiff =  0.5*(self.ThetaMaxByEdge[b,a]- self.ThetaMinByEdge[b,a])
                    mean =  0.5*(self.ThetaMaxByEdge[b,a] + self.ThetaMinByEdge[b,a])
                    self.mdl.addConstr(-np.sin(self.ThetaMinByEdge[b,a])*self.ReW[b,a] + np.cos(self.ThetaMinByEdge[b,a]) * self.ImW[b,a] >=  0)
                    self.mdl.addConstr(-np.sin(self.ThetaMaxByEdge[b,a])*self.ReW[b,a] + np.cos(self.ThetaMaxByEdge[b,a]) * self.ImW[b,a] <=  0)
                    
        ##################################################################################################################################################################################################################################
        for b in range(self.n):
            self.mdl.addConstr(self.ReW[b,b]==self.ReV[b]**2+self.ImV[b]**2)
        for b,a in self.edgesNoDiag:
            self.mdl.addConstr(self.ReW[b,a]==self.ReV[b]*self.ReV[a]+self.ImV[b]*self.ImV[a])
            self.mdl.addConstr(self.ImW[b,a]==-self.ReV[b]*self.ImV[a]+self.ImV[b]*self.ReV[a])
            
    
    def solve(self,tl):
        
        self.mdl.setParam('FeasibilityTol',1e-9)
        self.mdl.setParam('NonConvex', 2)
        self.mdl.setParam('TimeLimit', tl)
        t1 = time.time()
        self.mdl.optimize()
        mastertime = time.time() - t1

    
    def add_sdp_cut(self, idx_clique, vector):
        cl = self.cliques[idx_clique]
        nc = len(cl)
        vector = vector.reshape((nc,1))
        M = vector.dot(np.conj(vector.T))
        dicoMRe = {(cl[i],cl[j]):np.real(M[i,j]) for i in range(nc) for j in range(nc)}
        dicoMIm = {(cl[i],cl[j]):np.imag(M[i,j]) for i in range(nc) for j in range(nc)}
        self.mdl.addConstr(gp.quicksum((self.ReW[i,j]*dicoMRe[i,j] for i,j in dicoMRe)) + gp.quicksum((self.ImW[i,j]*dicoMIm[i,j] for i,j in dicoMIm)) >=0)
        
        print('todo')
    def add_sdp_duals_W(self, X):
        assert(len(X)==self.cliques_nbr)
        for i in range(self.cliques_nbr):
            mat = 0.5*(X[i]+np.conj(X[i].T))
            s, U = LA.eigh(mat)
            for k in range(self.ncliques[i]):
                vector = U[:,k]
                vector = vector.reshape((self.ncliques[i],1))
                self.add_sdp_cut(i, vector)
    

