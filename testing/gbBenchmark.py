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
        print("------ Piecewise relaxation solver for the ACOPF -----------")
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
        #self.Pgen2 = self.mdl.addVars(self.gn,lb = self.Pmin, ub = self.Pmax,name = "Pgen2")
        self.Qgen = self.mdl.addVars(self.gn,lb = self.Qmin, ub = self.Qmax,name = "Qgen")
        #Bus variables
        self.ReV = self.mdl.addVars(self.n,lb = [-v for v in self.Vmax], ub = self.Vmax,name = "ReV")
        self.ImV = self.mdl.addVars(self.n,lb = [-v for v in self.Vmax], ub = self.Vmax,name = "ImV")
        #self.L = self.mdl.continuous_var_list(self.n,lb = self.Vmin, ub = self.Vmax,name = "L")
        #self.theta = self.mdl.continuous_var_list(self.n,lb = -np.pi, ub = np.pi,name = "theta")
        #Bus and edges variables
        self.ReW = self.mdl.addVars(self.symedges,lb = -self.radius**2, ub = self.radius**2,name = "ReW")
        self.ImW = self.mdl.addVars(self.symedges,lb = -self.radius**2, ub = self.radius**2,name = "ImW")
        #self.R = self.mdl.continuous_var_dict(self.symedges,lb = 0, ub = self.radius**2,name = "R")
        ######################################### Objective function ################################################
        #self.mdl.addConstrs((self.Pgen2[i]>=self.Pgen[i]**2 for i in range(self.gn)),name = 'quad')
        
        #self.objective = self.offset+self.mdl.scal_prod(self.Pgen,self.lincost)+self.mdl.scal_prod(self.Pgen2,self.quadcost) 
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
        # #Voltage magnitude discretization
        # for b in range(self.n):
        #     # self.mdl.add_constraint(self.L[b]**2<=self.ReW[b,b])
        #     for k in range(kdiscret+1):
        #          xbar = self.Vmin[b] + (k/kdiscret) * (self.Vmax[b]-self.Vmin[b])
        #          self.mdl.add_constraint(2*xbar*(self.L[b]-xbar)+xbar**2<=self.ReW[b,b])
        # self.mdl.add_constraints([self.Vmin[b] <=self.L[b] for b in range(self.n)])
        # self.mdl.add_constraints([self.Vmax[b] >=self.L[b] for b in range(self.n)])
        
        # for b in range(self.n): 
        #     self.mdl.add_constraint(self.L[b] >= self.Vmin[b] + (self.ReW[b,b] - (self.Vmin[b]**2))*((self.Vmax[b] - self.Vmin[b])/(self.Vmax[b]**2 - self.Vmin[b]**2)))
        # #Voltage  products magnitude discretization
        #self.mdl.addConstr([self.ReW[b,b]==self.R[b,b] for b in range(self.n)])
        #self.mdl.addConstr([self.R[b,a]==self.R[a,b] for (b,a) in self.edges])
        # for b,a in self.symedges:
        #     if b<a:
        #         self.mdl.add_constraint(self.R[b,a] >= self.Vmin[b] * self.L[a] + self.Vmin[a] * self.L[b] - self.Vmin[b]*self.Vmin[a])
        #         self.mdl.add_constraint(self.R[b,a] >= self.Vmax[b]*self.L[a] + self.Vmax[a]*self.L[b] - self.Vmax[b]*self.Vmax[a] )
        #         self.mdl.add_constraint(self.R[b,a] <= self.Vmax[b]*self.L[a] + self.Vmin[a]*self.L[b] - self.Vmax[b]*self.Vmin[a] )
        #         self.mdl.add_constraint(self.R[b,a] <= self.Vmax[a]*self.L[b] + self.Vmin[b] * self.L[a] - self.Vmax[a] *self.Vmin[b])
        #Voltage products angle discretization
        if True:
            for b,a in self.edgesNoDiag:
                if self.ThetaMaxByEdge[b,a]- self.ThetaMinByEdge[b,a]<=np.pi:
                    halfdiff =  0.5*(self.ThetaMaxByEdge[b,a]- self.ThetaMinByEdge[b,a])
                    mean =  0.5*(self.ThetaMaxByEdge[b,a] + self.ThetaMinByEdge[b,a])
                    self.mdl.addConstr(-np.sin(self.ThetaMinByEdge[b,a])*self.ReW[b,a] + np.cos(self.ThetaMinByEdge[b,a]) * self.ImW[b,a] >=  0)
                    self.mdl.addConstr(-np.sin(self.ThetaMaxByEdge[b,a])*self.ReW[b,a] + np.cos(self.ThetaMaxByEdge[b,a]) * self.ImW[b,a] <=  0)
                    #self.mdl.addConstr( np.cos(mean)*self.ReW[b,a] + np.sin(mean)*self.ImW[b,a]>= self.R[b,a]*np.cos(halfdiff))
                    #self.mdl.addConstr(self.ThetaMinByEdge[b,a]<=self.theta[b]- self.theta[a])
                    #self.mdl.add_constraint(self.theta[b]- self.theta[a]<=self.ThetaMaxByEdge[b,a])
            #self.mdl.add_constraint(self.theta[0]==0)
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
        
    
    
    # def lazy_sdpcuts_procedure(self):
    #     for idx_clique in range(len(self.cliques)):
    #         cl = self.cliques[idx_clique]
    #         nc = len(cl)
    #         ReM = np.zeros((nc,nc) )
    #         ImM = np.zeros((nc,nc) )
    #         for i,idx_bus in enumerate(cl):
    #             for j,idx_bus2 in enumerate(cl):
    #                 ReM[i,j] = self.ReW[idx_bus,idx_bus2].solution_value 
    #                 ImM[i,j] = self.ImW[idx_bus,idx_bus2].solution_value
    #         M = ReM+1j*ImM
    #         s, U = LA.eigh(M)            
    #         for k in range(nc):
    #             if s[k]<-self.accuracyC:
    #                 vector = U[:,k]
    #                 for t in range(2*nc):
    #                     random_magnitude = 0.95+0.1*np.random.random(nc)
    #                     random_angle = np.exp(1j*(-0.1+0.2)*np.random.random(nc))
    #                     vector_modified = vector * random_magnitude *random_angle
    #                     vector_modified = vector_modified.reshape((nc,1))
    #                     self.add_sdp_cut(idx_clique, vector_modified,True)
        
    #     for idx_clique in range(len(self.cliques)):
    #         cl = self.cliques[idx_clique]
    #         nc = len(cl)
    #         M = np.zeros((1+nc,1+nc) )
    #         M[0,0] = 1
    #         for i,idx_bus in enumerate(cl):
    #             M[1+i,0] = self.L[idx_bus].solution_value
    #             M[0,1+i] = self.L[idx_bus].solution_value
    #             for j,idx_bus2 in enumerate(cl):
    #                 M[1+i,1+j] = self.R[idx_bus,idx_bus2].solution_value 
    #         s, U = LA.eigh(M)
    #         for k in range(nc):
    #             if s[k]<-self.accuracyC:
    #                 vector = U[:,k]
    #                 self.add_sdp_cutR(idx_clique, vector)
    #                 for t in range(2*nc):
    #                     random_magnitude = 0.95+0.1*np.random.random(nc+1)
    #                     vector_modified = vector * random_magnitude 
    #                     vector_modified = vector_modified.reshape((nc+1,1))
    #                     self.add_sdp_cutR(idx_clique, vector_modified,True)
        
                    
    # def cp_procedure(self):
    #     rankone = True 
    #     maxratio,sdpmeasure=  0,0
    #     #################### Cutting planes for W[idx_clique] is SDP ################################
    #     for idx_clique in range(len(self.cliques)):
    #         cl = self.cliques[idx_clique]
    #         nc = len(cl)
    #         ReM = np.zeros((nc,nc) )
    #         ImM = np.zeros((nc,nc) )
    #         for i,idx_bus in enumerate(cl):
    #             for j,idx_bus2 in enumerate(cl):
    #                 ReM[i,j] = self.ReW[idx_bus,idx_bus2].solution_value 
    #                 ImM[i,j] = self.ImW[idx_bus,idx_bus2].solution_value
    #         M = ReM+1j*ImM
    #         s, U = LA.eigh(M)
    #         lambda_max = s.max()
    #         others = [abs(el) for el in list(s) if el!=lambda_max]
    #         rankone = rankone and (lambda_max>0) and (max(others)/abs(lambda_max)<1E-6)
    #         maxratio = max(maxratio,max(others)/abs(lambda_max))
    #         sdpmeasure = min(sdpmeasure, s.min())
    #         for k in range(nc):
    #             if s[k]<-self.accuracyC:
    #                 vector = U[:,k]
    #                 vector = vector.reshape((nc,1))
    #                 self.add_sdp_cut(idx_clique, vector)
    #     #################### Cutting planes for the objective function ################################
    #     obj_error = 0
    #     for i in range(self.gn):
    #         if self.quadcost[i] :
    #             obj_error = max(obj_error,self.Pgen[i].solution_value**2-self.Pgen2[i].solution_value)
    #             if self.Pgen[i].solution_value**2>self.Pgen2[i].solution_value+self.accuracyC:
    #                 xbar = self.Pgen[i].solution_value
    #                 self.mdl.add_constraint(2*xbar*(self.Pgen[i]-xbar)+xbar**2<=self.Pgen2[i])
    #     #################### Cutting planes for S - flow limits ################################
    #     bool_lineconstraints = True
    #     flow_error = 0
    #     if self.config['lineconstraints']=='S':
    #         for idx_line,line in enumerate(self.clinelistinv):
    #             b,a,h = line
    #             index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
    #             X = np.conj(self.Yff[line])*self.ReW[index_bus_b,index_bus_b].solution_value + (np.conj(self.Yft[line]))*(self.ReW[index_bus_b,index_bus_a].solution_value+1j*self.ImW[index_bus_b,index_bus_a].solution_value)
    #             flow_error = max(flow_error,abs(X)-self.Imax[idx_line])
    #             if abs(X)>=self.Imax[idx_line]+self.accuracyC:
    #                 bool_lineconstraints = False
    #                 theta = np.angle(X)
    #                 coefRWbb = np.cos(theta)*np.real(self.Yff[line]) - np.sin(theta)*np.imag(self.Yff[line])
    #                 coefRWba = np.cos(theta)*np.real(self.Yft[line]) - np.sin(theta)*np.imag(self.Yft[line])
    #                 coefIWba = np.cos(theta)*np.imag(self.Yft[line]) + np.sin(theta)*np.real(self.Yft[line])
    #                 self.mdl.add_constraint(coefRWbb*self.ReW[index_bus_b,index_bus_b]+coefRWba*self.ReW[index_bus_b,index_bus_a]+coefIWba*self.ImW[index_bus_b,index_bus_a]<=self.Imax[idx_line])
    #             #Switching indices to have (a,b,h) \in L
    #             aux = index_bus_b
    #             index_bus_b = index_bus_a
    #             index_bus_a = aux
    #             X = np.conj(self.Ytt[line])*self.ReW[index_bus_b,index_bus_b].solution_value + (np.conj(self.Ytf[line]))*(self.ReW[index_bus_b,index_bus_a].solution_value+1j*self.ImW[index_bus_b,index_bus_a].solution_value)
    #             flow_error = max(flow_error,abs(X)-self.Imax[idx_line])
    #             if abs(X)>=self.Imax[idx_line]+self.accuracyC:
    #                 bool_lineconstraints = False
    #                 theta = np.angle(X)
    #                 coefRWbb = np.cos(theta)*np.real(self.Ytt[line]) - np.sin(theta)*np.imag(self.Ytt[line])
    #                 coefRWba = np.cos(theta)*np.real(self.Ytf[line]) - np.sin(theta)*np.imag(self.Ytf[line])
    #                 coefIWba = np.cos(theta)*np.imag(self.Ytf[line]) + np.sin(theta)*np.real(self.Ytf[line])
    #                 self.mdl.add_constraint(coefRWbb*self.ReW[index_bus_b,index_bus_b]+coefRWba*self.ReW[index_bus_b,index_bus_a]+coefIWba*self.ImW[index_bus_b,index_bus_a]<=self.Imax[idx_line])
        
    #     self.maxratiologs.append(maxratio)
    #     self.sdpmeasurelogs.append(abs(sdpmeasure))
    #     self.sdp_error = abs(sdpmeasure)
    #     self.flow_error = flow_error
    #     self.obj_error = obj_error
    #     sdp = sdpmeasure>= - self.accuracyC
    #     quad_obj = obj_error<self.accuracyC
    #     print("Rank measure = {0}".format(maxratio))
    #     print("Errors: SDP = {0}, FlowLim = {1}, Obj = {2}".format(self.sdp_error, self.flow_error, self.obj_error))
    #     if (rankone and quad_obj and bool_lineconstraints) or (sdp and quad_obj and bool_lineconstraints):
    #         return rankone, sdp, quad_obj, bool_lineconstraints
        
    #     for idx_clique in range(len(self.cliques)):
    #         cl = self.cliques[idx_clique]
    #         nc = len(cl)
    #         M = np.zeros((1+nc,1+nc) )
    #         M[0,0] = 1
    #         for i,idx_bus in enumerate(cl):
    #             M[1+i,0] = self.L[idx_bus].solution_value
    #             M[0,1+i] = self.L[idx_bus].solution_value
    #             for j,idx_bus2 in enumerate(cl):
    #                 M[1+i,1+j] = self.R[idx_bus,idx_bus2].solution_value 
    #         s, U = LA.eigh(M)
    #         for k in range(nc):
    #             if s[k]<-self.accuracyC:
    #                 vector = U[:,k]
    #                 vector = vector.reshape((nc+1,1))
    #                 self.add_sdp_cutR(idx_clique, vector)
        
    #     #################### Cutting planes for R[b,b]-L[b] convex relation ################################
    #     for b in range(self.n):
    #         xbar =self.L[b].solution_value
    #         if xbar**2>self.ReW[b,b].solution_value+self.accuracyC:
    #             self.mdl.add_constraint(2*xbar*(self.L[b]-xbar)+xbar**2<=self.ReW[b,b])
    #     #################### Cutting planes for |W[b,a]| \leq R[b,a] ################################
    #     for b,a in self.edgesNoDiag:
    #         if np.sqrt(self.ReW[b,a].solution_value**2 + self.ImW[b,a].solution_value**2) > self.R[b,a].solution_value + self.accuracyC:
    #             theta = np.angle(self.ReW[b,a].solution_value + 1j*self.ImW[b,a].solution_value)
    #             self.mdl.add_constraint(self.ReW[b,a]*np.cos(theta) + self.ImW[b,a]*np.sin(theta)<=self.R[b,a])
    #     return rankone, sdp, quad_obj, bool_lineconstraints
                    
    
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
    

    # def diagnosis(self):
    #     print("Objective value = {0}".format(self.mdl.objective_value))
        
        
    #     for i in range(self.gn):
    #         print(self.Pgen2[i].solution_value-self.Pgen[i].solution_value**2)
        
    #     for counter,cl in enumerate(self.cliques):
    #         nc = len(cl)
    #         ReW = np.zeros((nc,nc) )
    #         ImW = np.zeros((nc,nc) ) 
    #         for i,b in enumerate(cl):
    #             for j,a in enumerate(cl):
    #                 ReW[i,j] = self.ReW[b,a].solution_value
    #                 ImW[i,j] = self.ImW[b,a].solution_value
    #         cliquemat = ReW+1j*ImW
    #         w, v = LA.eigh(cliquemat)
    #         print(w)
            
    #     for idx_bus in range(self.n):
    #         row, col = self.HM[idx_bus].nonzero()
    #         dicoHmbRe = {(row[aux],col[aux]):np.real(self.HM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
    #         dicoHmbIm = {(row[aux],col[aux]):np.imag(self.HM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
    #         assert(abs(sum([self.Pgen[idx_gen].solution_value for idx_gen in self.bus_to_gen[idx_bus]])-(self.Pload[idx_bus]+sum([self.ReW[i,j].solution_value*dicoHmbRe[i,j] for i,j in dicoHmbRe]) + sum([self.ImW[i,j].solution_value*dicoHmbIm[i,j] for i,j in dicoHmbIm])))<1E-6)
    #         row, col = self.ZM[idx_bus].nonzero()
    #         dicoZmbRe = {(row[aux],col[aux]):np.real(1j*self.ZM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
    #         dicoZmbIm = {(row[aux],col[aux]):np.imag(1j*self.ZM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
    #         assert(abs(sum([self.Qgen[idx_gen].solution_value for idx_gen in self.bus_to_gen[idx_bus]])-(self.Qload[idx_bus]+sum([self.ReW[i,j].solution_value*dicoZmbRe[i,j] for i,j in dicoZmbRe]) + sum([self.ImW[i,j].solution_value*dicoZmbIm[i,j] for i,j in dicoZmbIm])))<1E-6)
            