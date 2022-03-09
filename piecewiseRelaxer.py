# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:56:53 2021

@author: aoust
"""
import EnhancedSdpRelaxer
from numpy import linalg as LA
from docplex.mp.advmodel import AdvModel
from docplex.mp.solution import SolveSolution
import numpy as np
import operator
import pandas as pd
import tools
import time


kdiscret= 3
kdiscret2 = 3
max_number_of_new_BP = 4
k_for_slimit = 4
rank_one_ratio = 5 * 1e-6
max_local_it = 20

class piecewiseRelaxer():
    
    def __init__(self, ACOPF, config, local_optimizer_results):
      
        """
              
        """
        print("------ Piecewise relaxation solver for the ACOPF -----------")
        self.name = ACOPF.name
        self.config = config
        self.local_optimizer_results = local_optimizer_results
        self.sdp_relaxer = EnhancedSdpRelaxer.EnhancedSdpRelaxer(ACOPF)
        
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
        assert( ACOPF.config['lineconstraints']=='I' or ACOPF.config['lineconstraints']=='S' or ACOPF.config['lineconstraints']==False)
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

        self.x = {}
        self.delta = {}
        self.x_indices = {}
        self.delta_indices = {}
        self.x_parents = {}
        self.delta_parents = {}
        self.phimin = {}
        self.phimax = {}
        self.umin = {}
        self.umax = {}
        self.x_leaves = {}
        self.delta_leaves = {}
        for i in range(self.n):
            self.x_indices[i] = []
            self.x_leaves[i] = []
        for (b,a) in self.edgesNoDiag:
            self.delta_indices[(b,a)] = [] 
            self.delta_leaves[(b,a)] = [] 
        self.bestLB=0
    
    
    def alphaoffset(self,b,a,k):
        lower = self.ThetaMinByEdge[b,a] - self.phimin[(b,a),k]
        upper = self.ThetaMaxByEdge[b,a] - self.phimin[(b,a),k]
        msin = tools.minsin2(lower,upper)
        Rbamin,Rbamax = self.Vmin[b]*self.Vmin[a],self.Vmax[b]*self.Vmax[a]
        return min(Rbamin*msin,Rbamax*msin)
    
    def betaoffset(self,b,a,k):
        lower = self.ThetaMinByEdge[b,a] - self.phimax[(b,a),k]
        upper = self.ThetaMaxByEdge[b,a] - self.phimax[(b,a),k]
        msin = tools.maxsin2(lower,upper)
        Rbamin,Rbamax = self.Vmin[b]*self.Vmin[a],self.Vmax[b]*self.Vmax[a]
        return max(Rbamin*msin,Rbamax*msin)
    
    def gammaoffset(self,b,a,k):
        halfdiff =  0.5*(self.phimax[(b,a),k]- self.phimin[(b,a),k])
        mean =  0.5*(self.phimax[(b,a),k] + self.phimin[(b,a),k])
        lower = self.ThetaMinByEdge[b,a] - mean
        upper = self.ThetaMaxByEdge[b,a] - mean
        mcos = tools.mincos2(lower,upper)
        Rbamin,Rbamax = self.Vmin[b]*self.Vmin[a],self.Vmax[b]*self.Vmax[a]
        return Rbamax*np.cos(halfdiff) - min(Rbamin*mcos,Rbamax*mcos)
              
    
    def moffset(self,i,k):
        extrem1 = self.umin[i,k]+(self.Vmin[i]**2 - (self.umin[i,k]**2))*((self.umax[i,k] - self.umin[i,k])/(self.umax[i,k]**2 - self.umin[i,k]**2)) - self.Vmin[i]
        extrem2 = self.umin[i,k]+(self.Vmax[i]**2 - (self.umin[i,k]**2))*((self.umax[i,k] - self.umin[i,k])/(self.umax[i,k]**2 - self.umin[i,k]**2)) - self.Vmax[i]
        assert(max(extrem1,extrem2)>=0)
        return max(extrem1,extrem2)

    def build_model(self):
        self.binaries = False
        self.mdl = AdvModel('PieceWiseRelaxation')
        ######################################### Variables ######################################
        #Gen variables
        self.Pgen = self.mdl.continuous_var_list(self.gn,lb = self.Pmin, ub = self.Pmax,name = "Pgen")
        self.Pgen2 = self.mdl.continuous_var_list(self.gn,lb = 0, ub = [self.Pmax[i]**2 for i in range(self.gn)],name = "Pgen2")
        self.Qgen = self.mdl.continuous_var_list(self.gn,lb = self.Qmin, ub = self.Qmax,name = "Qgen")
        #Bus variables
        self.L = self.mdl.continuous_var_list(self.n,lb = self.Vmin, ub = self.Vmax,name = "L")
        self.theta = self.mdl.continuous_var_list(self.n,lb = -np.pi, ub = np.pi,name = "theta")
        #Bus and edges variables
        self.ReW = self.mdl.continuous_var_dict(self.symedges,lb = -self.radius**2, ub = self.radius**2,name = "ReW")
        self.ImW = self.mdl.continuous_var_dict(self.symedges,lb = -self.radius**2, ub = self.radius**2,name = "ImW")
        self.R = self.mdl.continuous_var_dict(self.symedges,lb = 0, ub = self.radius**2,name = "R")
        ######################################### Objective function ################################################
        for i in range(self.gn):
            if self.quadcost[i]:
                for k in range(kdiscret+1):
                    xbar = self.Pmin[i] + (k/kdiscret) * (self.Pmax[i]-self.Pmin[i])
                    self.add_obj_cut(i,xbar)
                    
            else:
                self.mdl.add_constraint(self.Pgen2[i]==0)
        self.objective = self.offset+self.mdl.scal_prod(self.Pgen,self.lincost)+self.mdl.scal_prod(self.Pgen2,self.quadcost) 
        self.mdl.minimize(self.objective)
        ######################################### Linear constraints defining F ######################################
        self.mdl.add_constraints([self.Pgen[i]<= self.Pmax[i] for i in range(self.gn)])
        self.mdl.add_constraints([self.Pgen[i]>= self.Pmin[i] for i in range(self.gn)])
        self.mdl.add_constraints([self.Qgen[i]<= self.Qmax[i] for i in range(self.gn)])
        self.mdl.add_constraints([self.Qgen[i]>= self.Qmin[i] for i in range(self.gn)])
        self.mdl.add_constraints([self.ReW[b,a]==self.ReW[a,b] for (b,a) in self.edges])
        self.mdl.add_constraints([self.ImW[b,a]==-self.ImW[a,b] for (b,a) in self.edges])
        self.mdl.add_constraints([self.ReW[b,b]<=self.Vmax[b]**2 for b in range(self.n)])
        self.mdl.add_constraints([self.ReW[b,b]>=self.Vmin[b]**2 for b in range(self.n)])
        #Power conservation constraints
        for idx_bus in range(self.n):
            row, col = self.HM[idx_bus].nonzero()
            dicoHmbRe = {(row[aux],col[aux]):np.real(self.HM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            dicoHmbIm = {(row[aux],col[aux]):np.imag(self.HM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            self.mdl.add_constraint(self.mdl.sum([self.Pgen[idx_gen] for idx_gen in self.bus_to_gen[idx_bus]])==self.Pload[idx_bus]+self.mdl.sum([self.ReW[i,j]*dicoHmbRe[i,j] for i,j in dicoHmbRe]) + self.mdl.sum([self.ImW[i,j]*dicoHmbIm[i,j] for i,j in dicoHmbIm]))
            row, col = self.ZM[idx_bus].nonzero()
            dicoZmbRe = {(row[aux],col[aux]):np.real(1j*self.ZM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            dicoZmbIm = {(row[aux],col[aux]):np.imag(1j*self.ZM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            self.mdl.add_constraint(self.mdl.sum([self.Qgen[idx_gen] for idx_gen in self.bus_to_gen[idx_bus]])==self.Qload[idx_bus]+self.mdl.sum([self.ReW[i,j]*dicoZmbRe[i,j] for i,j in dicoZmbRe]) + self.mdl.sum([self.ImW[i,j]*dicoZmbIm[i,j] for i,j in dicoZmbIm]))
        if self.config['lineconstraints']=='I':
            for idx_line in range(self.cl):
                row, col = self.Nf[idx_line].nonzero()
                dicoNflineRe = {(row[aux],col[aux]):np.real(self.Nf[idx_line][row[aux],col[aux]]) for aux in range(len(row))}
                dicoNflineIm = {(row[aux],col[aux]):np.imag(self.Nf[idx_line][row[aux],col[aux]]) for aux in range(len(row))}
                self.mdl.add_constraint(self.mdl.sum([self.ReW[i,j]*dicoNflineRe[i,j] for i,j in dicoNflineRe]) + self.mdl.sum([self.ImW[i,j]*dicoNflineIm[i,j] for i,j in dicoNflineIm]) <=self.Imax[idx_line]**2)
                row, col = self.Nt[idx_line].nonzero()
                dicoNtlineRe = {(row[aux],col[aux]):np.real(self.Nt[idx_line][row[aux],col[aux]]) for aux in range(len(row))}
                dicoNtlineIm = {(row[aux],col[aux]):np.imag(self.Nt[idx_line][row[aux],col[aux]]) for aux in range(len(row))}
                self.mdl.add_constraint(self.mdl.sum([self.ReW[i,j]*dicoNtlineRe[i,j] for i,j in dicoNtlineRe]) + self.mdl.sum([self.ImW[i,j]*dicoNtlineIm[i,j] for i,j in dicoNtlineIm]) <=self.Imax[idx_line]**2)
        elif self.config['lineconstraints']=='S':
            
            
            for idx_line,line in enumerate(self.clinelistinv):
                b,a,h = line
                index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
                for k in range(k_for_slimit):
                    theta = -np.pi + 2*(k/k_for_slimit)*np.pi
                    self.add_SlimitFrom_cut(index_bus_b,index_bus_a,idx_line,theta)
                    # coefRWbb = np.cos(theta)*np.real(self.Yff[line]) - np.sin(theta)*np.imag(self.Yff[line])
                    # coefRWba = np.cos(theta)*np.real(self.Yft[line]) - np.sin(theta)*np.imag(self.Yft[line])
                    # coefIWba = np.cos(theta)*np.imag(self.Yft[line]) + np.sin(theta)*np.real(self.Yft[line])
                    # self.mdl.add_constraint(coefRWbb*self.ReW[index_bus_b,index_bus_b]+coefRWba*self.ReW[index_bus_b,index_bus_a]+coefIWba*self.ImW[index_bus_b,index_bus_a]<=self.Imax[idx_line])
                #Switching indices to have (a,b,h) \in L
                aux = index_bus_b
                index_bus_b = index_bus_a
                index_bus_a = aux
                for k in range(k_for_slimit):
                    theta = -np.pi + 2*(k/k_for_slimit)*np.pi
                    self.add_SlimitTo_cut(index_bus_b,index_bus_a,idx_line,theta)
                    # coefRWbb = np.cos(theta)*np.real(self.Ytt[line]) - np.sin(theta)*np.imag(self.Ytt[line])
                    # coefRWba = np.cos(theta)*np.real(self.Ytf[line]) - np.sin(theta)*np.imag(self.Ytf[line])
                    # coefIWba = np.cos(theta)*np.imag(self.Ytf[line]) + np.sin(theta)*np.real(self.Ytf[line])
                    # self.mdl.add_constraint(coefRWbb*self.ReW[index_bus_b,index_bus_b]+coefRWba*self.ReW[index_bus_b,index_bus_a]+coefIWba*self.ImW[index_bus_b,index_bus_a]<=self.Imax[idx_line])
        
        else:
            assert(self.config['lineconstraints']==False)
        ########################################## Discretization constraints #######################################################################   
        #Voltage magnitude discretization
        for b in range(self.n):
            # self.mdl.add_constraint(self.L[b]**2<=self.ReW[b,b])
            for k in range(kdiscret+1):
                 xbar = self.Vmin[b] + (k/kdiscret) * (self.Vmax[b]-self.Vmin[b])
                 self.add_quadLcut(b,xbar)
                 
        self.mdl.add_constraints([self.Vmin[b] <=self.L[b] for b in range(self.n)])
        self.mdl.add_constraints([self.Vmax[b] >=self.L[b] for b in range(self.n)])
        
        for b in range(self.n): 
            if self.Vmax[b] - self.Vmin[b] >0:
                self.mdl.add_constraint(self.L[b] >= self.Vmin[b] + (self.ReW[b,b] - (self.Vmin[b]**2))*((self.Vmax[b] - self.Vmin[b])/(self.Vmax[b]**2 - self.Vmin[b]**2)))
            else:
                self.mdl.add_constraint(self.L[b] == self.Vmin[b])
                self.mdl.add_constraint(self.ReW[b,b] == self.Vmin[b]**2)
        #Voltage  products magnitude discretization
        self.mdl.add_constraints([self.ReW[b,b]==self.R[b,b] for b in range(self.n)])
        self.mdl.add_constraints([self.R[b,a]==self.R[a,b] for (b,a) in self.edges])
        for b,a in self.symedges:
            if b<a:
                self.mdl.add_constraint(self.R[b,a] >= self.Vmin[b] * self.L[a] + self.Vmin[a] * self.L[b] - self.Vmin[b]*self.Vmin[a])
                self.mdl.add_constraint(self.R[b,a] >= self.Vmax[b]*self.L[a] + self.Vmax[a]*self.L[b] - self.Vmax[b]*self.Vmax[a] )
                self.mdl.add_constraint(self.R[b,a] <= self.Vmax[b]*self.L[a] + self.Vmin[a]*self.L[b] - self.Vmax[b]*self.Vmin[a] )
                self.mdl.add_constraint(self.R[b,a] <= self.Vmax[a]*self.L[b] + self.Vmin[b] * self.L[a] - self.Vmax[a] *self.Vmin[b])
        #Voltage products angle discretization
        for b,a in self.edgesNoDiag:
            if self.ThetaMaxByEdge[b,a]- self.ThetaMinByEdge[b,a]<=np.pi:
                halfdiff =  0.5*(self.ThetaMaxByEdge[b,a]- self.ThetaMinByEdge[b,a])
                mean =  0.5*(self.ThetaMaxByEdge[b,a] + self.ThetaMinByEdge[b,a])
                self.mdl.add_constraint(-np.sin(self.ThetaMinByEdge[b,a])*self.ReW[b,a] + np.cos(self.ThetaMinByEdge[b,a]) * self.ImW[b,a] >=  0)
                self.mdl.add_constraint(-np.sin(self.ThetaMaxByEdge[b,a])*self.ReW[b,a] + np.cos(self.ThetaMaxByEdge[b,a]) * self.ImW[b,a] <=  0)
                self.mdl.add_constraint( np.cos(mean)*self.ReW[b,a] + np.sin(mean)*self.ImW[b,a]>= self.R[b,a]*np.cos(halfdiff))
                self.mdl.add_constraint(self.ThetaMinByEdge[b,a]<=self.theta[b]- self.theta[a])
                self.mdl.add_constraint(self.theta[b]- self.theta[a]<=self.ThetaMaxByEdge[b,a])
        self.mdl.add_constraint(self.theta[0]==0)
        ##################################################################################################################################################################################################################################
        for b,a in self.edgesNoDiag:
            for k in range(kdiscret2+1):
                theta = self.ThetaMinByEdge[b,a] + (k/kdiscret2)*(self.ThetaMaxByEdge[b,a]-self.ThetaMinByEdge[b,a])
                self.add_circle_cut(b,a,theta)
                #self.mdl.add_constraint(self.ReW[b,a]*np.cos(theta) + self.ImW[b,a]*np.sin(theta)<=self.R[b,a])
        
    
    def solve(self,timelimit,maxit,rel_tol, ubcuts,with_lazy_random_sdp_cuts):
        self.mdl.context.solver.log_output = True
        self.mdl.parameters.emphasis.numerical = 1
        self.mdl.parameters.mip.tolerances.mipgap = 0.5*rel_tol
        #self.mdl.parameters.read.scale = -1
        self.mdl.parameters.simplex.tolerances.feasibility = 1E-9
      
        self.total_it_number = it =  0
        self.UB = self.local_optimizer_results.value
        self.bestLBlogs,self.bestUBlogs, self.maxratiologs, self.sdpmeasurelogs = [], [],[],[]
        assert(self.local_optimizer_results.success==1)
        self.accuracyNC = 1E-7
        self.accuracyC = 1E-7
        deadline = time.time()+timelimit
        while it<maxit and (time.time()<deadline):
            it+=1
            local_it= 0
            self.sdp_error, self.flow_error, self.obj_error = 1,1,1
            convex_tol = 1E-4
            while (max(self.sdp_error, max(self.flow_error, self.obj_error)))>min(1E-4,convex_tol) and (local_it<max_local_it):
                local_it+=1
                remaining_time = max(0,deadline - time.time())
                if remaining_time==0:
                    return "Time limit"
                self.mdl.set_time_limit(remaining_time)
                self.mdl.solve()
                self.total_it_number+=1
                print(self.mdl.solve_details.status)
                if 'time limit' in self.mdl.solve_details.status:
                     return "Time limit"
                 
                if ubcuts and self.mdl.objective_value>self.bestLB:
                    self.mdl.add_constraint(self.objective>=self.bestLB-(1e-7*abs(self.bestLB)))
                new_bound = self.mdl.solve_details.best_bound if self.binaries else self.mdl.objective_value
                self.bestLB = max(self.bestLB,new_bound)
                self.bestLBlogs.append(self.bestLB)
                self.bestUBlogs.append(self.local_optimizer_results.value)
                print("Best bound = {0}".format(self.bestLB))
                print("Objective value = {0}".format(self.mdl.objective_value))
                print("Relative gap = {0}".format((self.UB-self.bestLB)/self.UB))
                
                rank_one, sdp, quad_obj,bool_lineconstraints = self.cp_procedure()
                
               
                if local_it%8==7:
                    Vmin,Vmax,ThetaMinByEdge,ThetaMaxByEdge = self.current_subinterval_bounds()
                    value,X1,X2, PgenVal, LVal, ReWVal,ImWVal = self.sdp_relaxer.computeDuals(Vmin,Vmax,ThetaMinByEdge,ThetaMaxByEdge)
                    self.add_sdp_duals_W(X1)
                    self.add_sdp_duals_R(X2)
         
                print(rank_one, sdp, quad_obj,bool_lineconstraints)
                
                self.iteration_log()
                
                if (self.UB-self.bestLB)<rel_tol*self.UB:
                    print("Gap closed, MIPS solution is optimal")
                    return "Local solver's Gap closed"
                
                if rank_one and sdp and quad_obj and bool_lineconstraints:
                    print("Solved, better solution than MIPS found.")
                    self.diagnosis()
                    return "Better sol. found"
            
            if with_lazy_random_sdp_cuts:
                self.lazy_sdpcuts_procedure()
            edges_norm_violation_angle,edges_norm_violation_prod = [],[]
            
            for (b,a) in self.edgesNoDiag:
                violation_product = abs(self.R[b,a].solution_value - np.sqrt(self.ReW[b,b].solution_value * self.ReW[a,a].solution_value ))
                violation_angle = abs(self.R[b,a].solution_value**2 - (self.ReW[b,a].solution_value**2 + self.ImW[b,a].solution_value**2 ))
                edges_norm_violation_angle.append((b,a,violation_angle))
                edges_norm_violation_prod.append((b,a,violation_product))
            
            edges_norm_violation_angle.sort(key=operator.itemgetter(2),reverse = True)
            edges_norm_violation_prod.sort(key=operator.itemgetter(2),reverse = True)
            convex_tol = 0.05*(edges_norm_violation_angle[0][2]+edges_norm_violation_prod[0][2])
            
            new_break_points = False
            
            for k in range(min(max_number_of_new_BP,len(edges_norm_violation_angle))):
                edge = edges_norm_violation_angle[k][0],edges_norm_violation_angle[k][1]
                if edges_norm_violation_angle[k][2]>self.accuracyNC:
                    auxbool = self.add_detail_delta(edge)
                    new_break_points = new_break_points or auxbool
            
            bus_to_discretize = set()
            for k in range(min(max_number_of_new_BP,len(edges_norm_violation_prod))):
                if edges_norm_violation_prod[k][2]>self.accuracyNC:
                    bus_to_discretize.add(edges_norm_violation_prod[k][0])
                    bus_to_discretize.add(edges_norm_violation_prod[k][1])
            for b in bus_to_discretize:
                    auxbool = self.add_detail_R(b)
                    new_break_points = new_break_points or auxbool
            self.add_mip_start()
            self.local_heuristic()
            
            print("Best LB = {0}".format(self.bestLB))
        self.diagnosis()
        
        if it==maxit:
            return "Max number of it"
        else:
            return "Time limit"
    
    def lazy_sdpcuts_procedure(self):
        for idx_clique in range(len(self.cliques)):
            cl = self.cliques[idx_clique]
            nc = len(cl)
            ReM = np.zeros((nc,nc) )
            ImM = np.zeros((nc,nc) )
            for i,idx_bus in enumerate(cl):
                for j,idx_bus2 in enumerate(cl):
                    ReM[i,j] = self.ReW[idx_bus,idx_bus2].solution_value 
                    ImM[i,j] = self.ImW[idx_bus,idx_bus2].solution_value
            M = ReM+1j*ImM
            s, U = LA.eigh(M)            
            for k in range(nc):
                if s[k]<-self.accuracyC:
                    vector = U[:,k]
                    for t in range(2*nc):
                        random_magnitude = 0.95+0.1*np.random.random(nc)
                        random_angle = np.exp(1j*(-0.1+0.2)*np.random.random(nc))
                        vector_modified = vector * random_magnitude *random_angle
                        vector_modified = vector_modified.reshape((nc,1))
                        self.add_sdp_cut(idx_clique, vector_modified,True)
        
        for idx_clique in range(len(self.cliques)):
            cl = self.cliques[idx_clique]
            nc = len(cl)
            M = np.zeros((1+nc,1+nc) )
            M[0,0] = 1
            for i,idx_bus in enumerate(cl):
                M[1+i,0] = self.L[idx_bus].solution_value
                M[0,1+i] = self.L[idx_bus].solution_value
                for j,idx_bus2 in enumerate(cl):
                    M[1+i,1+j] = self.R[idx_bus,idx_bus2].solution_value 
            s, U = LA.eigh(M)
            for k in range(nc):
                if s[k]<-self.accuracyC:
                    vector = U[:,k]
                    self.add_sdp_cutR(idx_clique, vector)
                    for t in range(2*nc):
                        random_magnitude = 0.95+0.1*np.random.random(nc+1)
                        vector_modified = vector * random_magnitude 
                        vector_modified = vector_modified.reshape((nc+1,1))
                        self.add_sdp_cutR(idx_clique, vector_modified,True)
        
                    
    def cp_procedure(self):
        rankone = True 
        maxratio,sdpmeasure=  0,0
        #################### Cutting planes for W[idx_clique] is SDP ################################
        for idx_clique in range(len(self.cliques)):
            cl = self.cliques[idx_clique]
            nc = len(cl)
            ReM = np.zeros((nc,nc) )
            ImM = np.zeros((nc,nc) )
            for i,idx_bus in enumerate(cl):
                for j,idx_bus2 in enumerate(cl):
                    ReM[i,j] = self.ReW[idx_bus,idx_bus2].solution_value 
                    ImM[i,j] = self.ImW[idx_bus,idx_bus2].solution_value
            M = ReM+1j*ImM
            s, U = LA.eigh(M)
            lambda_max = s.max()
            others = [abs(el) for el in list(s) if el!=lambda_max]
            rankone = rankone and (lambda_max>0) and (max(others)/abs(lambda_max)<rank_one_ratio)
            maxratio = max(maxratio,max(others)/abs(lambda_max))
            sdpmeasure = min(sdpmeasure, s.min())
            for k in range(nc):
                if s[k]<-self.accuracyC:
                    vector = U[:,k]
                    vector = vector.reshape((nc,1))
                    self.add_sdp_cut(idx_clique, vector)
        #################### Cutting planes for the objective function ################################
        obj_error = 0
        for i in range(self.gn):
            if self.quadcost[i] :
                obj_error = max(obj_error,self.Pgen[i].solution_value**2-self.Pgen2[i].solution_value)
                if self.Pgen[i].solution_value**2>self.Pgen2[i].solution_value+self.accuracyC:
                    xbar = self.Pgen[i].solution_value
                    self.add_obj_cut(i,xbar)
                    #self.mdl.add_constraint(2*xbar*(self.Pgen[i]-xbar)+xbar**2<=self.Pgen2[i])
        #################### Cutting planes for S - flow limits ################################
        bool_lineconstraints = True
        flow_error = 0
        if self.config['lineconstraints']=='S':
            for idx_line,line in enumerate(self.clinelistinv):
                b,a,h = line
                index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
                X = np.conj(self.Yff[line])*self.ReW[index_bus_b,index_bus_b].solution_value + (np.conj(self.Yft[line]))*(self.ReW[index_bus_b,index_bus_a].solution_value+1j*self.ImW[index_bus_b,index_bus_a].solution_value)
                flow_error = max(flow_error,abs(X)-self.Imax[idx_line])
                if abs(X)>=self.Imax[idx_line]+self.accuracyC:
                    bool_lineconstraints = False
                    theta = np.angle(X)
                    self.add_SlimitFrom_cut(index_bus_b,index_bus_a,idx_line,theta)
                    # coefRWbb = np.cos(theta)*np.real(self.Yff[line]) - np.sin(theta)*np.imag(self.Yff[line])
                    # coefRWba = np.cos(theta)*np.real(self.Yft[line]) - np.sin(theta)*np.imag(self.Yft[line])
                    # coefIWba = np.cos(theta)*np.imag(self.Yft[line]) + np.sin(theta)*np.real(self.Yft[line])
                    # self.mdl.add_constraint(coefRWbb*self.ReW[index_bus_b,index_bus_b]+coefRWba*self.ReW[index_bus_b,index_bus_a]+coefIWba*self.ImW[index_bus_b,index_bus_a]<=self.Imax[idx_line])
                #Switching indices to have (a,b,h) \in L
                aux = index_bus_b
                index_bus_b = index_bus_a
                index_bus_a = aux
                X = np.conj(self.Ytt[line])*self.ReW[index_bus_b,index_bus_b].solution_value + (np.conj(self.Ytf[line]))*(self.ReW[index_bus_b,index_bus_a].solution_value+1j*self.ImW[index_bus_b,index_bus_a].solution_value)
                flow_error = max(flow_error,abs(X)-self.Imax[idx_line])
                if abs(X)>=self.Imax[idx_line]+self.accuracyC:
                    bool_lineconstraints = False
                    theta = np.angle(X)
                    self.add_SlimitTo_cut(index_bus_b,index_bus_a,idx_line,theta)
                    # coefRWbb = np.cos(theta)*np.real(self.Ytt[line]) - np.sin(theta)*np.imag(self.Ytt[line])
                    # coefRWba = np.cos(theta)*np.real(self.Ytf[line]) - np.sin(theta)*np.imag(self.Ytf[line])
                    # coefIWba = np.cos(theta)*np.imag(self.Ytf[line]) + np.sin(theta)*np.real(self.Ytf[line])
                    # self.mdl.add_constraint(coefRWbb*self.ReW[index_bus_b,index_bus_b]+coefRWba*self.ReW[index_bus_b,index_bus_a]+coefIWba*self.ImW[index_bus_b,index_bus_a]<=self.Imax[idx_line])
        
        self.maxratiologs.append(maxratio)
        self.sdpmeasurelogs.append(abs(sdpmeasure))
        self.sdp_error = abs(sdpmeasure)
        self.flow_error = flow_error
        self.obj_error = obj_error
        sdp = sdpmeasure>= - self.accuracyC
        quad_obj = obj_error<self.accuracyC
        print("Rank measure = {0}".format(maxratio))
        print("Errors: SDP = {0}, FlowLim = {1}, Obj = {2}".format(self.sdp_error, self.flow_error, self.obj_error))
        if (rankone and quad_obj and bool_lineconstraints) or (sdp and quad_obj and bool_lineconstraints):
            return rankone, sdp, quad_obj, bool_lineconstraints
        
        for idx_clique in range(len(self.cliques)):
            cl = self.cliques[idx_clique]
            nc = len(cl)
            M = np.zeros((1+nc,1+nc) )
            M[0,0] = 1
            for i,idx_bus in enumerate(cl):
                M[1+i,0] = self.L[idx_bus].solution_value
                M[0,1+i] = self.L[idx_bus].solution_value
                for j,idx_bus2 in enumerate(cl):
                    M[1+i,1+j] = self.R[idx_bus,idx_bus2].solution_value 
            s, U = LA.eigh(M)
            for k in range(nc):
                if s[k]<-self.accuracyC:
                    vector = U[:,k]
                    vector = vector.reshape((nc+1,1))
                    self.add_sdp_cutR(idx_clique, vector)
        
        #################### Cutting planes for R[b,b]-L[b] convex relation ################################
        for b in range(self.n):
            xbar =self.L[b].solution_value
            if xbar**2>self.ReW[b,b].solution_value+self.accuracyC:
                self.add_quadLcut(b,xbar)
                #self.mdl.add_constraint(2*xbar*(self.L[b]-xbar)+xbar**2<=self.ReW[b,b])
        #################### Cutting planes for |W[b,a]| \leq R[b,a] ################################
        for b,a in self.edgesNoDiag:
            if np.sqrt(self.ReW[b,a].solution_value**2 + self.ImW[b,a].solution_value**2) > self.R[b,a].solution_value + self.accuracyC:
                theta = np.angle(self.ReW[b,a].solution_value + 1j*self.ImW[b,a].solution_value)
                self.add_circle_cut(b,a,theta)
                #self.mdl.add_constraint(self.ReW[b,a]*np.cos(theta) + self.ImW[b,a]*np.sin(theta)<=self.R[b,a])
        return rankone, sdp, quad_obj, bool_lineconstraints
                    
###### Individual linear cut functions #################################################
    def add_sdp_cut(self, idx_clique, vector, lazy =False):
        cl = self.cliques[idx_clique]
        nc = len(cl)
        vector = vector.reshape((nc,1))
        M = vector.dot(np.conj(vector.T))
        dicoMRe = {(cl[i],cl[j]):np.real(M[i,j]) for i in range(nc) for j in range(nc)}
        dicoMIm = {(cl[i],cl[j]):np.imag(M[i,j]) for i in range(nc) for j in range(nc)}
        if not(lazy):
            self.mdl.add_constraint(self.mdl.sum([self.ReW[i,j]*dicoMRe[i,j] for i,j in dicoMRe]) + self.mdl.sum([self.ImW[i,j]*dicoMIm[i,j] for i,j in dicoMIm]) >=0)
        else:
            self.mdl.add_lazy_constraint(self.mdl.sum([self.ReW[i,j]*dicoMRe[i,j] for i,j in dicoMRe]) + self.mdl.sum([self.ImW[i,j]*dicoMIm[i,j] for i,j in dicoMIm]) >=0)
        
    def add_obj_cut(self,i, xbar):
        self.mdl.add_constraint(2*xbar*(self.Pgen[i]-xbar)+xbar**2<=self.Pgen2[i])
        
    def add_quadLcut(self,b,xbar):
        self.mdl.add_constraint(2*xbar*(self.L[b]-xbar)+xbar**2<=self.ReW[b,b])
        
    def add_circle_cut(self,b,a,theta):
        self.mdl.add_constraint(self.ReW[b,a]*np.cos(theta) + self.ImW[b,a]*np.sin(theta)<=self.R[b,a])
    
    def add_SlimitFrom_cut(self,index_bus_b,index_bus_a,idx_line,theta):
        line = self.clinelist[idx_line]
        assert(line[0]==self.buslist[index_bus_b])
        assert(line[1]==self.buslist[index_bus_a])
        coefRWbb = np.cos(theta)*np.real(self.Yff[line]) - np.sin(theta)*np.imag(self.Yff[line])
        coefRWba = np.cos(theta)*np.real(self.Yft[line]) - np.sin(theta)*np.imag(self.Yft[line])
        coefIWba = np.cos(theta)*np.imag(self.Yft[line]) + np.sin(theta)*np.real(self.Yft[line])
        self.mdl.add_constraint(coefRWbb*self.ReW[index_bus_b,index_bus_b]+coefRWba*self.ReW[index_bus_b,index_bus_a]+coefIWba*self.ImW[index_bus_b,index_bus_a]<=self.Imax[idx_line])
    
    def add_SlimitTo_cut(self,index_bus_b,index_bus_a,idx_line,theta):
        line = self.clinelist[idx_line]
        assert(line[1]==self.buslist[index_bus_b])
        assert(line[0]==self.buslist[index_bus_a])
        coefRWbb = np.cos(theta)*np.real(self.Ytt[line]) - np.sin(theta)*np.imag(self.Ytt[line])
        coefRWba = np.cos(theta)*np.real(self.Ytf[line]) - np.sin(theta)*np.imag(self.Ytf[line])
        coefIWba = np.cos(theta)*np.imag(self.Ytf[line]) + np.sin(theta)*np.real(self.Ytf[line])
        self.mdl.add_constraint(coefRWbb*self.ReW[index_bus_b,index_bus_b]+coefRWba*self.ReW[index_bus_b,index_bus_a]+coefIWba*self.ImW[index_bus_b,index_bus_a]<=self.Imax[idx_line])
                
    def add_sdp_cutR(self, idx_clique, vector,lazy = False):
        cl = self.cliques[idx_clique]
        nc = len(cl)
        assert(len(vector)==1+nc)
        assert(np.linalg.norm(vector-np.conj(vector))==0)
        vector = vector.reshape((nc+1,1))
        M = vector.dot(np.conj(vector.T))
        dicoM_partR = {(cl[i],cl[j]):(M[i+1,j+1]) for i in range(nc) for j in range(nc)}
        dicoM_partL1 = {cl[i]:(M[0,i+1]) for i in range(nc)}
        dicoM_partL2 = {cl[i]:(M[i+1,0]) for i in range(nc)} 
        
        ref= np.zeros((1+nc,1+nc))
        ref[0,0]=1.0
        if np.linalg.norm(M-ref)>0:
            if not(lazy):
                self.mdl.add_constraint(M[0,0]+ self.mdl.sum([self.L[b]*dicoM_partL1[b] for b in dicoM_partL1]) + self.mdl.sum([self.L[b]*dicoM_partL2[b] for b in dicoM_partL2]) + self.mdl.sum([self.R[b,a]*dicoM_partR[b,a] for b,a in dicoM_partR]) >=0)
            else:
                self.mdl.add_lazy_constraint(M[0,0]+ self.mdl.sum([self.L[b]*dicoM_partL1[b] for b in dicoM_partL1]) + self.mdl.sum([self.L[b]*dicoM_partL2[b] for b in dicoM_partL2]) + self.mdl.sum([self.R[b,a]*dicoM_partR[b,a] for b,a in dicoM_partR]) >=0)
            
    def add_UB(self,ub):
        self.mdl.add_constraint(self.objective<=ub)
    
    def add_sdp_duals_W(self, X):
        assert(len(X)==self.cliques_nbr)
        for i in range(self.cliques_nbr):
            mat = 0.5*(X[i]+np.conj(X[i].T))
            s, U = LA.eigh(mat)
            for k in range(self.ncliques[i]):
                vector = U[:,k]
                vector = vector.reshape((self.ncliques[i],1))
                self.add_sdp_cut(i, vector)
    def add_sdp_duals_R(self, X):
        assert(len(X)==self.cliques_nbr)
        for i in range(self.cliques_nbr):
            mat = 0.5*(X[i]+np.conj(X[i].T))
            s, U = LA.eigh(mat)
            for k in range(self.ncliques[i]):
                vector = U[:,k]
                vector = vector.reshape((1+self.ncliques[i],1))
                self.add_sdp_cutR(i, vector)
                
    
    def add_quad_cuts(self,PgenVal, LVal, ReWVal,ImWVal):
        for i in range(self.gn):
            if self.quadcost[i]:
                self.add_obj_cut(i, PgenVal[i])
        for i in range(self.n):
            self.add_quadLcut(i, LVal[i])
        for (b,a) in self.edgesNoDiag:
            theta = np.angle(ReWVal[(b,a)] + 1j*ImWVal[(b,a)])
            self.add_circle_cut(b,a,theta)
        if self.config['lineconstraints']=='S':
            for idx_line,line in enumerate(self.clinelistinv):
                b,a,h = line
                index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
                X = np.conj(self.Yff[line])*ReWVal[(index_bus_b,index_bus_b)] + (np.conj(self.Yft[line]))*(ReWVal[(index_bus_b,index_bus_a)]+1j*ImWVal[(index_bus_b,index_bus_a)])
                theta = np.angle(X)
                self.add_SlimitFrom_cut(index_bus_b,index_bus_a,idx_line,theta)
                #Switching indices to have (a,b,h) \in L
                aux = index_bus_b
                index_bus_b = index_bus_a
                index_bus_a = aux
                X = np.conj(self.Ytt[line])*ReWVal[(index_bus_b,index_bus_b)] + (np.conj(self.Ytf[line]))*(ReWVal[(index_bus_b,index_bus_a)]+1j*ImWVal[(index_bus_b,index_bus_a)])
                theta = np.angle(X)
                self.add_SlimitTo_cut(index_bus_b,index_bus_a,idx_line,theta)
            
            
            
    
    
##### Functions to increment the number of binary variables
    def add_detail_delta(self,edge):
        b,a = edge
        new = []
        assert(edge in self.edgesNoDiag)
        if len(self.delta_indices[edge])==0:
            thetamin, thetamax = self.ThetaMinByEdge[edge],self.ThetaMaxByEdge[edge]
            val = np.angle(self.ReW[b,a].solution_value + 1j*self.ImW[b,a].solution_value)
            if abs(thetamin-val)>1E-7 and abs(thetamax-val)>1E-7:
                self.binaries = True
                self.delta_indices[edge] = list(range(2))
                self.delta_leaves[edge] = list(range(2))
                
                if not(abs(val-thetamin)<=np.pi and abs(val-thetamax)<=np.pi):
                    val = 0.5*(thetamin+thetamax)
                k = 0
                self.phimin[edge,k] = thetamin 
                self.phimax[edge,k] = val
                self.delta_parents[edge,k] = -1
                self.delta[edge,k] = self.mdl.binary_var()
                k=1
                self.phimin[edge,k] = val 
                self.phimax[edge,k] = thetamax
                self.delta_parents[edge,k] = -1
                self.delta[edge,k] = self.mdl.binary_var()
                
                self.mdl.add_constraint(self.mdl.sum([self.delta[edge,k] for k in self.delta_indices[edge]])==1)
                new = self.delta_indices[edge]
        else:
            
            delta_values = np.array([ self.delta[edge,k].solution_value for k in self.delta_leaves[edge]])
            parent = self.delta_leaves[edge][delta_values.argmax()]
            assert(abs(self.delta[edge,parent].solution_value-1)<1E-5)
            val = np.angle(self.ReW[b,a].solution_value + 1j*self.ImW[b,a].solution_value)
            if abs( self.phimin[edge,parent]-val)>1E-7 and abs( self.phimax[edge,parent]-val)>1E-7:
                L = self.phimax[edge,parent]-self.phimin[edge,parent]
                assert(L>0)
                if val>0.2*L+self.phimin[edge,parent] and val<self.phimax[edge,parent] - 0.2*L:
                    "If the current value is not to close from its interval's bounds."
                    self.binaries = True
                    self.delta_leaves[edge].remove(parent)
                    K = len(self.delta_indices[edge])
                    self.delta_indices[edge].append(K)
                    self.delta_indices[edge].append(K+1)
                    self.delta_leaves[edge].append(K)
                    self.delta_leaves[edge].append(K+1)
                    self.delta_parents[edge,K] = parent
                    self.delta_parents[edge,K+1] = parent
                    self.phimin[edge,K],self.phimax[edge,K] = self.phimin[edge,parent],val
                    self.phimin[edge,K+1],self.phimax[edge,K+1] = val, self.phimax[edge,parent]
                    self.delta[edge,K],self.delta[edge,K+1] = self.mdl.binary_var(),self.mdl.binary_var()
                    self.mdl.add_constraint(self.delta[edge,K]+self.delta[edge,K+1] ==self.delta[edge,parent])
                    new = [K,K+1]
                else:
                    value1,value2 = min(val,0.2*L+self.phimin[edge,parent]),max(val,self.phimax[edge,parent] - 0.2*L)
                    assert(value1<value2)
                    self.binaries = True
                    self.delta_leaves[edge].remove(parent)
                    K = len(self.delta_indices[edge])
                    self.delta_indices[edge].append(K)
                    self.delta_indices[edge].append(K+1)
                    self.delta_indices[edge].append(K+2)
                    self.delta_leaves[edge].append(K)
                    self.delta_leaves[edge].append(K+1)
                    self.delta_leaves[edge].append(K+2)
                    self.delta_parents[edge,K] = parent
                    self.delta_parents[edge,K+1] = parent
                    self.delta_parents[edge,K+2] = parent
                    self.phimin[edge,K],self.phimax[edge,K] = self.phimin[edge,parent],value1
                    self.phimin[edge,K+1],self.phimax[edge,K+1] = value1, value2
                    self.phimin[edge,K+2],self.phimax[edge,K+2] = value2, self.phimax[edge,parent]
                    self.delta[edge,K],self.delta[edge,K+1],self.delta[edge,K+2] = self.mdl.binary_var(),self.mdl.binary_var(),self.mdl.binary_var()
                    self.mdl.add_constraint(self.delta[edge,K]+self.delta[edge,K+1]+self.delta[edge,K+2] ==self.delta[edge,parent])
                    new = [K,K+1,K+2]
                    
        if len(new)>0:
            self.mdl.add_constraint(self.mdl.sum([self.phimin[edge,k]*self.delta[edge,k] for k in self.delta_leaves[edge]])<=self.theta[b]- self.theta[a])
            self.mdl.add_constraint(self.theta[b]- self.theta[a]<= self.mdl.sum([self.phimax[edge,k]*self.delta[edge,k] for k in self.delta_leaves[edge]]))
           
        for k in new:
            halfdiff =  0.5*(self.phimax[edge,k]- self.phimin[edge,k])
            mean =  0.5*(self.phimax[edge,k] + self.phimin[edge,k])
            self.mdl.add_constraint(-np.sin(self.phimin[edge,k])*self.ReW[b,a] + np.cos(self.phimin[edge,k]) * self.ImW[b,a] >=    (1-self.delta[edge,k])*self.alphaoffset(b,a,k),'type_delta1')
            self.mdl.add_constraint(-np.sin(self.phimax[edge,k])*self.ReW[b,a] + np.cos(self.phimax[edge,k]) * self.ImW[b,a] <=    (1-self.delta[edge,k])*self.betaoffset(b,a,k),'type_delta2')
            self.mdl.add_constraint( (1-self.delta[edge,k])*self.gammaoffset(b,a,k)+ np.cos(mean)*self.ReW[b,a] + np.sin(mean)*self.ImW[b,a]>= self.R[b,a]*np.cos(halfdiff),'type_delta3')
        
        return len(new)>0
    
    
    def add_detail_R(self,b):
        new = []
        if len(self.x_indices[b])==0:
            vmin, vmax = self.Vmin[b],self.Vmax[b]
            val = self.L[b].solution_value
            print(abs(vmin-val),abs(vmax-val))
            if abs(vmin-val)>1E-7 and abs(vmax-val)>1E-7:
                self.binaries = True
                self.x_indices[b] = list(range(2))
                self.x_leaves[b] = list(range(2))
                k=0
                self.umin[b,k] = vmin 
                self.umax[b,k] = val
                self.x_parents[b,k] = -1
                self.x[b,k] = self.mdl.binary_var()
                k=1
                self.umin[b,k] = val
                self.umax[b,k] = vmax
                self.x_parents[b,k] = -1
                self.x[b,k] = self.mdl.binary_var()
                self.mdl.add_constraint(self.mdl.sum([self.x[b,k] for k in self.x_indices[b]])==1)
                new = self.x_indices[b]
        else:
            xvalues = np.array([ self.x[b,k].solution_value for k in self.x_leaves[b]])
            parent = self.x_leaves[b][xvalues.argmax()]
            assert(abs(self.x[b,parent].solution_value-1)<1E-3)
            val = self.L[b].solution_value
            assert(self.umin[b,parent]<=val+1E-7)
            assert(self.umax[b,parent]>=val-1E-7)
            if abs(self.umin[b,parent]-val)>1E-7 and abs(self.umax[b,parent]-val)>1E-7:
                L = self.umax[b,parent] - self.umin[b,parent]
                assert(L>0)
                if val>0.2*L+self.umin[b,parent] and val<self.umax[b,parent] - 0.2*L:
                    "If the current value is not to close from its interval's bounds."
                    self.binaries = True
                    self.x_leaves[b].remove(parent)
                    K = len(self.x_indices[b])
                    self.x_indices[b].append(K)
                    self.x_indices[b].append(K+1)
                    self.x_leaves[b].append(K)
                    self.x_leaves[b].append(K+1)
                    self.x_parents[b,K] = parent
                    self.x_parents[b,K+1] = parent
                    
                    self.umin[b,K],self.umax[b,K] = self.umin[b,parent],val
                    self.umin[b,K+1],self.umax[b,K+1] = val, self.umax[b,parent]
                    assert(abs(self.umin[b,K]-self.umax[b,K])>1e-8)
                    assert(abs(self.umin[b,K+1]-self.umax[b,K+1])>1e-8)
                    self.x[b,K],self.x[b,K+1] = self.mdl.binary_var(),self.mdl.binary_var()
                    self.mdl.add_constraint(self.x[b,K]+self.x[b,K+1] ==self.x[b,parent])
                    new = [K,K+1]
                else:
                    self.binaries = True
                    self.x_leaves[b].remove(parent)
                    value1,value2 = min(val,0.2*L+self.umin[b,parent]), max(val,self.umax[b,parent] - 0.2*L)
                    assert(value1<value2)
                    K = len(self.x_indices[b])
                    self.x_indices[b].append(K)
                    self.x_indices[b].append(K+1)
                    self.x_indices[b].append(K+2)
                    self.x_leaves[b].append(K)
                    self.x_leaves[b].append(K+1)
                    self.x_leaves[b].append(K+2)
                    self.x_parents[b,K] = parent
                    self.x_parents[b,K+1] = parent
                    self.x_parents[b,K+2] = parent
                    self.umin[b,K],self.umax[b,K] = self.umin[b,parent],value1
                    self.umin[b,K+1],self.umax[b,K+1] = value1, value2
                    self.umin[b,K+2],self.umax[b,K+2] = val, self.umax[b,parent]
                    assert(abs(self.umin[b,K]-self.umax[b,K])>1e-8)
                    assert(abs(self.umin[b,K+1]-self.umax[b,K+1])>1e-8)
                    self.x[b,K],self.x[b,K+1],self.x[b,K+2]  = self.mdl.binary_var(),self.mdl.binary_var(),self.mdl.binary_var()
                    self.mdl.add_constraint(self.x[b,K]+self.x[b,K+1]+self.x[b,K+2] ==self.x[b,parent])
                    new = [K,K+1,K+2]
        
        
        if len(new)>0:
            self.mdl.add_constraint(self.mdl.sum([self.umin[b,k] * self.x[b,k] for k in self.x_leaves[b]])<=self.L[b])
            self.mdl.add_constraint(self.mdl.sum([self.umax[b,k] * self.x[b,k] for k in self.x_leaves[b]])>=self.L[b])
            self.mdl.add_constraint(self.mdl.sum([(self.umin[b,k]**2) * self.x[b,k] for k in self.x_leaves[b]])<=self.ReW[b,b])
            self.mdl.add_constraint(self.mdl.sum([(self.umax[b,k]**2) * self.x[b,k] for k in self.x_leaves[b]])>=self.ReW[b,b])
            self.mdl.add_constraints([self.L[b] + self.moffset(b,k)*(1-self.x[b,k]) >= self.umin[b,k] + (self.ReW[b,b] - (self.umin[b,k]**2))*((self.umax[b,k] - self.umin[b,k])/(self.umax[b,k]**2 - self.umin[b,k]**2)) for k in new] )
        
        for k in new:
            for a in self.neighbors[b]:
                self.mdl.add_constraint(self.R[b,a] + (1-self.x[b,k]) * (self.umin[b,k]*self.Vmax[a] - self.umin[b,k]*self.Vmin[a]) >= self.umin[b,k] * self.L[a] + self.Vmin[a] * self.L[b] - self.umin[b,k]*self.Vmin[a])
                self.mdl.add_constraint(self.R[b,a] + (1-self.x[b,k]) * (self.Vmax[b]*self.Vmax[a] - self.Vmin[b]*self.Vmin[a])>= self.umax[b,k]*self.L[a] + self.Vmax[a]*self.L[b] - self.umax[b,k]*self.Vmax[a] )
                self.mdl.add_constraint(self.R[b,a] + (1-self.x[b,k]) * (self.Vmin[b]*self.Vmin[a] - self.Vmax[b]*self.Vmax[a])<= self.umax[b,k]*self.L[a] + self.Vmin[a]*self.L[b] - self.umax[b,k]*self.Vmin[a] )
                self.mdl.add_constraint(self.R[b,a] + (1-self.x[b,k]) * (self.umin[b,k]*self.Vmin[a] - self.umin[b,k]*self.Vmax[a]) <= self.Vmax[a]*self.L[b] + self.umin[b,k] * self.L[a] - self.Vmax[a] *self.umin[b,k])
                 
        return len(new)>0
    
    
    def add_mip_start(self):
        sol = SolveSolution(self.mdl)
        gen_index = [i for i in range(len(self.local_optimizer_results.Pgen)) if not(i in self.inactive_generators)]
        val = self.offset
        for i in range(self.gn):
            sol.add_var_value(self.Pgen[i],self.local_optimizer_results.Pgen[gen_index[i]])
            val+=self.lincost[i]*self.local_optimizer_results.Pgen[gen_index[i]]
            if self.quadcost[i]:
                sol.add_var_value(self.Pgen2[i],self.local_optimizer_results.Pgen[gen_index[i]]**2)
                val+=self.quadcost[i]*self.local_optimizer_results.Pgen[gen_index[i]]**2
            else:
                sol.add_var_value(self.Pgen2[i],0)
            sol.add_var_value(self.Qgen[i],self.local_optimizer_results.Qgen[gen_index[i]])
        
        print("MIP start Value = {0}".format(val))
        for i in range(self.n):
            sol.add_var_value(self.L[i],self.local_optimizer_results.VM[i])
            sol.add_var_value(self.theta[i],self.local_optimizer_results.theta[i] - self.local_optimizer_results.theta[0])
            
            sol.add_var_value(self.ReW[i,i],self.local_optimizer_results.VM[i]**2)
            sol.add_var_value(self.ImW[i,i],0)
        
        for (b,a) in self.symedges:
            prod = self.local_optimizer_results.V[b]*np.conj(self.local_optimizer_results.V[a])
            sol.add_var_value(self.ReW[b,a],np.real(prod))
            sol.add_var_value(self.ImW[b,a],np.imag(prod))
            sol.add_var_value(self.R[b,a],abs(prod))
        
        for edge in self.edgesNoDiag:
            b,a = edge
            diff = (self.local_optimizer_results.theta[b]-self.local_optimizer_results.theta[a])
            for k in self.delta_indices[edge]:
                boolean = (diff>=self.phimin[edge,k]) and (diff<self.phimax[edge,k])
                sol.add_var_value(self.delta[edge,k],int(boolean))
                
        for b in range(self.n):
            for k in self.x_indices[b]:
                boolean = (self.local_optimizer_results.VM[b]>=self.umin[b,k]) and (self.local_optimizer_results.VM[b]<=self.umax[b,k])
                sol.add_var_value(self.x[b,k],int(boolean))
                        
        #assert(sol.is_feasible_solution(silent=False,tolerance = 1e-4))
        self.mdl.add_mip_start(sol)
        

    def diagnosis(self):
        print("Objective value = {0}".format(self.mdl.objective_value))
                
        for i in range(self.gn):
            print(self.Pgen2[i].solution_value-self.Pgen[i].solution_value**2)
        
        for counter,cl in enumerate(self.cliques):
            nc = len(cl)
            ReW = np.zeros((nc,nc) )
            ImW = np.zeros((nc,nc) ) 
            for i,b in enumerate(cl):
                for j,a in enumerate(cl):
                    ReW[i,j] = self.ReW[b,a].solution_value
                    ImW[i,j] = self.ImW[b,a].solution_value
            cliquemat = ReW+1j*ImW
            w, v = LA.eigh(cliquemat)
            print(w)
            
        for idx_bus in range(self.n):
            row, col = self.HM[idx_bus].nonzero()
            dicoHmbRe = {(row[aux],col[aux]):np.real(self.HM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            dicoHmbIm = {(row[aux],col[aux]):np.imag(self.HM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            assert(abs(sum([self.Pgen[idx_gen].solution_value for idx_gen in self.bus_to_gen[idx_bus]])-(self.Pload[idx_bus]+sum([self.ReW[i,j].solution_value*dicoHmbRe[i,j] for i,j in dicoHmbRe]) + sum([self.ImW[i,j].solution_value*dicoHmbIm[i,j] for i,j in dicoHmbIm])))<1E-6)
            row, col = self.ZM[idx_bus].nonzero()
            dicoZmbRe = {(row[aux],col[aux]):np.real(1j*self.ZM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            dicoZmbIm = {(row[aux],col[aux]):np.imag(1j*self.ZM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            assert(abs(sum([self.Qgen[idx_gen].solution_value for idx_gen in self.bus_to_gen[idx_bus]])-(self.Qload[idx_bus]+sum([self.ReW[i,j].solution_value*dicoZmbRe[i,j] for i,j in dicoZmbRe]) + sum([self.ImW[i,j].solution_value*dicoZmbIm[i,j] for i,j in dicoZmbIm])))<1E-6)
    
    
    def current_subinterval_bounds(self):
        Vmin,Vmax = np.zeros(self.n),np.zeros(self.n)
        for b in range(self.n):
            if len(self.x_leaves[b])==0:
                Vmin[b] = self.Vmin[b]
                Vmax[b] = self.Vmax[b]
            else:
                check = False
                for k in self.x_leaves[b]:
                    if self.x[b,k].solution_value>=0.99:
                        assert(check==False)
                        check=True
                        Vmin[b] = self.umin[b,k]
                        Vmax[b] = self.umax[b,k]
        ThetaMinByEdge,ThetaMaxByEdge = {},{}
        for b,a in self.edgesNoDiag:
            if len(self.delta_leaves[b,a])==0:
                ThetaMinByEdge[(b,a)] = self.ThetaMinByEdge[(b,a)]
                ThetaMaxByEdge[(b,a)] = self.ThetaMaxByEdge[(b,a)]
                ThetaMinByEdge[(a,b)] = self.ThetaMinByEdge[(a,b)]
                ThetaMaxByEdge[(a,b)] = self.ThetaMaxByEdge[(a,b)]
            else:
                check = False
                for k in self.delta_leaves[(b,a)]:
                    if self.delta[(b,a),k].solution_value>=0.99:
                        assert(check==False)
                        check = True
                        ThetaMinByEdge[(b,a)] = self.phimin[(b,a),k]
                        ThetaMaxByEdge[(b,a)] = self.phimax[(b,a),k]
                        ThetaMinByEdge[(a,b)] = -self.phimax[(b,a),k]
                        ThetaMaxByEdge[(a,b)] = -self.phimin[(b,a),k]
        
        return Vmin,Vmax,ThetaMinByEdge,ThetaMaxByEdge
    
    def iteration_log(self):
        df = pd.DataFrame()
        df['gap'] = (np.ones(len(self.bestLBlogs))*self.UB - np.array(self.bestLBlogs))/self.UB
        df['LB'] = self.bestLBlogs
        df['UB'] = self.bestUBlogs
        df['rank_ratio'] = self.maxratiologs
        df['sdp_measure'] = self.sdpmeasurelogs
        
        
        if self.config['lineconstraints']=='I':
            df.to_csv('output_I/'+self.name+'_global_logs.csv')
        elif (self.config['lineconstraints']=='S'):
            df.to_csv('output_S/'+self.name+'_global_logs.csv')
        else:
            df.to_csv('output_no_lim/'+self.name+'_global_logs.csv')
            
    
    
    def local_heuristic(self):
        
        thetaValref= {(b,a) : np.angle(self.ReW[b,a].solution_value+1j*self.ImW[b,a].solution_value) if (b<a) else np.angle(self.ReW[a,b].solution_value-1j*self.ImW[a,b].solution_value)  for (b,a) in self.ThetaMinByEdge}
        vref = [self.L[b].solution_value for b in range(self.n)]
        PgenVal = [self.Pgen[b].solution_value for b in range(self.gn)]
        self.local_optimizer_results.update_pen(100)
        self.local_optimizer_results.update_pref(PgenVal)
        self.local_optimizer_results.update_thetaref(thetaValref)
        self.local_optimizer_results.update_vref(vref)
        self.local_optimizer_results.solve()
        self.UB = self.local_optimizer_results.value
            