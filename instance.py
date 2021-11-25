# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:53:00 2021

@author: aoust
"""

import parserOPF,PhaseDiffBound
import numpy as np
from scipy.sparse import lil_matrix
from cvxopt import spmatrix, amd
import chompack as cp
import random


#My infty
myinf_power_lim = 1E4

class ACOPFinstance():
    
    def __init__(self, filepath,name,config):
        """
        
        Parameters
        ----------
        filepath : string
            Location of the .m file in MATPOWER data format.

        -------
        Load the model.

        """
               
        self.name = name
        self.config = config
        
        parser = parserOPF.mpcCase(filepath) 
        OPF_parser = parserOPF.OPF_Data(parser)
                
        #Sizes
        self.baseMVA = OPF_parser.baseMVA
        self.n, self.m, self.gn = OPF_parser.n, OPF_parser.m, OPF_parser.gn
        #Generator quantities
        self.C = OPF_parser.C
        genlist,lincost, quadcost = [],[],[]
        for generator in OPF_parser.SLR:
            genlist.append(generator)
            lincost.append(self.C[(generator[0],generator[1],1)])
            quadcost.append(self.C[(generator[0],generator[1],2)])
        self.lincost = np.array(lincost)
        self.quadcost = np.array(quadcost)
        self.genlist = genlist
        assert(len(self.genlist)==self.gn)
        self.Pmin, self.Qmin, self.Pmax, self.Qmax = [OPF_parser.SLR[self.genlist[idx_gen]] for idx_gen in range(self.gn)], [OPF_parser.SLC[self.genlist[idx_gen]] for idx_gen in range(self.gn)], [OPF_parser.SUR[self.genlist[idx_gen]] for idx_gen in range(self.gn)], [OPF_parser.SUC[self.genlist[idx_gen]] for idx_gen in range(self.gn)]
       
        
        self.offset = 0
        for generator in OPF_parser.SLR:
            if (generator[0],generator[1],0) in self.C:
                self.offset+=self.C[(generator[0],generator[1],0)]
        self.inactive_generators = OPF_parser.inactive_generators
        
        #Bus quantities
        self.buslist = []
        self.buslistinv ={}
        i=0
        for bus in OPF_parser.busType:
            self.buslist.append(bus)
            self.buslistinv[bus] = i
            i+=1
        for i in range(self.n):
            assert(self.buslistinv[self.buslist[i]]==i)
        self.busType = OPF_parser.busType
        self.angmin, self.angmax = OPF_parser.angmin, OPF_parser.angmax
        self.Vmin, self.Vmax = [OPF_parser.VL[self.buslist[i]] for i in range(self.n)], [OPF_parser.VU[self.buslist[i]] for i in range(self.n)]
        self.A = OPF_parser.A
        self.Pload = [np.real(OPF_parser.SD[self.buslist[i]]) for i in range(self.n)]
        self.Qload = [np.imag(OPF_parser.SD[self.buslist[i]]) for i in range(self.n)]
        self.preprocessing_power_bounds()
                
        #Lines quantities
        self.status = OPF_parser.status
        self.Yff, self.Yft, self.Ytf, self.Ytt = OPF_parser.Yff, OPF_parser.Yft, OPF_parser.Ytf, OPF_parser.Ytt 
        self.cl = 0
        
        
        self.clinelist,self.clinelistinv = [],[]
        
        
        self.clinelist = []
        self.clinelistinv ={}
        i=0
        counter = 0
        for line in OPF_parser.SU:
            if OPF_parser.SU[line]>0:
                self.clinelist.append(line)
                self.clinelistinv[line] = i
                i+=1 
            counter+=1
        self.cl = len(self.clinelist)
            
            
        self.Imax = [OPF_parser.SU[self.clinelist[idx_line]] for idx_line in range(self.cl)]
        self.test_status()
        
        #Construct cliques
        self.build_cliques(config["cliques_strategy"])
        self.SVM = {idx_clique : sum([self.Vmax[bus]**2 for bus in self.cliques[idx_clique]]) for idx_clique in range(self.cliques_nbr)}
        
                
        # #Construct m_cb matrices
        self.M = {}
        #Parts of M related to the lines #self.M[bus] = lil_matrix((self.n,self.n),dtype = np.complex128)
        self.M = {}
        for bus in self.A:
            index_bus = self.buslistinv[bus]
            self.M[index_bus] = lil_matrix((self.n,self.n),dtype = np.complex128)
            
            self.M[index_bus][index_bus,index_bus] += self.A[bus]
        for (b,a,h) in self.Yff:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            self.M[index_bus_b][index_bus_b,index_bus_b] += self.Yff[(b,a,h)]
        for (b,a,h) in self.Yft:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            self.M[index_bus_b][index_bus_b,index_bus_a] += self.Yft[(b,a,h)]
        for (a,b,h) in self.Ytt:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            self.M[index_bus_b][index_bus_b,index_bus_b] += self.Ytt[(a,b,h)]
        for (a,b,h) in self.Ytf:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            self.M[index_bus_b][index_bus_b,index_bus_a] += self.Ytf[(a,b,h)]
        
        #Conversion to csc_matrices
        for couple in self.M:
            self.M[couple] = self.M[couple].tocsc()
            
        self.HM, self.ZM, self.assigned_buses, self.assigned_lines = {} , {},{},{}
        for idx_clique in range(self.cliques_nbr):
            self.assigned_buses[idx_clique] = set()
            self.assigned_lines[idx_clique] = set()
        del idx_clique
            
        self.HM, self.ZM = {},{}
        for index_bus_b in self.M:
            self.HM[index_bus_b] = 0.5 * (self.M[index_bus_b]+(self.M[index_bus_b]).H)
            self.ZM[index_bus_b] = 0.5 * (self.M[index_bus_b]-(self.M[index_bus_b]).H)
        
        #Build Nf and Nt matrices
        if self.config["lineconstraints"]=='I':
            self.Nf = {}
            self.Nt = {}
            for idx_line,line in enumerate(self.clinelistinv):
                b,a,h = line
                index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
                #Build Nf line matrix
                self.Nf[idx_line] = lil_matrix((self.n,self.n),dtype = np.complex128)
                self.Nf[idx_line][index_bus_b,index_bus_b] = np.conj(self.Yff[line]) * self.Yff[line]
                self.Nf[idx_line][index_bus_a,index_bus_b] = np.conj(self.Yft[line]) * self.Yff[line]
                self.Nf[idx_line][index_bus_b,index_bus_a] = np.conj(self.Yff[line]) * self.Yft[line]
                self.Nf[idx_line][index_bus_a,index_bus_a] = np.conj(self.Yft[line]) * self.Yft[line]
                
                #Build Nt line matrix
                self.Nt[idx_line] = lil_matrix((self.n,self.n),dtype = np.complex128)
                self.Nt[idx_line][index_bus_b,index_bus_b] = np.conj(self.Ytf[line]) * self.Ytf[line]
                self.Nt[idx_line][index_bus_a,index_bus_b] = np.conj(self.Ytt[line]) * self.Ytf[line]
                self.Nt[idx_line][index_bus_b,index_bus_a] = np.conj(self.Ytf[line]) * self.Ytt[line]
                self.Nt[idx_line][index_bus_a,index_bus_a] = np.conj(self.Ytt[line]) * self.Ytt[line] 
           
                
    def build_cliques(self,strategy):
        self.edges = {}
        I = [i for i in range(self.n)]
        J = [i for i in range(self.n)]
        
        if strategy == "ASP":
            for (b,a,h) in self.Yff:
                index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
                i, j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
                if not((i, j) in self.edges):
                    self.edges[(i, j)]=1
                    I.append(i)
                    J.append(j)
        if strategy == 'full':
            for j in range(self.n):
                for i in range(j+1,self.n):
                    self.edges[(i, j)]=1
                    I.append(i)
                    J.append(j)
        A = spmatrix(1.0, I, J, (self.n,self.n))
        symb = cp.symbolic(A, p=amd.order)
        self.cliques = (symb.cliques(reordered=False))
        for cl in self.cliques:
            cl.sort()
        self.ncliques = [len(cl) for cl in self.cliques]
        self.cliques_parent = symb.parent()
        self.localBusIdx = {}
        self.globalBusIdx_to_cliques = []
        self.N = 0
        for i in range(self.n):
            self.globalBusIdx_to_cliques.append([])
        for clique_idx,clique in enumerate(self.cliques):
            self.N+=len(clique)
            for local_idx,global_idx in enumerate(clique):
                self.localBusIdx[clique_idx,global_idx] = local_idx
                self.globalBusIdx_to_cliques[global_idx].append(clique_idx)
                
        self.cliques_intersection = []
        for clique_idx,clique in enumerate(self.cliques):
            if self.cliques_parent[clique_idx]==clique_idx:
                self.cliques_intersection.append([])
            else:
                set_clique,set_clique_parent = set(clique),set(self.cliques[self.cliques_parent[clique_idx]])
                self.cliques_intersection.append(set_clique.intersection(set_clique_parent))
        for i in range(len(self.cliques_intersection)):
            inter = list(self.cliques_intersection[i])
            inter.sort()
            self.cliques_intersection[i] = inter
        self.edges_to_clique = {}
        for (i,j) in self.edges:
            si = set(self.globalBusIdx_to_cliques[i])
            sj = set(self.globalBusIdx_to_cliques[j])
            inter = si.intersection(sj)
            assert(len(inter)>0)
            random.seed(i*self.n+j)
            self.edges_to_clique[(i,j)] = random.choice([k for k in inter])
        self.cliques_nbr = len(self.cliques)
        
        self.angleManagement()
   
    
    def test_status(self):
        """Check wether the lines are indeed active """
        for l in self.Yff:
            assert(self.status[l] ==1.0)
        for l in self.Yft:
            assert(self.status[l] ==1.0)
        for l in self.Ytf:
            assert(self.status[l] ==1.0)
        for l in self.Ytt:
            assert(self.status[l] ==1.0)
            
           
    def preprocessing_power_bounds(self):
        """Handle absence of bounds. """
        self.blocked_beta_gen_moins,self.blocked_beta_gen_plus, self.blocked_gamma_gen_moins, self.blocked_gamma_gen_plus = [],[],[],[]
        for i,gen in enumerate(self.genlist):
            assert(self.genlist[i]==gen)
            if self.Pmin[i]==-np.inf:
                print("Pmin = -inf for gen {0}. replaced by large negative value".format(gen))
                self.Pmin[i] = -myinf_power_lim
                self.blocked_beta_gen_moins.append(gen[0])
            if self.Pmax[i]==np.inf:
                print("Pmax = +inf for gen {0}. replaced by large positive value".format(gen))
                self.Pmax[i] = myinf_power_lim
                self.blocked_beta_gen_plus.append(gen[0])
            if self.Qmin[i]==-np.inf:
                print("Qmin = -inf for gen {0}. replaced by large negative value".format(gen))
                self.Qmin[i] = -myinf_power_lim
                self.blocked_gamma_gen_moins.append(gen[0])
            if self.Qmax[i]==np.inf:
                print("Qmax = +inf for gen {0}. replaced by large positive value".format(gen))
                self.Qmax[i] = myinf_power_lim
                self.blocked_gamma_gen_plus.append(gen[0])
                
    def angleManagement(self):
        self.SymEdgesNoDiag = set()
        self.SymEdgesNoDiag_to_clique = {}
        
        for cl in self.cliques:
            for i in cl:
                for j in cl:
                    if i!=j:
                        self.SymEdgesNoDiag.add((i,j))
                        si = set(self.globalBusIdx_to_cliques[i])
                        sj = set(self.globalBusIdx_to_cliques[j])
                        inter = si.intersection(sj)
                        assert(len(inter)>0)
                        random.seed(i*self.n+j)
                        self.SymEdgesNoDiag_to_clique[(i,j)] = random.choice([k for k in inter])
                        
        self.ThetaMinByEdge = {(b,a):-np.pi for (b,a) in self.SymEdgesNoDiag}
        self.ThetaMaxByEdge = {(b,a):np.pi for (b,a) in self.SymEdgesNoDiag}
        
        #Taking into account phase difference limit imposed by the self
        for line in self.angmin:
            bus_b,bus_a,h = line
            index_bus_b,index_bus_a = self.buslistinv[bus_b],self.buslistinv[bus_a]
            self.ThetaMinByEdge[(index_bus_b,index_bus_a)] = max(self.ThetaMinByEdge[(index_bus_b,index_bus_a)],np.pi*(self.angmin[line]/180))
        for line in self.angmax:
            bus_b,bus_a,h = line
            index_bus_b,index_bus_a = self.buslistinv[bus_b],self.buslistinv[bus_a]
            self.ThetaMaxByEdge[(index_bus_b,index_bus_a)] = min(self.ThetaMaxByEdge[(index_bus_b,index_bus_a)],np.pi*(self.angmax[line]/180))
        #Taking into account phase difference limit deduce from the flow constraints
        #print('Starting Computation of Phase Difference Limits from Line Constraints')
        for idx_line,line in enumerate(self.clinelistinv):
            b,a,h = line
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            if self.config['lineconstraints']=='I':
                z = -self.Yft[line]/self.Yff[line]
                l = self.Vmin[index_bus_a]/self.Vmax[index_bus_b]
                u = self.Vmax[index_bus_a]/self.Vmin[index_bus_b]
                R = self.Imax[idx_line]/(abs(self.Yff[line])*self.Vmin[index_bus_b])
                region = PhaseDiffBound.PhaseDiffBound(z,l,u,R)
                self.ThetaMinByEdge[(index_bus_b,index_bus_a)] = max(self.ThetaMinByEdge[(index_bus_b,index_bus_a)],region.phimin)
                self.ThetaMaxByEdge[(index_bus_b,index_bus_a)] = min(self.ThetaMaxByEdge[(index_bus_b,index_bus_a)],region.phimax)
                z = -self.Ytt[line]/self.Ytf[line]
                l = self.Vmin[index_bus_a]/self.Vmax[index_bus_b]
                u = self.Vmax[index_bus_a]/self.Vmin[index_bus_b]
                R = self.Imax[idx_line]/(abs(self.Ytf[line])*self.Vmin[index_bus_b])
                region = PhaseDiffBound.PhaseDiffBound(z,l,u,R)
                self.ThetaMinByEdge[(index_bus_b,index_bus_a)] = max(self.ThetaMinByEdge[(index_bus_b,index_bus_a)],region.phimin)
                self.ThetaMaxByEdge[(index_bus_b,index_bus_a)] = min(self.ThetaMaxByEdge[(index_bus_b,index_bus_a)],region.phimax)
            else:

                assert(self.config['lineconstraints']=='S')
                z = -np.conj(self.Yff[line]/self.Yft[line])
                l = self.Vmin[index_bus_b]/self.Vmax[index_bus_a]
                u = self.Vmax[index_bus_b]/self.Vmin[index_bus_a]
                R = self.Imax[idx_line]/(abs(self.Yft[line])*self.Vmin[index_bus_b]*self.Vmin[index_bus_a])
                region = PhaseDiffBound.PhaseDiffBound(z,l,u,R)
                #region.plot(self.name+'_{0}_{1}'.format(index_bus_b,index_bus_a))
                self.ThetaMinByEdge[(index_bus_b,index_bus_a)] = max(self.ThetaMinByEdge[(index_bus_b,index_bus_a)],region.phimin)
                self.ThetaMaxByEdge[(index_bus_b,index_bus_a)] = min(self.ThetaMaxByEdge[(index_bus_b,index_bus_a)],region.phimax)
                #Switching indices to have (a,b,h) \in L
                aux = index_bus_b
                index_bus_b = index_bus_a
                index_bus_a = aux
                z = -np.conj(self.Ytt[line]/self.Ytf[line])
                l = self.Vmin[index_bus_b]/self.Vmax[index_bus_a]
                u = self.Vmax[index_bus_b]/self.Vmin[index_bus_a]
                R = self.Imax[idx_line]/(abs(self.Ytf[line])*self.Vmin[index_bus_b]*self.Vmin[index_bus_a])
                region = PhaseDiffBound.PhaseDiffBound(z,l,u,R)
                #region.plot(self.name+'_{0}_{1}'.format(index_bus_b,index_bus_a))
                self.ThetaMinByEdge[(index_bus_b,index_bus_a)] = max(self.ThetaMinByEdge[(index_bus_b,index_bus_a)],region.phimin)
                self.ThetaMaxByEdge[(index_bus_b,index_bus_a)] = min(self.ThetaMaxByEdge[(index_bus_b,index_bus_a)],region.phimax)
            
        for i,j in self.SymEdgesNoDiag:
            self.ThetaMinByEdge[(i,j)] = max(self.ThetaMinByEdge[(i,j)], -self.ThetaMaxByEdge[(j,i)])
            self.ThetaMaxByEdge[(i,j)] = min(self.ThetaMaxByEdge[(i,j)], -self.ThetaMinByEdge[(j,i)])
        
        for idx_clique in range(len(self.cliques)):
            self.FloydWarshallOnClique(idx_clique)
        
        for i,j in self.SymEdgesNoDiag:
            self.ThetaMinByEdge[(i,j)] = max(self.ThetaMinByEdge[(i,j)], -self.ThetaMaxByEdge[(j,i)])
            self.ThetaMaxByEdge[(i,j)] = min(self.ThetaMaxByEdge[(i,j)], -self.ThetaMinByEdge[(j,i)])
        
    def FloydWarshallOnClique(self,idx_clique):
        cl = self.cliques[idx_clique]
        for k in cl:
            for i in cl:
                for j in cl:
                    if k!=i and k!=j and i!=j:
                        self.ThetaMaxByEdge[(i,j)] = min(self.ThetaMaxByEdge[(i,j)],self.ThetaMaxByEdge[(i,k)]+self.ThetaMaxByEdge[(k,j)])
                        self.ThetaMinByEdge[(i,j)] = -min(-self.ThetaMinByEdge[(i,j)],-self.ThetaMinByEdge[(i,k)]-self.ThetaMinByEdge[(k,j)])
                
class sparseACOPFinstance():
    
    def __init__(self, ACOPF_instance):
        self.name = ACOPF_instance.name
        self.baseMVA = ACOPF_instance.baseMVA
        self.n, self.gn, self.m, self.cl = ACOPF_instance.n, ACOPF_instance.gn, ACOPF_instance.gn, ACOPF_instance.cl
        self.Vmin, self.Vmax = ACOPF_instance.Vmin, ACOPF_instance.Vmax
        self.Pmin,self.Pmax,self.Qmin, self.Qmax = ACOPF_instance.Pmin,ACOPF_instance.Pmax,ACOPF_instance.Qmin, ACOPF_instance.Qmax
        self.offset, self.lincost, self.quadcost = ACOPF_instance.offset, np.array(ACOPF_instance.lincost), ACOPF_instance.quadcost
        self.buslist, self.buslistinv,self.genlist = ACOPF_instance.buslist, ACOPF_instance.buslistinv, ACOPF_instance.genlist
        self.clinelist,self.clinelistinv = ACOPF_instance.clinelist,ACOPF_instance.clinelistinv
        self.cliques, self.ncliques, self.cliques_nbr = ACOPF_instance.cliques, ACOPF_instance.ncliques, ACOPF_instance.cliques_nbr
        self.cliques_parent, self.cliques_intersection, self.localBusIdx = ACOPF_instance.cliques_parent, ACOPF_instance.cliques_intersection, ACOPF_instance.localBusIdx
        self.Pload, self.Qload = np.array(ACOPF_instance.Pload), np.array(ACOPF_instance.Qload) 
        self.M = ACOPF_instance.M
        self.Imax = ACOPF_instance.Imax
        self.N, self.C = ACOPF_instance.N, ACOPF_instance.C
        self.busType = ACOPF_instance.busType
        self.A = ACOPF_instance.A
        self.status = ACOPF_instance.status
        self.SVM = ACOPF_instance.SVM
        self.ThetaMinByEdge, self.ThetaMaxByEdge = ACOPF_instance.ThetaMinByEdge, ACOPF_instance.ThetaMaxByEdge
        
        self.bus_to_gen = {}
        for idx in range(self.n):
            self.bus_to_gen[idx] = []
        for idx_gen,gen in enumerate(self.genlist):
            bus,index = self.genlist[idx_gen]
            index_bus =  self.buslistinv[bus]
            self.bus_to_gen[index_bus].append(idx_gen)
                        
       ##Construct m_cb matrices
        self.M = {}
        #Parts of M related to the lines #self.M[bus] = lil_matrix((self.n,self.n),dtype = np.complex128)
        for (b,a,h) in ACOPF_instance.Yff:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
            cliqueff = ACOPF_instance.edges_to_clique[(i,j)]
            if not((cliqueff,index_bus_b) in self.M):
                nc = self.ncliques[cliqueff]
                self.M[cliqueff,index_bus_b] = lil_matrix((nc,nc),dtype = np.complex128)
            local_index_bus_b, local_index_bus_a = self.localBusIdx[cliqueff,index_bus_b],self.localBusIdx[cliqueff,index_bus_a]
            assert(local_index_bus_b!=local_index_bus_a)
            self.M[cliqueff,index_bus_b][local_index_bus_b,local_index_bus_b] += ACOPF_instance.Yff[(b,a,h)]
            
                
        del cliqueff, local_index_bus_b, local_index_bus_a
        for (b,a,h) in ACOPF_instance.Yft:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
            cliqueft = ACOPF_instance.edges_to_clique[(i,j)]
            if not((cliqueft,index_bus_b) in self.M):
                nc = self.ncliques[cliqueft]
                self.M[cliqueft, index_bus_b] = lil_matrix((nc,nc),dtype = np.complex128)
            local_index_bus_b, local_index_bus_a = self.localBusIdx[cliqueft,index_bus_b],self.localBusIdx[cliqueft,index_bus_a]
            assert(local_index_bus_b!=local_index_bus_a)
            self.M[cliqueft,index_bus_b][local_index_bus_b,local_index_bus_a] += ACOPF_instance.Yft[(b,a,h)]
            
                
        del cliqueft, local_index_bus_b, local_index_bus_a
        for (a,b,h) in ACOPF_instance.Ytt:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
            cliquett = ACOPF_instance.edges_to_clique[(i,j)]
            if not((cliquett, index_bus_b) in self.M):
                nc= self.ncliques[cliquett]
                self.M[cliquett,index_bus_b] =lil_matrix((nc,nc),dtype = np.complex128)
            local_index_bus_b, local_index_bus_a = self.localBusIdx[cliquett,index_bus_b],self.localBusIdx[cliquett,index_bus_a]
            assert(local_index_bus_b!=local_index_bus_a)
            self.M[cliquett,index_bus_b][local_index_bus_b, local_index_bus_b] += ACOPF_instance.Ytt[(a,b,h)]
            
        del cliquett, local_index_bus_b, local_index_bus_a
        for (a,b,h) in ACOPF_instance.Ytf:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
            cliquetf = ACOPF_instance.edges_to_clique[(i,j)]
            if not((cliquetf,index_bus_b) in self.M):
                nc= self.ncliques[cliquetf]
                self.M[cliquetf,index_bus_b] = lil_matrix((nc,nc),dtype = np.complex128)
            local_index_bus_b, local_index_bus_a = self.localBusIdx[cliquetf,index_bus_b],self.localBusIdx[cliquetf,index_bus_a]
            assert(local_index_bus_b!=local_index_bus_a)
            self.M[cliquetf,index_bus_b][local_index_bus_b,local_index_bus_a] += ACOPF_instance.Ytf[(a,b,h)]
            
        del cliquetf, local_index_bus_b, local_index_bus_a, index_bus_b
        #Parts of M related to the shunts
        aux,test_sum = {},0
        for clique,index_bus in self.M:
            if not(index_bus in aux):
                test_sum+=1
                aux[index_bus] = 1
                local_index_bus = self.localBusIdx[clique,index_bus]
                self.M[clique,index_bus][local_index_bus,local_index_bus] += ACOPF_instance.A[self.buslist[index_bus]]
        assert(test_sum==self.n)
        del aux, test_sum, clique,local_index_bus
        
               
        #Conversion to csc_matrices
        for couple in self.M:
            self.M[couple] = self.M[couple].tocsc()
            
        
        self.HM, self.ZM, self.assigned_buses, self.assigned_lines = {} , {},{},{}
        for idx_clique in range(self.cliques_nbr):
            self.assigned_buses[idx_clique] = set()
            self.assigned_lines[idx_clique] = set()
        del idx_clique
            
        for couple in self.M:
            self.HM[couple] = 0.5 * (self.M[couple]+(self.M[couple]).H)
            self.ZM[couple] = 0.5 * (self.M[couple]-(self.M[couple]).H)
            clique,idx_bus = couple
            self.assigned_buses[clique].add(idx_bus)
        del clique,idx_bus
        self.Nf = {}
        self.Nt = {}
        
        #Build Nf and Nt matrices
        if ACOPF_instance.config["lineconstraints"]=='I':
            
            for idx_line,line in enumerate(ACOPF_instance.clinelistinv):
                b,a,h = line
                index_bus_b,index_bus_a = ACOPF_instance.buslistinv[b],ACOPF_instance.buslistinv[a]
                i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
                clique = ACOPF_instance.edges_to_clique[(i,j)]
                nc = self.ncliques[clique]
                local_index_bus_b,local_index_bus_a = self.localBusIdx[clique,index_bus_b],self.localBusIdx[clique,index_bus_a]
                assert(local_index_bus_b!=local_index_bus_a)
                self.assigned_lines[clique].add(idx_line)
                #Build Nf line matrix
                self.Nf[clique,idx_line] = lil_matrix((nc,nc),dtype = np.complex128)
                self.Nf[clique,idx_line][local_index_bus_b,local_index_bus_b] = np.conj(ACOPF_instance.Yff[line]) * ACOPF_instance.Yff[line]
                self.Nf[clique,idx_line][local_index_bus_a,local_index_bus_b] = np.conj(ACOPF_instance.Yft[line]) * ACOPF_instance.Yff[line]
                self.Nf[clique,idx_line][local_index_bus_b,local_index_bus_a] = np.conj(ACOPF_instance.Yff[line]) * ACOPF_instance.Yft[line]
                self.Nf[clique,idx_line][local_index_bus_a,local_index_bus_a] = np.conj(ACOPF_instance.Yft[line]) * ACOPF_instance.Yft[line]
                
                #Build Nt line matrix
                self.Nt[clique,idx_line] = lil_matrix((nc,nc),dtype = np.complex128)
                self.Nt[clique,idx_line][local_index_bus_b,local_index_bus_b] = np.conj(ACOPF_instance.Ytf[line]) * ACOPF_instance.Ytf[line]
                self.Nt[clique,idx_line][local_index_bus_a,local_index_bus_b] = np.conj(ACOPF_instance.Ytt[line]) * ACOPF_instance.Ytf[line]
                self.Nt[clique,idx_line][local_index_bus_b,local_index_bus_a] = np.conj(ACOPF_instance.Ytf[line]) * ACOPF_instance.Ytt[line]
                self.Nt[clique,idx_line][local_index_bus_a,local_index_bus_a] = np.conj(ACOPF_instance.Ytt[line]) * ACOPF_instance.Ytt[line] 
                


        
    