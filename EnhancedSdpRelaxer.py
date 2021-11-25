# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:12:03 2021

@author: aoust
"""



import itertools
from scipy.sparse import lil_matrix, coo_matrix
import numpy as np
from mosek.fusion import *
myZeroforCosts = 1E-6
#My infty
myinf_power_lim = 1E4

class EnhancedSdpRelaxer():
    
    def __init__(self, ACOPF):
        self.name = ACOPF.name
        self.baseMVA = ACOPF.baseMVA
        self.config = ACOPF.config
        self.n, self.gn, self.m, self.cl = ACOPF.n, ACOPF.gn, ACOPF.gn, ACOPF.cl
        self.Vmin, self.Vmax = ACOPF.Vmin, ACOPF.Vmax
        self.Pmin,self.Pmax,self.Qmin, self.Qmax = ACOPF.Pmin,ACOPF.Pmax,ACOPF.Qmin, ACOPF.Qmax
        self.offset, self.lincost, self.quadcost = ACOPF.offset, np.array(ACOPF.lincost), ACOPF.quadcost
        self.buslist, self.buslistinv,self.genlist = ACOPF.buslist, ACOPF.buslistinv, ACOPF.genlist
        self.cliques, self.ncliques, self.cliques_nbr = ACOPF.cliques, ACOPF.ncliques, ACOPF.cliques_nbr
        self.cliques_parent, self.cliques_intersection, self.localBusIdx = ACOPF.cliques_parent, ACOPF.cliques_intersection, ACOPF.localBusIdx
        self.Pload, self.Qload = np.array(ACOPF.Pload), np.array(ACOPF.Qload) 
        self.M = ACOPF.M
        
        self.cl = ACOPF.cl
        self.clinelist, self.clinelistinv = ACOPF.clinelist, ACOPF.clinelistinv
        self.edges_to_clique = ACOPF.edges_to_clique
        self.Yff, self.Yft, self.Ytf, self.Ytt = ACOPF.Yff, ACOPF.Yft, ACOPF.Ytf, ACOPF.Ytt 
        self.Imax = ACOPF.Imax

        
        self.bus_to_gen = {}
        for idx in range(self.n):
            self.bus_to_gen[idx] = []
        for idx_gen,gen in enumerate(self.genlist):
            bus,index = self.genlist[idx_gen]
            index_bus =  self.buslistinv[bus]
            self.bus_to_gen[index_bus].append(idx_gen)
            
        
        self.ThetaMinByEdge, self.ThetaMaxByEdge = ACOPF.ThetaMinByEdge, ACOPF.ThetaMaxByEdge
        
            
       # #Construct m_cb matrices
        self.M = {}
        #Parts of M related to the lines #self.M[bus] = lil_matrix((self.n,self.n),dtype = np.complex128)
        for (b,a,h) in ACOPF.Yff:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
            cliqueff = ACOPF.edges_to_clique[(i,j)]
            if not((cliqueff,index_bus_b) in self.M):
                nc = self.ncliques[cliqueff]
                self.M[cliqueff,index_bus_b] = lil_matrix((nc,nc),dtype = np.complex128)
            local_index_bus_b, local_index_bus_a = self.localBusIdx[cliqueff,index_bus_b],self.localBusIdx[cliqueff,index_bus_a]
            assert(local_index_bus_b!=local_index_bus_a)
            self.M[cliqueff,index_bus_b][local_index_bus_b,local_index_bus_b] += ACOPF.Yff[(b,a,h)]
            
                
        del cliqueff, local_index_bus_b, local_index_bus_a
        for (b,a,h) in ACOPF.Yft:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
            cliqueft = ACOPF.edges_to_clique[(i,j)]
            if not((cliqueft,index_bus_b) in self.M):
                nc = self.ncliques[cliqueft]
                self.M[cliqueft, index_bus_b] = lil_matrix((nc,nc),dtype = np.complex128)
            local_index_bus_b, local_index_bus_a = self.localBusIdx[cliqueft,index_bus_b],self.localBusIdx[cliqueft,index_bus_a]
            assert(local_index_bus_b!=local_index_bus_a)
            self.M[cliqueft,index_bus_b][local_index_bus_b,local_index_bus_a] += ACOPF.Yft[(b,a,h)]
            
                
        del cliqueft, local_index_bus_b, local_index_bus_a
        for (a,b,h) in ACOPF.Ytt:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
            cliquett = ACOPF.edges_to_clique[(i,j)]
            if not((cliquett, index_bus_b) in self.M):
                nc= self.ncliques[cliquett]
                self.M[cliquett,index_bus_b] =lil_matrix((nc,nc),dtype = np.complex128)
            local_index_bus_b, local_index_bus_a = self.localBusIdx[cliquett,index_bus_b],self.localBusIdx[cliquett,index_bus_a]
            assert(local_index_bus_b!=local_index_bus_a)
            self.M[cliquett,index_bus_b][local_index_bus_b, local_index_bus_b] += ACOPF.Ytt[(a,b,h)]
            
        del cliquett, local_index_bus_b, local_index_bus_a
        for (a,b,h) in ACOPF.Ytf:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
            cliquetf = ACOPF.edges_to_clique[(i,j)]
            if not((cliquetf,index_bus_b) in self.M):
                nc= self.ncliques[cliquetf]
                self.M[cliquetf,index_bus_b] = lil_matrix((nc,nc),dtype = np.complex128)
            local_index_bus_b, local_index_bus_a = self.localBusIdx[cliquetf,index_bus_b],self.localBusIdx[cliquetf,index_bus_a]
            assert(local_index_bus_b!=local_index_bus_a)
            self.M[cliquetf,index_bus_b][local_index_bus_b,local_index_bus_a] += ACOPF.Ytf[(a,b,h)]
            
        del cliquetf, local_index_bus_b, local_index_bus_a, index_bus_b
        #Parts of M related to the shunts
        aux,test_sum = {},0
        for clique,index_bus in self.M:
            if not(index_bus in aux):
                test_sum+=1
                aux[index_bus] = 1
                local_index_bus = self.localBusIdx[clique,index_bus]
                self.M[clique,index_bus][local_index_bus,local_index_bus] += ACOPF.A[self.buslist[index_bus]]
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
        if self.config["lineconstraints"]=='I':
            
            for idx_line,line in enumerate(ACOPF.clinelistinv):
                b,a,h = line
                index_bus_b,index_bus_a = ACOPF.buslistinv[b],ACOPF.buslistinv[a]
                i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
                clique = ACOPF.edges_to_clique[(i,j)]
                nc = self.ncliques[clique]
                local_index_bus_b,local_index_bus_a = self.localBusIdx[clique,index_bus_b],self.localBusIdx[clique,index_bus_a]
                assert(local_index_bus_b!=local_index_bus_a)
                self.assigned_lines[clique].add(idx_line)
                #Build Nf line matrix
                self.Nf[clique,idx_line] = lil_matrix((nc,nc),dtype = np.complex128)
                self.Nf[clique,idx_line][local_index_bus_b,local_index_bus_b] = np.conj(ACOPF.Yff[line]) * ACOPF.Yff[line]
                self.Nf[clique,idx_line][local_index_bus_a,local_index_bus_b] = np.conj(ACOPF.Yft[line]) * ACOPF.Yff[line]
                self.Nf[clique,idx_line][local_index_bus_b,local_index_bus_a] = np.conj(ACOPF.Yff[line]) * ACOPF.Yft[line]
                self.Nf[clique,idx_line][local_index_bus_a,local_index_bus_a] = np.conj(ACOPF.Yft[line]) * ACOPF.Yft[line]
                
                #Build Nt line matrix
                self.Nt[clique,idx_line] = lil_matrix((nc,nc),dtype = np.complex128)
                self.Nt[clique,idx_line][local_index_bus_b,local_index_bus_b] = np.conj(ACOPF.Ytf[line]) * ACOPF.Ytf[line]
                self.Nt[clique,idx_line][local_index_bus_a,local_index_bus_b] = np.conj(ACOPF.Ytt[line]) * ACOPF.Ytf[line]
                self.Nt[clique,idx_line][local_index_bus_b,local_index_bus_a] = np.conj(ACOPF.Ytf[line]) * ACOPF.Ytt[line]
                self.Nt[clique,idx_line][local_index_bus_a,local_index_bus_a] = np.conj(ACOPF.Ytt[line]) * ACOPF.Ytt[line] 
                

        self.cliques_contribution = {}
        for index_bus in range(self.n):
            self.cliques_contribution[index_bus] = set()
       
        for couple in self.M:
            clique,index_bus = couple
            self.cliques_contribution[index_bus].add(clique)
    
        self.preprocessing_power_bounds()
    def Joperator(self,matrix):
        A = np.real(matrix)
        B = np.imag(matrix)
        line1 = np.hstack([A, -B])
        line2 = np.hstack([B, A])
        return (1/(np.sqrt(2)))*np.vstack([line1,line2])
    
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

                    
    def computeDuals(self):
        """
        Method to solve the real formulation of the rank relaxation

        Returns
        -------
        None.

        """
        scale = 0.001
        with Model("OPF-rank-relaxation") as M:
        
            #Upper level var
            Pgen = M.variable("Pgen", self.gn, Domain.unbounded())
            aux = M.variable("aux", self.gn+2, Domain.inRotatedQCone())
            Qgen = M.variable("Qgen", self.gn, Domain.unbounded())
            
            #Objective
            M.objective( ObjectiveSense.Minimize, Expr.add(Expr.add(scale*self.offset,Expr.dot(scale*self.lincost,Pgen)),Expr.mul(2,aux.index(1))))
            M.constraint(aux.index(0), Domain.equalsTo(1))
            M.constraint(Expr.sub(aux.pick(range(2,self.gn+2)),Expr.mulElm([np.sqrt(scale*cost) for cost in self.quadcost],Pgen)), Domain.equalsTo(0,self.gn))
            
             # # #Active Power bounds
            M.constraint(Pgen,Domain.greaterThan(np.array([self.Pmin[idx] for idx in range(self.gn)])))
            M.constraint(Pgen,Domain.lessThan(np.array([self.Pmax[idx] for idx in range(self.gn)])))
            
            # # #Reactive Power bounds
            M.constraint(Qgen,Domain.greaterThan(np.array([self.Qmin[idx] for idx in range(self.gn)])))
            M.constraint(Qgen,Domain.lessThan(np.array([self.Qmax[idx] for idx in range(self.gn)])))
            
            X,A,B,R = {},{},{},{}
            
            already_covered = set()
            for idx_clique in range(self.cliques_nbr):
                nc = self.ncliques[idx_clique]
                clique = self.cliques[idx_clique]
                X[idx_clique] = M.variable(Domain.inPSDCone(2*nc))
                R[idx_clique] = M.variable("R"+str(idx_clique), [nc+1,nc+1], Domain.unbounded())
                A[idx_clique] = M.variable("A"+str(idx_clique), [nc,nc], Domain.unbounded())
                B[idx_clique] = M.variable("B"+str(idx_clique), [nc,nc], Domain.unbounded())
                # # #Voltage bounds
                M.constraint(A[idx_clique].diag(),Domain.greaterThan(np.array([self.Vmin[idx]**2 for idx in clique])))
                M.constraint(A[idx_clique].diag(),Domain.lessThan(np.array([self.Vmax[idx]**2 for idx in clique])))
            
                #Link between isometry matrix X and matrices A and B
                M.constraint(Expr.sub(Expr.mul(1/np.sqrt(2),A[idx_clique]),X[idx_clique].slice([0,0], [nc,nc])), Domain.equalsTo(0,nc,nc))
                M.constraint(Expr.sub(Expr.mul(1/np.sqrt(2),A[idx_clique]),X[idx_clique].slice([nc,nc], [2*nc,2*nc])), Domain.equalsTo(0,nc,nc))
                M.constraint(Expr.sub(Expr.mul(1/np.sqrt(2),B[idx_clique]),X[idx_clique].slice([nc,0], [2*nc,nc])), Domain.equalsTo(0,nc,nc))
                M.constraint(Expr.add(Expr.mul(1/np.sqrt(2),B[idx_clique]),X[idx_clique].slice([0,nc], [nc,2*nc])), Domain.equalsTo(0,nc,nc))
                
                #R[idx_clique][0,0] = 1
                M.constraint(R[idx_clique].index(0,0),Domain.equalsTo(1.0))
                
                for i in range(nc):
                    b = clique[i]
                    M.constraint(Expr.sub(A[idx_clique].index(i,i),R[idx_clique].index(1+i,1+i)),Domain.equalsTo(0))
                    M.constraint(R[idx_clique].index(0,1+i),Domain.greaterThan(self.Vmin[b]))
                    M.constraint(R[idx_clique].index(0,1+i),Domain.lessThan(self.Vmax[b]))
                    slope = ((self.Vmax[b] - self.Vmin[b])/(self.Vmax[b]**2 - self.Vmin[b]**2))
                    M.constraint(Expr.sub(R[idx_clique].index(0,1+i),Expr.mul(slope,A[idx_clique].index(i,i))), Domain.greaterThan(self.Vmin[b] - (self.Vmin[b]**2)*slope))
                    for j in range(nc):
                        
                        if i<j and not((clique[i],clique[j]) in already_covered):
                            assert(clique[i]<clique[j])
                            b,a = clique[i],clique[j]
                            already_covered.add((clique[i],clique[j]))
                            phimin,phimax = self.ThetaMinByEdge[(clique[i],clique[j])],self.ThetaMaxByEdge[(clique[i],clique[j])]
                            if phimax-phimin<=np.pi:
                                M.constraint(Expr.add(Expr.mul(-np.sin(phimin),A[idx_clique].index(i,j)),Expr.mul(np.cos(phimin),B[idx_clique].index(i,j))),Domain.greaterThan(0))
                                M.constraint(Expr.add(Expr.mul(-np.sin(phimax),A[idx_clique].index(i,j)),Expr.mul(np.cos(phimax),B[idx_clique].index(i,j))),Domain.lessThan(0))
                                halfdiff =  0.5*(phimax- phimin)
                                mean =  0.5*(phimax + phimin)
                                M.constraint(Expr.sub(Expr.add(Expr.mul(np.cos(mean),A[idx_clique].index(i,j)),Expr.mul(np.sin(mean),B[idx_clique].index(i,j))), Expr.mul(np.cos(halfdiff),R[idx_clique].index(1+i,1+j))),Domain.greaterThan(0))
            
                            M.constraint(Expr.sub(R[idx_clique].index(1+i,1+j),Expr.add(Expr.mul(self.Vmin[b], R[idx_clique].index(0,1+j)), Expr.mul( self.Vmin[a], R[idx_clique].index(0,1+i)))),Domain.greaterThan(- self.Vmin[b]*self.Vmin[a]))
                            M.constraint(Expr.sub(R[idx_clique].index(1+i,1+j),Expr.add(Expr.mul(self.Vmax[b], R[idx_clique].index(0,1+j)), Expr.mul( self.Vmax[a], R[idx_clique].index(0,1+i)))),Domain.greaterThan(- self.Vmax[b]*self.Vmax[a]))
                            M.constraint(Expr.sub(R[idx_clique].index(1+i,1+j),Expr.add(Expr.mul(self.Vmax[b], R[idx_clique].index(0,1+j)), Expr.mul( self.Vmin[a], R[idx_clique].index(0,1+i)))),Domain.lessThan(- self.Vmax[b]*self.Vmin[a]))
                            M.constraint(Expr.sub(R[idx_clique].index(1+i,1+j),Expr.add(Expr.mul(self.Vmin[b], R[idx_clique].index(0,1+j)), Expr.mul( self.Vmax[a], R[idx_clique].index(0,1+i)))),Domain.lessThan(- self.Vmin[b]*self.Vmax[a]))
                            
                            #Constraint |W_ba| \leq R_{ba}
                            M.constraint(Expr.vstack(R[idx_clique].index(1+i,1+j),A[idx_clique].index(i,j),B[idx_clique].index(i,j)), Domain.inQCone(3))
            
            #Active and Reactive Power conservation
            for index_bus in range(self.n):
                sumPgen = Expr.zeros(1)
                sumQgen = Expr.zeros(1)
                for i in self.bus_to_gen[index_bus]:
                    sumPgen = Expr.add(sumPgen, Pgen.index(i))
                    sumQgen = Expr.add(sumQgen, Qgen.index(i))
                
                #JHMbus, JiZMbus = {},{}
                Ptransfer, Qtransfer = Expr.zeros(1),Expr.zeros(1)
                
                for idx_clique in self.cliques_contribution[index_bus]:
                    nc = self.ncliques[idx_clique]
                    auxHM = self.Joperator(self.HM[idx_clique,index_bus].toarray())
                    auxiZHM = self.Joperator(1j*(self.ZM[idx_clique,index_bus]).toarray())
                    auxHM = coo_matrix(auxHM)
                    auxHM.eliminate_zeros()
                    auxiZHM = coo_matrix(auxiZHM)
                    auxiZHM.eliminate_zeros()
                    JHMbus = Matrix.sparse(2*nc,2*nc,auxHM.row, auxHM.col, auxHM.data)
                    JiZMbus = Matrix.sparse(2*nc,2*nc,auxiZHM.row, auxiZHM.col, auxiZHM.data)
                    Ptransfer = Expr.add(Ptransfer,Expr.dot(JHMbus,X[idx_clique]))   
                    Qtransfer = Expr.add(Qtransfer,Expr.dot(JiZMbus,X[idx_clique]))  
                if len(self.bus_to_gen[index_bus])>0:
                    M.constraint(Expr.sub(sumPgen,Ptransfer),Domain.equalsTo(self.Pload[index_bus]))
                    M.constraint(Expr.sub(sumQgen,Qtransfer),Domain.equalsTo(self.Qload[index_bus]))
                else:
                    M.constraint(Expr.neg(Ptransfer),Domain.equalsTo(self.Pload[index_bus]))
                    M.constraint(Expr.neg(Qtransfer),Domain.equalsTo(self.Qload[index_bus]))                   
            
            #Lines intensity constraints
            if self.config['lineconstraints']=='I':
                for idx_clique, idx_line in self.Nt:
                    nc = self.ncliques[idx_clique]
                    Nf = self.Joperator(self.Nf[idx_clique,idx_line].toarray())
                    Nf = coo_matrix(Nf)
                    Nf.eliminate_zeros()
                    Nf = Matrix.sparse(2*nc,2*nc,Nf.row, Nf.col, Nf.data)
                    Nt = self.Joperator(self.Nt[idx_clique,idx_line].toarray())
                    Nt = coo_matrix(Nt)
                    Nt.eliminate_zeros()
                    Nt = Matrix.sparse(2*nc,2*nc,Nt.row, Nt.col, Nt.data)
                    M.constraint(Expr.dot(Nf,X[idx_clique]), Domain.lessThan((self.Imax[idx_line]**2)))
                    M.constraint(Expr.dot(Nt,X[idx_clique]), Domain.lessThan((self.Imax[idx_line]**2)))
            else:
                assert(self.config['lineconstraints']=='S')
                for idx_line,line in enumerate(self.clinelistinv):
                    b,a,h = line
                    index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
                    i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
                    clique = self.edges_to_clique[(i,j)]
                    nc = self.ncliques[clique]
                    local_index_bus_b,local_index_bus_a = self.localBusIdx[clique,index_bus_b],self.localBusIdx[clique,index_bus_a]
                    rex = Expr.add(Expr.mul(np.real(self.Yff[line]),A[clique].index(local_index_bus_b,local_index_bus_b)),Expr.add(Expr.mul(np.real(self.Yft[line]),A[clique].index(local_index_bus_b,local_index_bus_a)),Expr.mul(np.imag(self.Yft[line]),B[clique].index(local_index_bus_b,local_index_bus_a))))
                    imx = Expr.add(Expr.mul(-np.imag(self.Yff[line]),A[clique].index(local_index_bus_b,local_index_bus_b)),Expr.add(Expr.mul(-np.imag(self.Yft[line]),A[clique].index(local_index_bus_b,local_index_bus_a)),Expr.mul(np.real(self.Yft[line]),B[clique].index(local_index_bus_b,local_index_bus_a))))
                    M.constraint(Expr.vstack(self.Imax[idx_line],rex,imx), Domain.inQCone(3))
                    #Switching indices to have (a,b,h) \in L
                    aux = local_index_bus_b
                    local_index_bus_b = local_index_bus_a
                    local_index_bus_a = aux
                    rex = Expr.add(Expr.mul(np.real(self.Ytt[line]),A[clique].index(local_index_bus_b,local_index_bus_b)),Expr.add(Expr.mul(np.real(self.Ytf[line]),A[clique].index(local_index_bus_b,local_index_bus_a)),Expr.mul(np.imag(self.Ytf[line]),B[clique].index(local_index_bus_b,local_index_bus_a))))
                    imx = Expr.add(Expr.mul(-np.imag(self.Ytt[line]),A[clique].index(local_index_bus_b,local_index_bus_b)),Expr.add(Expr.mul(-np.imag(self.Ytf[line]),A[clique].index(local_index_bus_b,local_index_bus_a)),Expr.mul(np.real(self.Ytf[line]),B[clique].index(local_index_bus_b,local_index_bus_a))))
                    M.constraint(Expr.vstack(self.Imax[idx_line],rex,imx), Domain.inQCone(3))
                    
            
            #Overlapping constraints for W
            for clique_idx in range(self.cliques_nbr):
                nc = self.ncliques[clique_idx]
                clique_father_idx = self.cliques_parent[clique_idx]
                for global_idx_bus_b in self.cliques_intersection[clique_idx]:
                    local_index_bus_b = self.localBusIdx[clique_idx,global_idx_bus_b]
                    local_index_bus_b_father = self.localBusIdx[clique_father_idx,global_idx_bus_b]
                    M.constraint(Expr.sub(A[clique_idx].index(local_index_bus_b,local_index_bus_b),A[clique_father_idx].index(local_index_bus_b_father,local_index_bus_b_father)), Domain.equalsTo(0.0))
                for global_idx_bus_b,global_idx_bus_a in itertools.combinations(self.cliques_intersection[clique_idx], 2):
                    local_index_bus_b,local_index_bus_a = self.localBusIdx[clique_idx,global_idx_bus_b],self.localBusIdx[clique_idx,global_idx_bus_a]
                    local_index_bus_b_father,local_index_bus_a_father = self.localBusIdx[clique_father_idx,global_idx_bus_b],self.localBusIdx[clique_father_idx,global_idx_bus_a]
                    M.constraint(Expr.sub(A[clique_idx].index(local_index_bus_b,local_index_bus_a),A[clique_father_idx].index(local_index_bus_b_father,local_index_bus_a_father)), Domain.equalsTo(0.0))
                    M.constraint(Expr.sub(B[clique_idx].index(local_index_bus_b,local_index_bus_a),B[clique_father_idx].index(local_index_bus_b_father,local_index_bus_a_father)), Domain.equalsTo(0.0))
            
            #Overlapping constraints for R
            for clique_idx in range(self.cliques_nbr):
                nc = self.ncliques[clique_idx]
                clique_father_idx = self.cliques_parent[clique_idx]
                for global_idx_bus_b in self.cliques_intersection[clique_idx]:
                    local_index_bus_b = self.localBusIdx[clique_idx,global_idx_bus_b]
                    local_index_bus_b_father = self.localBusIdx[clique_father_idx,global_idx_bus_b]
                    M.constraint(Expr.sub(R[clique_idx].index(1+local_index_bus_b,1+local_index_bus_b),R[clique_father_idx].index(1+local_index_bus_b_father,1+local_index_bus_b_father)), Domain.equalsTo(0.0))
                    M.constraint(Expr.sub(R[clique_idx].index(0,1+local_index_bus_b),R[clique_father_idx].index(0,1+local_index_bus_b_father)), Domain.equalsTo(0.0))
                for global_idx_bus_b,global_idx_bus_a in itertools.combinations(self.cliques_intersection[clique_idx], 2):
                    local_index_bus_b,local_index_bus_a = self.localBusIdx[clique_idx,global_idx_bus_b],self.localBusIdx[clique_idx,global_idx_bus_a]
                    local_index_bus_b_father,local_index_bus_a_father = self.localBusIdx[clique_father_idx,global_idx_bus_b],self.localBusIdx[clique_father_idx,global_idx_bus_a]
                    M.constraint(Expr.sub(R[clique_idx].index(1+local_index_bus_b,1+local_index_bus_a),R[clique_father_idx].index(1+local_index_bus_b_father,1+local_index_bus_a_father)), Domain.equalsTo(0.0))
    
            M.acceptedSolutionStatus(AccSolutionStatus.Anything)
            M.setSolverParam("intpntCoTolPfeas", 1.0e-8)
            M.setSolverParam("intpntCoTolDfeas", 1.0e-8)
            M.setSolverParam("intpntSolveForm", "dual")
            M.solve()
            value = M.dualObjValue()/scale
            print("SDP Dual Objective value ={0}".format(M.dualObjValue()*1/scale))
            
            res,res2 = [],[]
            for idx_clique in range(self.cliques_nbr):
                nc = self.ncliques[idx_clique]
                bigXdual = (X[idx_clique].dual()).reshape((2*nc,2*nc))
                
                res.append(bigXdual[:nc,:nc] +1j * (bigXdual[nc:2*nc,:nc]))
                res2.append((R[idx_clique].dual()).reshape((1+nc,1+nc)))
            return value,res, res2
        

 
