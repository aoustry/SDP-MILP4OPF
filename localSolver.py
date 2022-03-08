# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:33:56 2022

@author: aoust
"""

from pyomo.environ import ConcreteModel, RangeSet, Var,SolverFactory,value, cos, sin,Param,Objective, Constraint, ConstraintList,minimize
import instance
import numpy as np

lineconstraints = 'S'


def linstance(name_instance):
    np.random.seed(10)
    instance_config = {"lineconstraints" : lineconstraints,  "cliques_strategy":"ASP"}
    Instance = instance.ACOPFinstance("data/pglib-opf/{0}.m".format(name_instance),name_instance,instance_config)
    return Instance


class localACOPFsolver():
    """Class based on Pyomo Concrete Model, for locally solving the ACOPF problem."""
    
    def __init__(self, ACOPF):
        self.name = ACOPF.name
        self.config = {}
        
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
        
        self.edges, self.SymEdgesNoDiag  = ACOPF.edges,ACOPF.SymEdgesNoDiag
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
        self.Yff, self.Yft, self.Ytf, self.Ytt = ACOPF.Yff, ACOPF.Yft, ACOPF.Ytf, ACOPF.Ytt
        
        if self.config['lineconstraints']=='I':
            self.Nf, self.Nt = ACOPF.Nf, ACOPF.Nt
        
        self.value,self.success = np.inf, 0
        self.build_model()
            
    def __vmax_function(self,model, i):
        return self.Vmax[i]
    
    def __vmin_function(self,model, i):
        return self.Vmin[i]
    
    def __pmax_function(self,model, i):
        return self.Pmax[i]
    
    def __pmin_function(self,model, i):
        return self.Pmin[i]
    
    def __qmax_function(self,model, i):
        return self.Qmax[i]
    
    def __qmin_function(self,model, i):
        return self.Qmin[i]
    
    def __tmax_function(self,model, b,a):
        return self.ThetaMaxByEdge[(b,a)]
    
    def __tmin_function(self,model, b,a):
        return self.ThetaMinByEdge[(b,a)]
    
    def __feasible(self,pgen,qgen,V):
        tol, res = 1e-6, True
        for i in range(self.gn):
            res = res and (pgen[i]<= self.Pmax[i] + tol)
            res = res and (pgen[i]>= self.Pmin[i] - tol)
            res = res and (qgen[i]<= self.Qmax[i]+tol)
            res = res and (qgen[i]>= self.Qmin[i]-tol)
        for i in range(self.n):
            res = res and (np.abs(V[i])<= self.Vmax[i]+tol)
            res = res and (np.abs(V[i])>= self.Vmin[i]-tol)
        
        #Power conservation constraints
        for idx_bus in range(self.n):
            row, col = self.HM[idx_bus].nonzero()
            dicoHmb = {(row[aux],col[aux]):(self.HM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            pmis = np.abs(sum(pgen[idx_gen] for idx_gen in self.bus_to_gen[idx_bus])-self.Pload[idx_bus]-sum(V[j]*np.conj(V[i])*dicoHmb[i,j] for i,j in dicoHmb))
            row, col = self.ZM[idx_bus].nonzero()
            dicoZmb = {(row[aux],col[aux]):(1j*self.ZM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            qmis=  np.abs(sum(qgen[idx_gen] for idx_gen in self.bus_to_gen[idx_bus])-self.Qload[idx_bus]-sum(V[j]*np.conj(V[i])*dicoZmb[i,j] for i,j in dicoZmb))
            res = res and pmis<tol and qmis<tol
        

        
        if self.config['lineconstraints']=='I':
            for idx_line,line in enumerate(self.clinelistinv):
                b,a,h = line
                index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
                I1 = self.Yff[line]*V[index_bus_b] + self.Yft[line]*V[index_bus_a]
                res = res and (np.abs(I1)<=self.Imax[idx_line]+tol)
                aux = index_bus_b
                index_bus_b = index_bus_a
                index_bus_a = aux
                I2 = self.Ytt[line]*V[index_bus_b] + self.Ytf[line]*V[index_bus_a]
                res = res and (np.abs(I2)<=self.Imax[idx_line]+tol)
              
        
        elif self.config['lineconstraints']=='S':
            for idx_line,line in enumerate(self.clinelistinv):
                b,a,h = line
                index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
                S1 = np.conj(self.Yff[line])*V[index_bus_b]*np.conj(V[index_bus_b]) + np.conj(self.Yft[line])*V[index_bus_b]*np.conj(V[index_bus_a])
                res = res and (np.abs(S1)<=self.Imax[idx_line]+tol)
                #Switching indices to have (a,b,h) \in L
                aux = index_bus_b
                index_bus_b = index_bus_a
                index_bus_a = aux
                S2 = np.conj(self.Ytt[line])*V[index_bus_b]*np.conj(V[index_bus_b]) + np.conj(self.Ytf[line])*V[index_bus_b]*np.conj(V[index_bus_a])
                res = res and (np.abs(S2)<=self.Imax[idx_line]+tol)
               
        else:
            assert(self.config['lineconstraints']==False)
        theta = np.angle(V)
        for (index_bus_b,index_bus_a) in self.SymEdgesNoDiag:
            res = res and (theta[index_bus_b] - theta[index_bus_a] >=  self.ThetaMinByEdge[index_bus_b,index_bus_a]-tol)
            res = res and (theta[index_bus_b] - theta[index_bus_a] <=  self.ThetaMaxByEdge[index_bus_b,index_bus_a]+tol)

        return res

    def build_model(self):
        
        self.binaries = False
        self.mdl =  ConcreteModel()
        self.mdl.bus = RangeSet(0,self.n-1)
        self.mdl.generators = RangeSet(0,self.gn-1)
        self.mdl.penal = Param(initialize=0,mutable=True)
        self.mdl.Pref = Param(self.mdl.generators,initialize=0,mutable=True)
        self.mdl.Vref = Param(self.mdl.bus,initialize=1,mutable=True)
        self.mdl.targetThetaByEdge = Param(self.SymEdgesNoDiag,initialize=0,mutable=True)
        self.mdl.mutableVmax = Param(self.mdl.bus,initialize=self.__vmax_function,mutable=True)
        self.mdl.mutableVmin = Param(self.mdl.bus,initialize=self.__vmin_function,mutable=True)
        self.mdl.mutablePmax = Param(self.mdl.generators,initialize=self.__pmax_function,mutable=True)
        self.mdl.mutablePmin = Param(self.mdl.generators,initialize=self.__pmin_function,mutable=True)
        self.mdl.mutableQmax = Param(self.mdl.generators,initialize=self.__qmax_function,mutable=True)
        self.mdl.mutableQmin = Param(self.mdl.generators,initialize=self.__qmin_function,mutable=True)
        self.mdl.mutableThetaMaxByEdge = Param(self.SymEdgesNoDiag,initialize=self.__tmax_function,mutable=True)
        self.mdl.mutableThetaMinByEdge = Param(self.SymEdgesNoDiag,initialize=self.__tmin_function,mutable=True)
        ######################################### Variables ######################################
        #Gen variables
        self.mdl.Pgen = Var(range(self.gn))
      
        self.mdl.Qgen = Var(range(self.gn))
        #Bus variables
        self.mdl.theta = Var(range(self.n),initialize=0)
        #Bus and edges variables
        self.mdl.VM = Var(range(self.n),initialize= 1)
        ######################################### Objective function ################################################
        penal_expr = sum( (self.mdl.VM[i]-self.mdl.Vref[i])**2 for i in range(self.n))+sum( (self.mdl.Pgen[i]-self.mdl.Pref[i])**2 for i in range(self.gn)) + sum( (self.mdl.theta[b] - self.mdl.theta[a] - self.mdl.targetThetaByEdge[(b,a)])**2 for (b,a) in self.edges)
        self.mdl.objective = Objective(expr = self.offset+sum(self.mdl.Pgen[i] * self.lincost[i]+(self.mdl.Pgen[i]**2) * self.quadcost[i] for i in range(self.gn)) + self.mdl.penal * penal_expr, sense=minimize)
        ######################################### Linear constraints defining F ######################################
        self.mdl.bounds = ConstraintList()
        for i in range(self.gn):
            self.mdl.bounds.add(self.mdl.Pgen[i]<= self.mdl.mutablePmax[i])
            self.mdl.bounds.add(self.mdl.Pgen[i]>= self.mdl.mutablePmin[i])
            self.mdl.bounds.add(self.mdl.Qgen[i]<= self.mdl.mutableQmax[i])
            self.mdl.bounds.add(self.mdl.Qgen[i]>= self.mdl.mutableQmin[i])
        for i in range(self.n):
            self.mdl.bounds.add(self.mdl.VM[i]<= self.mdl.mutableVmax[i])
            self.mdl.bounds.add(self.mdl.VM[i]>= self.mdl.mutableVmin[i])
        self.mdl.ref_node = Constraint(expr=self.mdl.theta[0]==0)

        self.mdl.p_mis, self.mdl.q_mis = ConstraintList(),ConstraintList()
        #Power conservation constraints
        for idx_bus in range(self.n):
            row, col = self.HM[idx_bus].nonzero()
            dicoHmbRe = {(row[aux],col[aux]):np.real(self.HM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            dicoHmbIm = {(row[aux],col[aux]):np.imag(self.HM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            self.mdl.p_mis.add(sum(self.mdl.Pgen[idx_gen] for idx_gen in self.bus_to_gen[idx_bus])==self.Pload[idx_bus]+sum(self.mdl.VM[i]*self.mdl.VM[j]*cos(self.mdl.theta[i]-self.mdl.theta[j])*dicoHmbRe[i,j] for i,j in dicoHmbRe) + sum(self.mdl.VM[i]*self.mdl.VM[j]*sin(self.mdl.theta[i]-self.mdl.theta[j])*dicoHmbIm[i,j] for i,j in dicoHmbIm))
            row, col = self.ZM[idx_bus].nonzero()
            dicoZmbRe = {(row[aux],col[aux]):np.real(1j*self.ZM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            dicoZmbIm = {(row[aux],col[aux]):np.imag(1j*self.ZM[idx_bus][row[aux],col[aux]]) for aux in range(len(row))}
            self.mdl.q_mis.add(sum(self.mdl.Qgen[idx_gen] for idx_gen in self.bus_to_gen[idx_bus])==self.Qload[idx_bus]+sum(self.mdl.VM[i]*self.mdl.VM[j]*cos(self.mdl.theta[i]-self.mdl.theta[j])*dicoZmbRe[i,j] for i,j in dicoZmbRe) + sum(self.mdl.VM[i]*self.mdl.VM[j]*sin(self.mdl.theta[i]-self.mdl.theta[j])*dicoZmbIm[i,j] for i,j in dicoZmbIm))
        
        
        self.mdl.line_constraints = ConstraintList()
        if self.config['lineconstraints']=='I':
            for idx_line in range(self.cl):
                row, col = self.Nf[idx_line].nonzero()
                dicoNflineRe = {(row[aux],col[aux]):np.real(self.Nf[idx_line][row[aux],col[aux]]) for aux in range(len(row))}
                dicoNflineIm = {(row[aux],col[aux]):np.imag(self.Nf[idx_line][row[aux],col[aux]]) for aux in range(len(row))}
                self.mdl.line_constraints.add(sum(self.mdl.VM[i]*self.mdl.VM[j]*cos(self.mdl.theta[i]-self.mdl.theta[j])*dicoNflineRe[i,j] for i,j in dicoNflineRe) + sum(self.mdl.VM[i]*self.mdl.VM[j]*sin(self.mdl.theta[i]-self.mdl.theta[j])*dicoNflineIm[i,j] for i,j in dicoNflineIm) <=self.Imax[idx_line]**2)
                row, col = self.Nt[idx_line].nonzero()
                dicoNtlineRe = {(row[aux],col[aux]):np.real(self.Nt[idx_line][row[aux],col[aux]]) for aux in range(len(row))}
                dicoNtlineIm = {(row[aux],col[aux]):np.imag(self.Nt[idx_line][row[aux],col[aux]]) for aux in range(len(row))}
                self.mdl.line_constraints.add(sum(self.mdl.VM[i]*self.mdl.VM[j]*cos(self.mdl.theta[i]-self.mdl.theta[j])*dicoNtlineRe[i,j] for i,j in dicoNtlineRe) + sum(self.mdl.VM[i]*self.mdl.VM[j]*sin(self.mdl.theta[i]-self.mdl.theta[j])*dicoNtlineIm[i,j] for i,j in dicoNtlineIm) <=self.Imax[idx_line]**2)
        
        
        elif self.config['lineconstraints']=='S':
            for idx_line,line in enumerate(self.clinelistinv):
                b,a,h = line
                index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
                reYbaff, imYbaff, reYbaft, imYbaft = (np.real(self.Yff[line])),(np.imag(self.Yff[line])),(np.real(self.Yft[line])),(np.imag(self.Yft[line]))
                expr_re = ((self.mdl.VM[index_bus_b]**2)*reYbaff) + (self.mdl.VM[index_bus_b]*self.mdl.VM[index_bus_a])*(reYbaft * cos(self.mdl.theta[index_bus_b]-self.mdl.theta[index_bus_a])+imYbaft * sin(self.mdl.theta[index_bus_b]-self.mdl.theta[index_bus_a]))
                expr_im = -((self.mdl.VM[index_bus_b]**2)*imYbaff) + (self.mdl.VM[index_bus_b]*self.mdl.VM[index_bus_a])*(reYbaft * sin(self.mdl.theta[index_bus_b]-self.mdl.theta[index_bus_a]) - imYbaft * cos(self.mdl.theta[index_bus_b]-self.mdl.theta[index_bus_a]))
                self.mdl.line_constraints.add(expr_re**2 + expr_im**2 <= self.Imax[idx_line]**2)
                #Switching indices to have (a,b,h) \in L
                aux = index_bus_b
                index_bus_b = index_bus_a
                index_bus_a = aux
                del reYbaff, imYbaff, reYbaft, imYbaft
                reYbatt, imYbatt, reYbatf, imYbatf = (np.real(self.Ytt[line])),(np.imag(self.Ytt[line])),(np.real(self.Ytf[line])),(np.imag(self.Ytf[line]))
                expr_re = ((self.mdl.VM[index_bus_b]**2)*reYbatt) + (self.mdl.VM[index_bus_b]*self.mdl.VM[index_bus_a])*(reYbatf * cos(self.mdl.theta[index_bus_b]-self.mdl.theta[index_bus_a])+imYbatf * sin(self.mdl.theta[index_bus_b]-self.mdl.theta[index_bus_a]))
                expr_im = -((self.mdl.VM[index_bus_b]**2)*imYbatt) + (self.mdl.VM[index_bus_b]*self.mdl.VM[index_bus_a])*(reYbatf * sin(self.mdl.theta[index_bus_b]-self.mdl.theta[index_bus_a]) - imYbatf * cos(self.mdl.theta[index_bus_b]-self.mdl.theta[index_bus_a]))
                self.mdl.line_constraints.add(expr_re**2 + expr_im**2 <= self.Imax[idx_line]**2)
          
        else:
            assert(self.config['lineconstraints']==False)
            
            
        self.mdl.angle_const = ConstraintList()
        for (index_bus_b,index_bus_a) in self.SymEdgesNoDiag:
            self.mdl.angle_const.add(self.mdl.theta[index_bus_b] - self.mdl.theta[index_bus_a] >=  self.mdl.mutableThetaMinByEdge[index_bus_b,index_bus_a])
            self.mdl.angle_const.add(self.mdl.theta[index_bus_b] - self.mdl.theta[index_bus_a] <=  self.mdl.mutableThetaMaxByEdge[index_bus_b,index_bus_a])

    def update_pen(self,pen):
        self.mdl.penal = pen

    def update_pref(self, pref):    
        for i in range(self.gn):
            self.mdl.Pref[i] = pref[i]
        
    def update_vref(self, vref):    
        for i in range(self.n):
            self.mdl.Vref[i] = vref[i]
    
    def update_thetaref(self, thetaref):    
        for index_bus_b,index_bus_a in self.SymEdgesNoDiag:
            self.mdl.targetThetaByEdge[(index_bus_b,index_bus_a)] = thetaref[(index_bus_b,index_bus_a)]

    def update_active_power_bounds(self,Pmin, Pmax):
        for i in range(self.gn):
            self.mdl.mutablePmin[i],self.mdl.mutablePmax[i] = Pmin[i], Pmax[i]
            
    def update_reactive_power_bounds(self,Qmin, Qmax):
        for i in range(self.gn):
            self.mdl.mutableQmin[i],self.mdl.mutableQmax[i] = Qmin[i], Qmax[i]
    
    def update_magnitude_bounds(self,Vmin, Vmax):
        for i in range(self.n):
            self.mdl.mutableVmin[i],self.mdl.mutableVmax[i] = Vmin[i], Vmax[i]
            
    def update_diff_angle_bounds(self,ThetaMin, ThetaMax):
        for index_bus_b,index_bus_a in self.SymEdgesNoDiag:
            self.mdl.mutableThetaMinByEdge[index_bus_b,index_bus_a],self.mdl.mutableThetaMaxByEdge[index_bus_b,index_bus_a] = ThetaMin[index_bus_b,index_bus_a], ThetaMax[index_bus_b,index_bus_a]

    def solve(self):
        opt = SolverFactory('ipopt')
        opt.options['tol'] = 1E-10
        results = opt.solve(self.mdl)
        #self.mdl.display()
        print(results)
        pgen = np.array([value(self.mdl.Pgen[i]) for i in range(self.gn)])
        qgen = np.array([value(self.mdl.Qgen[i]) for i in range(self.gn)])
        theta = np.array([value(self.mdl.theta[i]) for i in range(self.n)])
        VM = np.array([value(self.mdl.VM[i]) for i in range(self.n)])
        V = VM * np.exp(1j*theta)
        
        cost = self.offset
        for i in range(self.gn):
            cost+=pgen[i]*self.lincost[i] + (pgen[i]**2)*self.quadcost[i]
        print(self.name,cost)
        
        if self.__feasible(pgen,qgen, V) and cost<self.value:
            self.success = 1
            self.value = cost
            self.Pgen, self.Qgen, self.V = pgen,qgen, V
            self.VM,self.theta = np.abs(V),np.angle(V)
        
        
        return pgen,qgen, V
    
