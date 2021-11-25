# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:05:36 2021

@author: aoust
"""

import pandas as pd
import cmath
import math
import numpy as np 
import time

class mipsResultParser():
    
    """Class to parse the results of the local solver MIPS. """
    
    def __init__(self,folder,name,baseMVA):
        
        self.name = name
        self.value = pd.read_csv(folder+"/{0}_obj.csv".format(name),header=None)[0][0]
        self.success = pd.read_csv(folder+"/{0}_success.csv".format(name),header=None)[0][0]
        self.df_bus = pd.read_csv(folder+"/{0}_bus.csv".format(name),header=None,index_col=None)
        self.df_gen = pd.read_csv(folder+"/{0}_gen.csv".format(name),header=None)
        self.df_branch = pd.read_csv(folder+"/{0}_branch.csv".format(name),header=None)
        
        #Bus values
        self.bus_indexes = self.df_bus[0].values
        self.VM = self.df_bus[7].values
        self.VA = self.df_bus[8].values
        aux = self.df_bus[8].values/180
        self.V = np.array([self.VM[i]*cmath.exp(1j*math.pi*aux[i]) for i in range(len(self.VM))])
        self.PD = self.df_bus[2].values/baseMVA
        self.QD = self.df_bus[3].values/baseMVA
        self.alpha_plus = self.df_bus[15].values
        self.alpha_moins = self.df_bus[16].values
        self.beta = self.df_bus[13].values 
        self.gamma = self.df_bus[14].values 
        
        #Gen values
        self.gen_bus_index = self.df_gen[0].values
        self.Pg = self.df_gen[1].values/baseMVA
        self.Qg = self.df_gen[2].values/baseMVA
        self.beta_gen_plus = self.df_gen[21].values 
        self.beta_gen_moins = self.df_gen[22].values 
        self.gamma_gen_plus = self.df_gen[23].values 
        self.gamma_gen_moins = self.df_gen[24].values 
        
        #Branch values
        self.line_fbus = self.df_branch[0].values
        self.line_tbus = self.df_branch[1].values
        self.lambda_f = self.df_branch[17].values
        self.lambda_t = self.df_branch[18].values
        
    
    def test_validity(self,instance):
         local_gen_indices = [i for i in range(len(self.df_gen[0].values)) if i not in instance.inactive_generators]
         ###Active Power bounds
         diffP1=self.Pg[local_gen_indices]-np.array([instance.Pmin[idx] for idx in range(instance.gn)])
         diffP2=np.array([instance.Pmax[idx] for idx in range(instance.gn)]) - self.Pg[local_gen_indices]
         #print(diffP1.min(),diffP2.min())
         ###Reactive Power bounds
         diffQ1=self.Qg[local_gen_indices]-np.array([instance.Qmin[idx] for idx in range(instance.gn)])
         diffQ2=np.array([instance.Qmax[idx] for idx in range(instance.gn)]) - self.Qg[local_gen_indices]
         #print(diffQ1.min(),diffQ2.min())
         ###Voltage magnitude
         diffV1 = self.VM - np.array(instance.Vmin)
         diffV2 = np.array(instance.Vmax) - self.VM
         #print(diffV1.min(),diffV2.min())
         ###power equations
         pgen,qgen = np.zeros(instance.n),np.zeros(instance.n)
         for idx_gen,gen in enumerate(instance.genlist):
             bus,index = instance.genlist[idx_gen]
             index_bus =  instance.buslistinv[bus]
             pgen[index_bus]+=self.Pg[local_gen_indices[idx_gen]]
             qgen[index_bus]+=self.Qg[local_gen_indices[idx_gen]]
         
         
         pmax,qmax = 0,0
         v = (self.V.reshape((len(self.V),1)))
         matrix = v.dot(np.conj(v).T)
        
         for index_bus in range(instance.n):
            Psomme = instance.Pload[index_bus]
            Qsomme = instance.Qload[index_bus]
            Psomme+= (np.trace(((instance.HM[index_bus])).dot(matrix)))
            Qsomme+= (np.trace(((1j*instance.ZM[index_bus])).dot(matrix)))
            pmax,qmax = max(pmax,abs(pgen[index_bus]-Psomme)),max(qmax,abs(qgen[index_bus]-Qsomme))
         print(pmax,qmax)
         #i1carre, i2carre = [],[]
         # for  idx_line in instance.Nt:
         #     i1carre.append(np.real(np.trace(((instance.Nt[idx_line])).dot(matrix))))
         #     i2carre.append(np.real(np.trace(((instance.Nf[idx_line])).dot(matrix))))
         # i1carre, i2carre = np.array(i1carre), np.array(i2carre)
         #print((np.array(instance.Imax)**2-i1carre).min(),(np.array(instance.Imax)**2-i2carre).min())

# name_instance='case89pegase'

# instance_config = {"lineconstraints" : True,  "cliques_strategy":"ASP"}
# Instance = instance.ACOPFinstance("matpower_data/{0}.m".format(name_instance),name_instance,instance_config)
# I = FileResultParser(name_instance,Instance.baseMVA)
# I.test_validity(Instance)       