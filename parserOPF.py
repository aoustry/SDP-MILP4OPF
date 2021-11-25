# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:37:57 2021

@author: aoust
"""

#!/usr/bin/env python


## inspired by Leo Liberti's code

################ imports ################

import sys
import os
import types
import math
import cmath
import numpy as np

################ user-configurable ###############

myInf = 1e30
myZero = 1e-10
verbose = False
#verbose = True

################ Classes ################

class mpcBus:
    def __init__(self, col):
        self.ID = col[0]
        self.Type = int(col[1])
        self.Pd = float(col[2])
        self.Qd = float(col[3])
        self.Gs = float(col[4])
        self.Bs = float(col[5])
        self.area = int(col[6])
        self.Vm = float(col[7]) # starting point for polar formulation on input
        self.Va = (math.pi) * float(col[8]) / 180 # startpt for polar on input
        self.baseKV = float(col[9])
        self.zone = int(col[10])
        self.Vmax = float(col[11])
        self.Vmin = float(col[12])
    def fieldnames(self):
        out = "ID,Type,Pd,Qd,Gs,Bs,area,Vm,Va,baseKV,zone,Vmax,Vmin"
        return out
    # printing
    def __str__(self):
        out = "bus " + str(self.ID) + ","
        out += str(self.Type) + "," + str(self.Pd) + "," + str(self.Qd) + "," + str(self.Gs) + ","
        out += str(self.Bs) + "," + str(self.area) + "," + str(self.Vm) + "," + str(self.Va) + ","
        out += str(self.baseKV) + "," + str(self.zone) + "," + str(self.Vmax) + "," + str(self.Vmin)
        return out    

class mpcGen:
    # gen and gencost dicts are indexed by same keys
    def __init__(self, col, count):
        self.bus = col[0]        # bus where generator is installed
        self.counter = count     # counter of generator at bus
        self.Pg = float(col[1])  # Pg is a decision var
        self.Qg = float(col[2])  # Qg is a decision var
        self.Qmax = float(col[3])
        self.Qmin = float(col[4])
        self.Vg = float(col[5])
        self.mBase = float(col[6])
        self.status = int(col[7])
        self.Pmax = float(col[8])
        self.Pmin = float(col[9])
        try:
            self.Pc1 = float(col[10])
            self.Pc2 = float(col[11])
            self.Qc1min = float(col[12])
            self.Qc1max = float(col[13])
            self.Qc2min = float(col[14])
            self.Qc2max = float(col[15])
            self.ramp_agc = float(col[16])
            self.ramp_10 = float(col[17])
            self.ramp_30 = float(col[18])
            self.ramp_q = float(col[19])
            self.apf = float(col[20])
        except:
            pass
    def fieldnames(self):
        out = "ID,Pg,Qg,Qmax,Qmin,Vg,mBase,status,Pmax,Pmin,Pc1,Pc2,Qc1min,Qc1max,Qc2min,Qc2max,ramp_agc,ramp_10,ramp_30,ramp_q,apf"
        return out
    # printing
    def __str__(self):
        out = "gen " + str(self.bus) + "," + str(self.counter) + ":"
        out += str(self.Pg) + "," + str(self.Qg) + "," + str(self.Qmax) + "," + str(self.Qmin) + ","
        out += str(self.Vg) + "," + str(self.mBase) + "," + str(self.status) + "," + str(self.Pmax) + ","
        out += str(self.Pmin) + ","
        try:
            out += str(self.Pc1) + "," + str(self.Pc2) + "," + str(self.Qc1min) + ","
            out += str(self.Qc1max) + "," + str(self.Qc2min) + "," + str(self.Qc2max) + "," + str(self.ramp_agc) + ","
            out += str(self.ramp_10) + "," + str(self.ramp_30) + "," + str(self.ramp_q) + "," + str(self.apf)
        except:
            pass
        return out
    
class mpcBranch:
    def __init__(self, col, parid):
        self.fbus = col[0]
        self.tbus = col[1]
        # ID of parallel edge
        self.parallelID = parid
        self.r = float(col[2])
        self.x = float(col[3])
        self.b = float(col[4])
        self.rateA = float(col[5])
        self.rateB = float(col[6])
        self.rateC = float(col[7])
        self.ratio = float(col[8])
        self.angle = float(col[9])
        self.status = int(col[10])
        self.angmin = float(col[11])
        self.angmax = float(col[12])
        
    def fieldnames(self):
        out = "fbus,tbus,r,x,b,rateA,rateB,rateC,ratio,angle,status,angmin,angmax"
        return out
    # printing
    def __str__(self):
        out = "branch " + str(self.fbus) + "," + str(self.tbus) + "," + str(self.parallelID) + ":"
        out += str(self.r) + "," + str(self.x) + "," + str(self.b) + "," + str(self.rateA) + ","
        out += str(self.rateB) + "," + str(self.rateC) + "," + str(self.ratio) + "," + str(self.angle) + ","
        out += str(self.status) + "," + str(self.angmin) + "," + str(self.angmax)
        return out

class mpcGenCost:
    # gen and gencost dicts are indexed by same keys
    def __init__(self, col):
        self.Type = int(col[0])
        self.startup = float(col[1])
        self.shutdown = float(col[2])
        self.n = int(col[3])
        self.data = [float(c) for c in col[4:]]
    def fieldnames(self):
        out = "type,startup,shutdown,n"
        for i in range(self.n):
            out += ",data" + str(i)
        return out
    # printing
    def __str__(self):
        out = "gencost " + str(self.Type) + "," + str(self.startup) + "," + str(self.shutdown) + "," + str(self.n)
        for i in range(self.n):
            out += "," + str(self.data[i])
        return out
    
class mpcCase:
    def __init__(self, mpcfilename):
        self.filename = mpcfilename
        f = open(self.filename, "r")
        lines = f.readlines()
        status = "outer"
        self.bus = dict()
        self.branch = dict()
        self.parbranch = dict()
        self.maxparbranches = 0
        self.gen = dict()
        self.gencost = dict()
        buscounter = 0
        branchcounter = 0
        gencounter = 0
        gencostcounter = 0
        for l in lines:
            line = l.strip(' \n').replace('\t', ' ')
            if len(line)>0 and line[0] != '%':
                if status == "outer":
                    if line.startswith("mpc.version"):
                        self.version = int(line.split('=')[1].strip(' ;\r').strip('\''))
                    elif line.startswith("mpc.baseMVA"):
                        self.baseMVA = float(line.split('=')[1].strip(' ;\r'))
                    elif line.startswith("mpc.bus"):
                        status = "bus"
                    elif line.startswith("mpc.gen") and not line.startswith("mpc.gencost"):
                        status = "gen"
                    elif line.startswith("mpc.branch"):
                        status = "branch"
                    elif line.startswith("mpc.gencost"):
                        status = "gencost"
                else:
                    if "]" in line:
                        status = "outer"
                        rows = line.split(']')[0]
                    elif ";" in line:
                        rows = line.split(';')
                    else:
                        rows = [line]
                    for r in rows:
                        if len(r) > 1:
                            cols = r.split()
                            if cols[0] == "%":
                                continue
                            if status == "bus" and len(cols) >= 12:
                                self.bus[buscounter] = mpcBus(cols)
                                if verbose:
                                    print(self.bus[buscounter])
                                buscounter += 1
                            elif status == "gen" and len(cols) >= 10:
                                # gen and gencost are indexed by same keys
                                count = len([self.gen[g].bus for g in self.gen if self.gen[g].bus == cols[0]])
                                self.gen[gencounter] = mpcGen(cols, count)
                                if verbose:
                                    print(self.gen[gencounter])
                                gencounter += 1 # gencost idx (starts from 1)
                            elif status == "branch" and len(cols) >= 13:
                                ID = (cols[0],cols[1]) # ID is link adjacencies
                                if ID in self.parbranch:
                                    self.parbranch[ID] += 1
                                else:
                                    self.parbranch[ID] = 1
                                self.branch[branchcounter] = mpcBranch(cols, self.parbranch[ID])
                                if self.parbranch[ID] > self.maxparbranches:
                                    self.maxparbranches = self.parbranch[ID]
                                if verbose:
                                    print(self.branch[branchcounter])
                                branchcounter += 1
                            elif status == "gencost" and len(cols) >= 4:
                                # gen and gencost are indexed by same keys
                                self.gencost[gencostcounter] = mpcGenCost(cols)
                                if verbose:
                                    print(self.gencost[gencostcounter])
                                gencostcounter += 1 
        f.close()

    def fieldnames(self):
        out = "CASEFN:filename,version,baseMVA"
        return out


class OPF_Data():
    
    def __init__(self,mpccase):
        bset = set([mpccase.bus[b].ID for b in mpccase.bus])
        gset = set([(mpccase.gen[g].bus,mpccase.gen[g].counter) for g in mpccase.gen if mpccase.gen[g].status==1 ])#if mpccase.gen[g].status==1])
        lset = set([(mpccase.branch[g].fbus,mpccase.branch[g].tbus,mpccase.branch[g].parallelID) for g in mpccase.branch if mpccase.branch[g].status ==1])
        # bus quantities
        self.baseMVA = mpccase.baseMVA
        self.busType, SDR, SDC, self.VL, self.VU, shR, shC = {},{},{},{},{},{},{}
        for b in mpccase.bus:
            self.busType[mpccase.bus[b].ID] = mpccase.bus[b].Type
            SDR[mpccase.bus[b].ID] = mpccase.bus[b].Pd/mpccase.baseMVA
            SDC[mpccase.bus[b].ID] = mpccase.bus[b].Qd/mpccase.baseMVA
            shR[mpccase.bus[b].ID] = mpccase.bus[b].Gs/mpccase.baseMVA
            shC[mpccase.bus[b].ID] = mpccase.bus[b].Bs/mpccase.baseMVA
            self.VL[mpccase.bus[b].ID] = mpccase.bus[b].Vmin
            self.VU[mpccase.bus[b].ID] = mpccase.bus[b].Vmax
        
        # line quantities
        self.status, self.SU, r, x, bb, tau, theta, self.pdLB, self.pdUB = {},{},{},{},{},{},{},{},{}
        self.angmin, self.angmax = {},{}
        for l in mpccase.branch:
            if (mpccase.branch[l].status ==1):
                code = (mpccase.branch[l].fbus,mpccase.branch[l].tbus,mpccase.branch[l].parallelID)
                self.status[code] = mpccase.branch[l].status
                self.SU[code] = mpccase.branch[l].rateA / mpccase.baseMVA 
                r[code] = mpccase.branch[l].r
                x[code] = mpccase.branch[l].x
                bb[code] = mpccase.branch[l].b
                tap= mpccase.branch[l].ratio
                if abs(tap) < myZero:
                    tap = 1.0;
                tau[code] = tap
                theta[code] = (math.pi * mpccase.branch[l].angle / 180)
                self.angmin[code] = mpccase.branch[l].angmin
                self.angmax[code] = mpccase.branch[l].angmax
                if self.angmin[code] ==0 and self.angmax[code]==0:
                    self.angmin[code] = -180
                    self.angmax[code] = 180
                assert(self.angmin[code]==-self.angmax[code])
                #self.pdLB = {bah:lines['pdLB'][i] for i,bah in enumerate(lset)}
                #self.pdUB = {bah:lines['pdUB'][i] for i,bah in enumerate(lset)}
        
        #lset1 = [(a,b,h) for (b,a,h) in lset]
        
        # generator quantities
        self.SLR,self.SLC,self.SUR,self.SUC = {},{},{},{}
        self.inactive_generators = []

        for i,g in enumerate(mpccase.gen):
            #assert(mpccase.gen[g].mBase == mpccase.baseMVA)
            if (mpccase.gen[g].status==1):
                code = (mpccase.gen[g].bus,mpccase.gen[g].counter)
                self.SLR[code] = mpccase.gen[g].Pmin/ mpccase.baseMVA#mpccase.gen[g].mBase
                self.SLC[code] = mpccase.gen[g].Qmin/ mpccase.baseMVA#mpccase.gen[g].mBase
                self.SUR[code] = mpccase.gen[g].Pmax/ mpccase.baseMVA#mpccase.gen[g].mBase
                self.SUC[code] = mpccase.gen[g].Qmax/ mpccase.baseMVA#mpccase.gen[g].mBase
            else:
                self.inactive_generators.append(i)
                
        ### MP formulation
        # sizes
        self.n = len(bset)  # number of buses (B)
        self.m = len(lset)  # number of lines (L0)
        self.gn = len(gset)  # number of generators
        
        assert(len(self.SLR)==self.gn)
        
        ## parameters
        # the Y matrix for arcs in L0
        self.Yff = {bah:(1/(r[bah]+1j*x[bah]) + 1j*bb[bah]/2)/tau[bah]**2 for bah in lset}
        self.Yft = {bah:-1/((r[bah]+1j*x[bah])*tau[bah]*cmath.exp(-1j*theta[bah])) for bah in lset}
        self.Ytf = {bah:-1/((r[bah]+1j*x[bah])*tau[bah]*cmath.exp(1j*theta[bah])) for bah in lset}
        self.Ytt = {bah: 1/(r[bah]+1j*x[bah]) + 1j*bb[bah]/2 for bah in lset}
        
        # complex power demand
        self.SD = {b:SDR[b] + 1j*SDC[b] for b in bset}
        
        # line shunt
        self.A = {b:shR[b] + 1j*shC[b] for b in bset}
        
        C = {}
        for gncost in mpccase.gencost:
            thebus = mpccase.gen[gncost].bus
            theidx = mpccase.gen[gncost].counter
            if (mpccase.gen[gncost].status!=0):
                K = range(mpccase.gencost[gncost].n)
                for k in K:
                    thedeg = mpccase.gencost[gncost].n-k-1
                    thecoeff = mpccase.gen[gncost].status * mpccase.gencost[gncost].data[k] * (mpccase.baseMVA**thedeg)
                    C[(thebus,theidx,thedeg)] = thecoeff
        
        # in case costs are empty
        for (b,g) in gset:
            if (b,g,1) not in C:
                C[(b,g,1)] = 0.0
            if (b,g,2) not in C:
                C[(b,g,2)] = 0.0
                
        self.C=C
