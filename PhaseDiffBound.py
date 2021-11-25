# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:21:03 2021

@author: aoust
"""
import instance
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Circle

class PhaseDiffBound():
    
    def __init__(self,z,l,u,R):
        self.z,self.l,self.u,self.R = z,l,u,R
        self.rho = abs(z)
        self.phi = np.angle(z)
        self.vertex = []
        self.vertex.append(self.z*self.l + self.R * np.exp(1j*(self.phi+np.pi/2)))
        self.vertex.append(self.z*self.u + self.R * np.exp(1j*(self.phi+np.pi/2))) 
        self.vertex.append(self.z*self.u - self.R * np.exp(1j*(self.phi+np.pi/2)))
        self.vertex.append(self.z*self.l - self.R * np.exp(1j*(self.phi+np.pi/2)) )
        self.rectangle = Polygon([[np.real(z),np.imag(z)] for z in self.vertex],color = 'b')
        self.circle1 = Circle((self.l*np.real(self.z), self.l*np.imag(self.z)), self.R, color='r')
        self.circle2 = Circle((self.u*np.real(self.z), self.u*np.imag(self.z)), self.R, color='r')
        self.phimin,_ = self.scanning_phimin(100)
        _, self.phimax = self.scanning_phimax(100)
    
    def distance(self,point):
        if self.contains(point):
            return 0
        else:
            dist, proj = self.aux_proj_rectangle(point)
            distC1, projC1 = self.aux_proj_circle(point, self.l*self.z)
            if distC1<dist:
                dist=distC1
                proj = projC1
            distC2, projC2 = self.aux_proj_circle(point, self.u*self.z)
            if distC2<dist:
                dist=distC2
                proj = projC2
            return dist
            
    def aux_proj_circle(self,point,center):
        dtocenter = abs(point-center)
        delta = point-center
        proj = center+delta*(self.R/dtocenter)
        assert(abs(abs(point-proj)-(dtocenter - self.R))<=1E-5)
        return dtocenter - self.R, point
    
    def aux_proj_segment(self,point,zA,zB):
        #t-> |point - (t.zA+(1-t).zB) |^2
        segment_length = abs(zA-zB)
        if segment_length==0:
            return abs(point-zA),zA
        vect1, vect2 = zA-zB, point + zB
        tstar = (np.real(vect1)*np.real(vect2)+np.imag(vect1)*np.imag(vect2))/(segment_length**2)
        if tstar<=0:
            return abs(point-zB),zB
        if tstar>=1:
            return abs(point-zA),zA
        proj = tstar*zA + (1-tstar)*zB
        return abs(point-proj), proj
    
    def aux_proj_rectangle(self,point):
        dist, proj = np.inf, 0
        for i in range(4):
            auxdist,auxproj = self.aux_proj_segment(point, self.vertex[i],self.vertex[(i+1)%4])
            if auxdist<dist:
                dist = auxdist
                proj = auxproj
        return dist, proj
    
    def contains(self,point):
        bool_circle1 = abs(point-self.l*self.z)<=self.R
        bool_circle2 = abs(point-self.u*self.z)<=self.R
        rotated_x,rotated_y = np.real(point*np.exp(-1j*self.phi)),np.imag(point*np.exp(-1j*self.phi))
        bool_rectangle = (rotated_x>=self.rho*self.l) and (rotated_x<=self.rho*self.u) and (abs(rotated_y)<=self.R)
        return bool_circle1 or bool_circle2 or bool_rectangle
    

    def scanning_phimax(self, N):
        phi1,phi2 = -np.pi, np.pi
        while abs(phi2-phi1)>1E-6:
            assert(phi2>phi1)
            discretization_rad = (phi2-phi1)/N
            minimal_distance = abs(np.exp(1j*discretization_rad)-1)
            theta = np.linspace(phi1, phi2,N)
            z = np.exp(1j*theta)
            distances =  np.array(list(map(self.distance, z)))
            contains = np.array(list(map(self.contains, z)))
            if sum(contains.astype(int))==0:
                if N<1E5:
                    return self.scanning_phimax(10*N)
                else:
                    assert('Infeasible constraint')
            i2 = N
            while i2>=1 and distances[i2-1]>minimal_distance:
                i2+=-1
            phi2 = theta[min(N-1,i2)]
            i1 = N-1
            while i1>=1 and not(contains[i1]):
                i1+=-1
            phi1 = theta[i1]
        return phi1,phi2
    
    def scanning_phimin(self, N):
        phi1,phi2 = -np.pi, np.pi
        while abs(phi2-phi1)>1E-6:
            assert(phi2>phi1)
            discretization_rad = (phi2-phi1)/N
            minimal_distance = abs(np.exp(1j*discretization_rad)-1)
            theta = np.linspace(phi1, phi2,N)
            z = np.exp(1j*theta)
            distances =  np.array(list(map(self.distance, z)))
            contains = np.array(list(map(self.contains, z)))
            if sum(contains.astype(int))==0:
                if N<1E5:
                    return self.scanning_phimin(10*N)
                else:
                    assert('Infeasible constraint')
            i1 = -1
            while i1<=N-2 and distances[i1+1]>minimal_distance:
                i1+=1
            phi1 = theta[max(0,i1)]
            i2 = 0
            while i2<=N-1 and not(contains[i2]):
                i2+=1
            phi2 = theta[i2]
        return phi1,phi2
            
    
    def plot(self, name,testing = False):
        theta = np.linspace(-np.pi, np.pi,1000)
        x,y = np.cos(theta), np.sin(theta)
        if testing:
            x1 = np.linspace(-2,2,500)
            y1 = np.linspace(-2,2,500)
            xtoplot, ytoplot = [],[]
            for a in x1:
                for b in y1:
                    if self.contains(a+1j*b):
                        xtoplot.append(a)
                        ytoplot.append(b)       
        fig, ax = plt.subplots() 
        ax.add_patch(self.circle1)
        ax.add_patch(self.circle2)
        ax.add_patch(self.rectangle)
        ax.set_aspect(1)
        plt.plot(x,y)
        theta = np.linspace(self.phimin, self.phimax,1000)
        x,y = np.cos(theta), np.sin(theta)
        plt.plot(x,y,color = 'black')
        if testing:
            plt.scatter(xtoplot, ytoplot, color ='black',marker='.')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        fig.savefig('plots/'+name+'_constraints.png')
        