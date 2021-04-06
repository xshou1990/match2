#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:51:53 2020

@author: xiaoshou
"""



import numpy as np

class Simdata():
    
    """ 
        THIS CLASS simulate X,T,y 
    """
    
    
    def __init__(self,featdim = 1,nsamps=1): 
        self.featdim = featdim
        self.nsamps = nsamps
        
    def f1(self,x):
        value = -2*np.sin(2*x)  
        return value   
    
    def f2(self,x):
        value = x**2 -1/3
        return value
    
    def f3(self,x):
        value = x - 0.5
        return value
    
    def f4(self,x):
        value = np.exp(-x)-np.exp(-1) - 1
        return  value
    
    def f5(self,x):
        value = (x-0.5)**2 + 2
        return value
    
    def f6(self,x):
        value = np.maximum(0,x)
        return value 
    
    def f7(self,x):
        value = np.exp(-x)
        return value
    
    def f8(self,x):
        value = np.cos(x)
        return value
    
    def f9(self,x):
        value = x**2
        return value 
    
    def f10(self,x):
        value = x
        return value
    
    def simN(self):
        mean = np.zeros(self.featdim)
        cov = np.identity(self.featdim)
        x = np.random.multivariate_normal(mean, cov, self.nsamps)
        T = (self.f1(x[:,0])+self.f2(x[:,1])+ self.f3(x[:,2])+self.f4(x[:,3])+self.f5(x[:,4]) >0)*1
        ymean = self.f6(x[:,0])+self.f7(x[:,1])+self.f8(x[:,2])+self.f9(x[:,3])+self.f10(x[:,4]) + T
        y = np.random.normal(ymean, np.ones(self.nsamps), self.nsamps)
        xTy = np.append(np.append(x,np.expand_dims(T,axis=1),axis=1),np.expand_dims(y,axis=1),axis=1)
        return xTy
        
        