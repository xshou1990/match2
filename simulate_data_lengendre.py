#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:51:53 2020

@author: xiaoshou
"""



import numpy as np
from scipy.special import eval_legendre

class Simdata():
    
    """ 
        THIS CLASS simulate X,T,y 
    """
    
    
    def __init__(self,featdim = 25,nsamps=100):
#        self.f1 = None
#        self.f2 = None
#        self.f3 = None
#        self.f4 = None
#        self.f5 = None
#        self.f6 = None
#        self.f7 = None
#        self.f8 = None
#        self.f9 = None
#        self.f10 = None   
        self.featdim = featdim
        self.nsamps = nsamps
        
    def f1(self,x):
    
        return eval_legendre(1,x)  
    
    def f2(self,x):
        return eval_legendre(2,x)  
    
    def f3(self,x):
        return eval_legendre(3,x)  
    
    def f4(self,x):
        return eval_legendre(4,x)  
    
    def f5(self,x):
        return eval_legendre(5,x)  
    
    def f6(self,x):
        return eval_legendre(6,x)  
    
    def f7(self,x):
        return eval_legendre(7,x)  
    
    def f8(self,x):
        return eval_legendre(8,x)  
    
    def f9(self,x):
        return eval_legendre(9,x)  
    
    def f10(self,x):
        return eval_legendre(10,x)  
    
    def simN(self):
        mean = np.zeros(self.featdim)
        cov = np.identity(self.featdim)
        x = np.random.multivariate_normal(mean, cov, self.nsamps)
        #print(x)
        #
        #print(self.f10(x[:,1]))
        T = (self.f1(x[:,0])+self.f2(x[:,1])+ self.f3(x[:,2])+self.f4(x[:,3])+self.f5(x[:,4]) >0)*1
        ymean = self.f6(x[:,0])+self.f7(x[:,1])+self.f8(x[:,2])+self.f9(x[:,3])+self.f10(x[:,4]) + T
        y = np.random.normal(ymean, np.ones(self.nsamps), self.nsamps)
        xTy = np.append(np.append(x,np.expand_dims(T,axis=1),axis=1),np.expand_dims(y,axis=1),axis=1)
        return xTy
        
        