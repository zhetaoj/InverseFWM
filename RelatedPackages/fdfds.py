#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 00:41:37 2022

@author: Zhetao Jia
"""
import numpy as np


class fdfds():
    def __init__(self, sims):
        self.sim = sims[2]
        self.lensim = len(sims)
        self.dx = self.sim.dx
        self.ndims = self.sim.ndims
        self.sims = sims

        self.w_pml_left= self.sim.w_pml_left
        self.w_pml_right = self.sim.w_pml_right
        self.w_pml_top = self.sim.w_pml_top
        self.w_pml_bottom = self.sim.w_pml_bottom
        self.M = self.sim.M
        self.N = self.sim.N
        self.X = self.sim.X
        self.Y = self.sim.Y
        self.nunks = self.sim.nunks
        
    def solve_forward(self):
        # TODO; parallize multiple solvers
        for sim in self.sims:
            sim.solve_forward()
        # Parallel(n_jobs=48)(sim.solve_forward for sim in self.sims)
        # Parallel(n_jobs=3)(delayed(lambda x:x.forward())(sim) for sim in self.sims)
        # print('parallel fwd')
    def solve_adjoint(self):
        for sim in self.sims:
            sim.solve_adjoint() 

    def update(self, bbox = None):
        for sim in self.sims:
            sim.update(bbox)
            
    def get_A_diag(self, vdiag=None):
        # return self.sim.get_A_diag(vdiag=None)
        # adiags = np.concatenate(self.sims[0].get_A_diag(vdiag=None), self.sims[1].get_A_diag(vdiag=None), self.sims[2].get_A_diag(vdiag=None))
        adiags = [self.sims[0].get_A_diag(vdiag=None), 
                  self.sims[1].get_A_diag(vdiag=None), 
                  self.sims[2].get_A_diag(vdiag=None)]
        return adiags
    
    # def get_A_diag(self, vdiag=None):
    #     # return self.sim.get_A_diag(vdiag=None)
    #     # adiags = np.concatenate(self.sims[0].get_A_diag(vdiag=None), self.sims[1].get_A_diag(vdiag=None), self.sims[2].get_A_diag(vdiag=None))
    #     vdiag = [self.sims[0].get_A_diag(self.sims[0]._Adiag1), 
    #               self.sims[1].get_A_diag(self.sims[1]._Adiag1), 
    #               self.sims[2].get_A_diag(self.sims[2]._Adiag1)]
    #     return vdiag
    
    
    def set_adjoint_sources(self, dFdxs):
        for i_sim, sim in enumerate(self.sims):
            sim.set_adjoint_sources(dFdxs[i_sim])
            
    def calc_ydAx(self, Ai):
        products = []
        for i_sim, sim in enumerate(self.sims):
            product_i = sim.calc_ydAx(Ai[i_sim])
            products.append(np.sum(product_i[...]))
        return np.sum(products)
    
