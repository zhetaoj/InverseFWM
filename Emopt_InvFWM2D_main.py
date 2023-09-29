#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:58:59 2022

@author: zhetao Jia

To run the code, execute::
    mpirun -n 16 python file_name.py
   
"""

import emopt
from RelatedPackages.fdfds import fdfds
from RelatedPackages.plot_iteration_FWM import plot_iteration_FWM
from RelatedPackages.adjoint_method import AdjointMethod

from emopt.misc import NOT_PARALLEL, RANK, run_on_master
from RelatedPackages.adjoint_method import AdjointMethodPNF2D, AdjointMethod, AdjointMethodMO, AdjointMethodFM2D
import numpy as np
from math import pi
import os
from datetime import datetime
import time

from shutil import copyfile

class Finite1D_Multi_AM(AdjointMethodFM2D):
    """Compute the merit function and gradient of a grating coupler.
    Parameters
    ----------
    sim : emopt.fdfd.FDFD
        The simulation object
    grating_etch : list of Rectangle
        The list of rectangles which define the grating etch.
    """
    def __init__(self, simwrap, finite_gratings, Ng1, Ng2, x_grating, dx, gdoms):
        super(Finite1D_Multi_AM, self).__init__(simwrap, step=1e-10)
        # save the variables for later
        self.finite_gratings = finite_gratings
        self.Ng1 = Ng1
        self.Ng2 = Ng2
        self.x_grating = x_grating
        self.dx = dx
        self.ds = self.dx*self.dx
        self.current_fom = 0.0
        self.gdoms = gdoms # domain of grating region

    def update_system(self, params):
        """Update the geometry of the grating coupler based on the provided
        design parameters."""
        # need to update all three grating objects
        for i_grat in range(len(self.finite_gratings)):
            for i_rect in range(self.Ng1+self.Ng2):
                self.finite_gratings[i_grat][i_rect].x0  = self.x_grating+sum(params[0:2*i_rect+1])+params[2*i_rect+1]/2
                self.finite_gratings[i_grat][i_rect].width = params[2*i_rect+1]

    @run_on_master
    def calc_fom(self, simwrap, params):    
        sims = simwrap.sims
        EzP, HxP, HyP= sims[0].saved_fields[0]
        EzS, HxS, HyS= sims[1].saved_fields[0]
        EzI, HxI, HyI= sims[2].saved_fields[0]
        eps_grid = sims[0].eps.get_values_in(self.gdoms[0])  
        eps_bg = np.min(eps_grid, axis=None)
        
        # need to be derivable for epsilon
        self.current_fom = -self.ds*abs(np.sum((eps_grid-eps_bg)*EzP**2*EzS*EzI))**2
        return self.current_fom
    
    @run_on_master
    def calc_dFdx(self, simwrap, params):
        """Calculate the derivative of figure
        of merit with respect to the electric and magnetic fields at each
        location in the grid.
        """
        sims = simwrap.sims
        simM, simN = sims[0].M, sims[0].N
        dfdEzP, dfdHxP, dfdHyP, dfdEzS, dfdHxS, dfdHyS, dfdEzI, dfdHxI, dfdHyI = np.zeros([9, simM, simN], dtype=np.complex128)
        
        EzP, HxP, HyP= sims[0].saved_fields[0]
        EzS, HxS, HyS= sims[1].saved_fields[0]
        EzI, HxI, HyI= sims[2].saved_fields[0]
        
        gdom = self.gdoms[0]
        eps_grid = sims[0].eps.get_values_in(gdom)
        eps_bg = np.min(eps_grid, axis = None)
        
        dfdEzP[gdom.j, gdom.k] = -self.ds*(eps_grid-eps_bg)*2*EzP*EzS*EzI*np.conj(np.sum((eps_grid-eps_bg)*EzP**2*EzS*EzI))
        dfdEzS[gdom.j, gdom.k] = -self.ds*(eps_grid-eps_bg)*EzP**2*EzI*np.conj(np.sum((eps_grid-eps_bg)*EzP**2*EzS*EzI))
        dfdEzI[gdom.j, gdom.k] = -self.ds*(eps_grid-eps_bg)*EzP**2*EzS*np.conj(np.sum((eps_grid-eps_bg)*EzP**2*EzS*EzI))

        dFdEzP, dFdHxP, dFdHyP = emopt.fomutils.interpolated_dFdx_2D(sims[0], dfdEzP, dfdHxP, dfdHyP)
        dFdEzS, dFdHxS, dFdHyS = emopt.fomutils.interpolated_dFdx_2D(sims[1], dfdEzS, dfdHxS, dfdHyS)
        dFdEzI, dFdHxI, dFdHyI = emopt.fomutils.interpolated_dFdx_2D(sims[2], dfdEzI, dfdHxI, dfdHyI)
        dFdxs = [(dFdEzP, dFdHxP, dFdHyP), (dFdEzS, dFdHxS, dFdHyS), (dFdEzI, dFdHxI, dFdHyI)]
        return dFdxs
    
    @run_on_master
    def calc_dFdm(self, simwrap, params):
        sims = simwrap.sims
        simM, simN = sims[0].M, sims[0].N
        dFdeps, dFdeps_conj, dFdmu, dFdmu_conj = np.zeros([4, simM, simN], dtype=np.complex128)

        EzP, HxP, HyP= sims[0].saved_fields[0]
        EzS, HxS, HyS= sims[1].saved_fields[0]
        EzI, HxI, HyI= sims[2].saved_fields[0]
        
        gdom = self.gdoms[0]
        eps_grid = sims[0].eps.get_values_in(gdom) 
        eps_bg = np.min(eps_grid, axis = None)

        dFdeps     [gdom.j, gdom.k] = -self.ds*EzP**2*EzS*EzI*np.conj(np.sum((eps_grid-eps_bg)*EzP**2*EzS*EzI))
        dFdeps_conj[gdom.j, gdom.k] = -self.ds*np.conj(EzP**2*EzS*EzI)*np.sum((eps_grid-eps_bg)*EzP**2*EzS*EzI)

        return dFdeps, dFdeps_conj, dFdmu, dFdmu_conj
    
    # @run_on_master
    def calc_grad_p(self, simwrap, params):
        dFdp = np.zeros(params.shape)   
        return dFdp
    
    # @run_on_master
    def get_update_boxes(self, simwrap, params):
        lenp = int(len(params)/2)
        x_grating = 2.4

        xs = [] # centers of gratings
        for i_rect in range(lenp):
            xs.append(x_grating+sum(params[0:2*i_rect+1])+params[2*i_rect+1]/2)
            
        # define boxes surrounding grating
        d0 = 0.1 # may avoid subpixel avg error
        boxes = []
        for i_rect in range(lenp):
            boxes.append((xs[i_rect]-params[2*i_rect+1]/2-d0, 
                          xs[-1]+params[2*(Ng1+Ng2)-1]/2+d0,
                          0,
                          0.12+d0))

            boxes.append((xs[i_rect]-params[2*i_rect+1]/2-d0, 
                          xs[-1]+params[2*(Ng1+Ng2)-1]/2+d0,
                          0,
                          0.12+d0))
        return boxes

#%% Export parameters to txt    
def export_data(params, foms, it):    
    lumF = os.getcwd()+'/Data/opt_it%d.txt' %it
    xsF, wsF, hsF = np.zeros([3, Ng1+Ng2])
    xsF = np.zeros(Ng1+Ng2)
    for i_rect in range(Ng1+Ng2):
        xsF[i_rect]  = x_grating+sum(params[0:2*i_rect+1])+params[2*i_rect+1]/2
        wsF[i_rect] = params[2*i_rect+1]
        hsF[i_rect] = 0.22 
        
    txtF = open(lumF, "w")
    for param_i in range(Ng1+Ng2):
        txtF.write('%2.6f\n' %(xsF[param_i]))
        txtF.write('%2.6f\n' %(wsF[param_i]))
        txtF.write('%2.6f\n' %(hsF[param_i]))
    txtF.close()

    npzfile = os.getcwd()+'/Data/opt_it%d.npz' %it
    np.savez(npzfile, wsF=wsF, xsF=xsF, hsF=hsF, params = params, foms = foms)
        
def plot_update(params, fom_list, simwrap, am):
    """Save a snapshot of the current state of the structure.
    This function is passed to an Optimizer object and is called after each
    iteration of the optimization. It plots the current refractive index
    distribution, the electric field, and the full figure of merit history.
    """
    print('Finished iteration %d' % (len(fom_list)+1))
    current_fom = am.calc_fom(simwrap, params)
    fom_list.append(current_fom)

    foms = {'fom' : fom_list}
    sims = simwrap.sims
    EzP, HxP, HyP = sims[0].saved_fields[-1]
    EzS, HxS, HyS = sims[1].saved_fields[-1]
    EzI, HxI, HyI = sims[2].saved_fields[-1]
    eps_grid = sims[0].eps.get_values_in(sims[0].field_domains[-1]).real
    # eps_grid = sims[0].eps.get_values_in(gdom) 
    eps_bg = np.min(eps_grid, axis = None)

    field_overlap = np.flipud((eps_grid-eps_bg)*(EzP**2*EzS*EzI))
    Ez_psi = np.concatenate((np.real(EzP)/np.max(abs(EzP)), np.real(EzS)/np.max(abs(EzS)), np.real(EzI)/np.max(abs(EzI)))) 

    #bottom to top: P 1.55, S 1.53 ,I 1.57
    # Ey_psi = np.concatenate((np.real(EyP), np.zeros(EyP.shape), np.real(EyI)))
    # eps_psi = np.concatenate((eps, eps, eps, eps))
    it = len(fom_list)-1
    lbds = [sims[0].wavelength, sims[1].wavelength, sims[2].wavelength]
    
    # save to png every 3 iterations, convert to gif afterwards
    if it%3 == 0:
        img_name = os.getcwd() + '/Data/Images/%d.png' %it
        export_data(params, fom_list, len(fom_list)-1)
    else:
        img_name = ''

    plot_iteration_FWM(np.flipud(np.real(Ez_psi)), np.flipud(eps_grid), field_overlap, params,
                       sims[0].field_domains,sims[0].Xreal,
                       sims[0].Yreal*3, foms, lbds = lbds, fname=img_name, show_now=True,
                       dark=False)
    
    
if __name__ == '__main__':

    ####################################################################################
    # file location
    ####################################################################################
    
    pwd = os.getcwd()
    data_folder = pwd + '/Data/'
    currentF = os.path.basename(__file__)
    t0 = time.time()
    
    test = os.listdir(data_folder)
    for item in test:
        if item.endswith(".npz") or item.endswith(".txt"):
            os.remove(os.path.join(data_folder, item))
    copyfile(pwd+'/'+currentF, data_folder+currentF)
        
    ####################################################################################
    # define the system parameters
    ####################################################################################
    c_const = 299792458

    w_spac1, w_spac2 = 0.147, 0.147#0.213, 0.24 #0.262
    w_rect1, w_rect2 = w_spac1, w_spac2#00.213, 0.24

    h_rect = 0.22 
    Ng1, Ng2 =  25,10
    x_grating = 2.4

    X = x_grating+(w_spac1+w_rect1)*Ng1+ (w_spac2+w_rect2)*Ng2 +2
    Y = 2
    dx = 0.02
    dy = dx
    w_pml = 0.8

    ams = []
    simlist = []
    grating_domains = []
    finite_gratings = []
            
    wlen_p = 1.549 
    wlen_i = 1.542
    wlen_s = c_const/(c_const/wlen_p*2-c_const/wlen_i)    
    print('pump = %2.3fum,  signal = %2.3fum, idler = %2.3fum' %(wlen_p, wlen_s, wlen_i))

    for i_wlen, wlen in enumerate([wlen_p,wlen_s,wlen_i]): 

        n_SiO2 = 1.444  
        n_Si2D = 3.476
    
        simlist.append(emopt.fdfd.FDFD_TE(X, Y, dx, dy, wlen))
        
        simlist[i_wlen].w_pml = [w_pml, w_pml, w_pml, 0]
        simlist[i_wlen].bc = '0E' # apply symmetry at y=0
        
        # Get the actual width and height
        # The true width/height will not necessarily match what we used when
        # initializing the solver. This is the case when the width is not an integer
        # multiple of the grid spacing used.
        X = simlist[i_wlen].X
        Y = simlist[i_wlen].Y
        M = simlist[i_wlen].M
        N = simlist[i_wlen].N
        
        ####################################################################################
        # Setup system materials
        ####################################################################################
        eps_background = emopt.grid.Rectangle(X/2, Y/2, X, Y)
        eps_background.layer = 3
        eps_background.material_value = n_SiO2**2
        
        eps_wg = emopt.grid.Rectangle(x_grating/2, 0, x_grating, h_rect)
        eps_wg.layer = 2
        eps_wg.material_value = n_Si2D**2

        ####################################################################################
        # Setup the optimization
        ####################################################################################
        # params0 = np.zeros((Ng1+Ng2)*3)
        # an example for Ng1 = 1, Ng2 = 1
        # params = [w_spac1, w_rect1, w_spac2, w_rect2, h_rect1, h_rect2]
        
        params0 = np.zeros((Ng1+Ng2)*2)
        for i_rect in range(Ng1):
            params0[i_rect*2] = w_spac1
            params0[i_rect*2+1] = w_rect1
        for i_rect in range(Ng2):
            params0[i_rect*2+Ng1*2] = w_spac2
            params0[i_rect*2+1+Ng1*2] = w_rect2
        
        # grating objects are not the same for different wavelengths
        finite_grating = []
        for i_rect in range(Ng1+Ng2):
            rect = emopt.grid.Rectangle(x_grating + sum(params0[0:2*i_rect+1])+params0[2*i_rect+1]/2, 0, params0[2*i_rect+1], h_rect)
            rect.layer = 1
            rect.material_value = n_Si2D**2
            finite_grating.append(rect)
        finite_gratings.append(finite_grating)
        
        eps = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)
        for elem in finite_grating:
            eps.add_primitive(elem)
        eps.add_primitive(eps_background)
        eps.add_primitive(eps_wg)
        mu = emopt.grid.ConstantMaterial2D(1.0)
        simlist[i_wlen].set_materials(eps, mu)
        
        ####################################################################################
        # setup the sources
        ####################################################################################
        w_src = 2.8
       	src_line = emopt.misc.DomainCoordinates(w_pml+1, w_pml+1, 0, w_src/2, 0, 0, dx, dy, 1.0)

        # Setup the mode solver.    
        mode = emopt.modes.ModeTE(wlen, eps, mu, src_line, n0=n_Si2D, neigs=4)
        mode.bc = 'E'
        mode.build()
        mode.solve()
        mindex = mode.find_mode_index(0)
        simlist[i_wlen].set_sources(mode, src_line, mindex)
        
        grating_domain = emopt.misc.DomainCoordinates(x_grating, X-0.5, 0, 0.24, 0, 0, dx, dy, 1.0)   
        sim_area = emopt.misc.DomainCoordinates(w_pml, X-w_pml, 0, Y-w_pml, 0.0, 0.0, dx, dy, 1.0)
        simlist[i_wlen].field_domains = [grating_domain, sim_area] #, src_line, sim_area
        simlist[i_wlen].build()
        grating_domains.append(grating_domain)
        
    simwrap = fdfds(simlist)
    am = Finite1D_Multi_AM(simwrap, finite_gratings, Ng1, Ng2, x_grating, dx, grating_domains)
    # am.check_gradient(params0) # compare gradients calculated by adjoint method with finite difference method
    
    fom_list = []
    def callback(x):
          plot_update(x, fom_list, simwrap, am)

    # # setup and run the optimization!
    bounds = []
    for i_rect in range((Ng1+Ng2)*2):
        bounds.append((0.11, 0.3))

    opt = emopt.optimizer.Optimizer(am, params0, tol=1e-20,
                                    opt_method='L-BFGS-B',
                                    bounds = bounds,    
                                    callback_func=callback,
                                    Nmax=3000)

    # # Run the optimization
    final_fom, final_params = opt.run()
    print('Elapsed time = ', time.time()-t0)

    print('Initial params:')
    print(params0)
    print('Final FOM:')
    print(final_fom)
    print('Final params:')
    print(final_params)


# Combine exported pngs to gif 
  
# ffmpeg -pattern_type glob -i '*.png' -vf palettegen palette.png 
# ffmpeg -f image2 -framerate 10 -pattern_type glob -i '*.png' -i palette.png -lavfi paletteuse out.gif
