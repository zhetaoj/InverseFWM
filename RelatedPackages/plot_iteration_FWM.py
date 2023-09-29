#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 07:59:10 2022

@author: zhetao
"""

from __future__ import absolute_import

from builtins import zip
from builtins import range
from emopt.misc import run_on_master, warning_message, NOT_PARALLEL, COMM
from emopt.grid import Polygon
import numpy as np
import cmocean
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    
__author__ = "Andrew Michaels"
__license__ = "GPL License, Version 3.0"
__version__ = "2019.5.6"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

@run_on_master
def plot_iteration_FWM(field, structure, overlap, params, field_domains, W, H, foms, fname='', layout='balanced',
                   lbds = [1.545,1.565,1.535], show_now=True, dark=True, Nmats=2):
    """Plot the current iteration of an optimization.

    Plots the following:
        1) A 2D field
        2) A 2D representation of the geometry
        3) A line plot of the figure of merit history

    When plotting 3D structures, multiple 2D slices of the geometry can be
    passed in by passing a 3D array to structure. The different "pages" of this
    array will be flattened into a 2D representation of the sliced geometry.

    Furthermore, multiple figures of merit may be plotted. This is done by
    putting more than one key:value pair in the foms dictionary. The key names
    will be used as the legend label.

    Parameters
    ----------
    field : numpy.ndarray
        2D array containing current iteration's field slice
    structure : numpy.ndarray
        Either a 2D array containing a representation of the current structure
        or a 3D array containing a FEW 2D slices.
    W : float
        The width of the field/structure
    H : float
        The height of the field/structure
    foms : dict
        A dictionary containing the fom history. The key strings should be
        names which describe each supplied figure of merit
    fname : str
        Filename for saving
    layout : str
        Layout method. 'auto' = automatically choose layout based on aspect
        ratio. 'horizontal' = single row layour. 'vertical' = single column
        layout. 'balanced' = field+structure on left, foms on right.
    show_now : bool
        If True, show plot now (warning: this is a blocking operation)
    dark : bool
        If True, use a dark color scheme for plot. (default = True)

    Returns
    -------
    matplotlib.pyplot.figure
        The current matplotlib figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap

    dpi=300
    aspect = W/H

    # determine and setup the plot layout
    unknown_layout = layout not in ['auto', 'horizontal', 'vertical', 'balanced']
    if(unknown_layout):
        warning_message("Unknown layout '%s'. Using 'auto'" % (layout), 'emopt.io')

    if(layout == 'auto' or unknown_layout):
        if(aspect < 1.0):
            layout = 'horizontal'
        elif(aspect >= 1.0 and aspect <= 2.5):
            layout = 'balanced'
        else:
            layout = 'vertical'

    gs = None
    Wplot = 15.0
    if(layout == 'horizontal'):
        f = plt.figure(figsize=(Wplot*aspect*3, Wplot))
        gs = gridspec.GridSpec(1,3, height_ratios=[1])
        ax_field = f.add_subplot(gs[:,0])
        ax_struct = f.add_subplot(gs[:,1])
        ax_foms = f.add_subplot(gs[:,2])
    elif(layout == 'vertical'):
        f = plt.figure(figsize=(Wplot, Wplot/aspect*4))
        gs = gridspec.GridSpec(3,1, height_ratios=[3,1,1])
        ax_field = f.add_subplot(gs[0,:])
        ax_struct = f.add_subplot(gs[1,:])
        ax_foms = f.add_subplot(gs[2, :])
    elif(layout == 'balanced'):
        f = plt.figure(figsize=(Wplot, Wplot/aspect*2/1.5))
        gs = gridspec.GridSpec(3,3, height_ratios=[3,1,1])
        ax_field = f.add_subplot(gs[0,0:2])
        ax_struct = f.add_subplot(gs[1,0:2])
        ax_overlap = f.add_subplot(gs[2,0:2])
        ax_foms = f.add_subplot(gs[:,2])

    # define dark colormaps
    if(dark):
        field_cols=['#3d9aff', '#111111', '#ff3d63']
        field_cmap=LinearSegmentedColormap.from_list('field_cmap', field_cols)

        struct_cols=['#212730', '#bcccdb']
        struct_cmap=LinearSegmentedColormap.from_list('struct_cmap', struct_cols)
    else:
        field_cmap = 'seismic'
        struct_cmap = 'Blues'

    extent = [0, W, 0, H]
    extent2 = [0, W, 0, H/3]
    fmax = np.max(field)
    fmin = np.min(field)
    
    # "flatten" multi-layer structure
    if(len(structure.shape) == 3):
        Nlevels = structure.shape[0] + 1
        structure = np.sum(structure, axis=0) / structure.shape[0]
    else:
        Nlevels = Nmats
        
    if(fmax > 0 and fmin < 0):
        fmin = -1*fmax # symmetrize
    if(fmax < 0 and fmin < 0):
        fmax = 0
        field_cmap = 'hot_r'
    if(fmax > 0 and fmin > 0):
        fmin = 0
        field_cmap = 'hot'

    ax_field.imshow(field, extent=extent, vmin=fmin, vmax=fmax, cmap=field_cmap)
    # outline structure in field plot
    # Temporary fix for old versions of Matplotlib which dont support int for
    # levels
    structure_psi =  np.concatenate((structure, structure, structure))
    ax_field.contour(np.flipud(structure_psi), extent=extent, levels=Nlevels,
                      colors='#666666', linewidths=0.4)
    # src = np.zeros(structure.shape)
    # for i_domain, domain in enumerate([field_domains[1]]):
    #     src[(domain.j1-field_domains[-1].j1):(domain.j2-field_domains[-1].j1), (domain.k1-field_domains[-1].k1):(domain.k2-field_domains[-1].k1)]= fmax
    # src_psi =  np.concatenate((src, src, src))
    # ax_field.contour(src_psi, extent=extent, vmin=fmin, vmax=fmax, cmap=field_cmap)
    ax_field.grid(axis = 'x')
    
    boundary = np.zeros(field.shape)
    for i_domain, domain in enumerate([field_domains[0]]):
        boundary[(domain.j1-field_domains[-1].j1):(domain.j2-field_domains[-1].j1), (domain.k1-field_domains[-1].k1):(domain.k2-field_domains[-1].k1)]= 1
    ax_struct.contour(boundary, extent=extent, levels=Nlevels,
                      colors='#666666')
    # ax_field.imshow(field, extent=extent, vmin=fmin, vmax=fmax, cmap=field_cmap)
    ax_struct.grid(axis = 'x')

#%% plot structure
    smin = np.min(structure)
    smax = np.max(structure)
    ax_struct.imshow(structure, extent=extent2, vmin=smin, vmax=smax,
                     cmap=struct_cmap)

#%% plot structure

    eps_full = structure_psi
    eps_bg = np.min(eps_full, axis = None)
    
    y_ary, x_ary= np.meshgrid(np.linspace(0, H, overlap.shape[0]),np.linspace(0, W, overlap.shape[1]))
    x_grating = 2.4
    sz_rect = int(len(params)/2)
    w_pml = 0.8
    
    overlap_sums = []
    xs = []
    xlefts = []
    ys = []
    ws = []
    rect_domains = np.zeros(np.transpose(overlap).shape)
    for i_rect in range(sz_rect):
        xmin = x_grating+sum(params[0:2*i_rect+1])
        xmax = x_grating+sum(params[0:2*i_rect+1])+params[i_rect*2+1]
        ymin = H-0.5#params[sz_rect*2+i_rect]/2
        ymax = H
        rect_domain = (x_ary>xmin) & (x_ary<xmax) & (y_ary>ymin) & (y_ary<ymax)
        rect_domains = rect_domains + rect_domain
        overlap_sums.append(sum(overlap[np.transpose(rect_domain)]))
        
        xs.append((xmin+xmax)/2) #remover PML?
        ws.append(xmax-xmin)
        xlefts.append(xmin-w_pml+(xmax-xmin)/2)
    # print(xs)
    overlap_sums = np.asarray(overlap_sums)
    overlap_sums_abs = np.abs(overlap_sums)
    sz_norm =  max(overlap_sums_abs)
    idx_maxnorm = np.unravel_index(np.argmax(overlap_sums_abs, axis=None), overlap_sums_abs.shape)
    phase_norm = overlap_sums[idx_maxnorm]
    

    # z = np.angle(overlap_sums/phase_norm, deg = False)
    z = np.angle(overlap_sums/1, deg = False)

    minima = min(z)
    maxima = max(z)
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmocean.cm.phase)
    mapper_list = [mapper.to_rgba(x) for x in z]
    
    bar_plot = ax_overlap.bar(xlefts, np.abs(overlap_sums)/sz_norm*1.3,  
                              color=mapper_list, width=ws)
    
    # scalebar = AnchoredSizeBar(ax_overlap.transData,
                            # 1, '1 um', 'lower right', 
                            # pad=0.1,
                            # color='black',
                            # frameon=False,
                            # size_vertical=0.1)
    # ax_overlap.add_artist(scalebar)
    # ax_overlap.text(W/2, H*0.9/5,  'normalized by %3.1e' %sz_norm, fontsize = 16)
    ax_overlap.set_xlim(0,W)
    ax_overlap.set_xlim(0,1)


#%% define fom plot colors
    Nplot = len(list(foms.keys()))
    red = np.linspace(0.2, 1.0, Nplot)
    blue = np.linspace(1.0, 0.2, Nplot)
    green = np.zeros(Nplot)
    red_base = 0.0
    blue_base = 0.0
    green_base = 0.55

    i = 0
    Niter = 0
    current_foms = []
    for desc in list(foms.keys()):
        fom = foms[desc]
        Niter = len(fom)
        iters = np.arange(Niter)
        current_foms.append(fom[-1])

        pcolor = (red_base + red[i], green_base +green[i], blue_base + blue[i])
        i += 1
        fom = -np.asarray(fom)
        pline = ax_foms.semilogy(iters, fom, '.-', color=pcolor, markersize=10)

    ax_foms.set_xlabel('Iteration', fontsize=14)
    ax_foms.set_ylabel('Figure of Merit', fontsize=14)
    ax_foms.legend(list(foms.keys()), loc=4)
    ax_foms.grid(True, linewidth=0.5)

    # general tick properties
    for ax in [ax_field, ax_struct, ax_foms]:
        ax.get_yaxis().set_tick_params(which='both', direction='in', top=True,
                                      right=True)
        ax.get_xaxis().set_tick_params(which='both', direction='in', top=True,
                                      right=True)

    # Dark theme easier on eyes
    if(dark):
        c_text_main = '#BBBBBB'
        c_bg_main = '#101010'
        c_plot_bg = '#353535'
        c_lines = '#666666'
        c_plot_tick = '#CCCCCC'
        c_plot_grid = '#555555'
        f.patch.set_facecolor(c_bg_main)
        for ax in [ax_field, ax_struct, ax_foms]:
            for tl in ax.get_xticklabels():
                tl.set_color(c_text_main)
            for tl in ax.get_yticklabels():
                tl.set_color(c_text_main)
            ax.xaxis.get_label().set_color(c_text_main)
            ax.yaxis.get_label().set_color(c_text_main)

            for spine in ax.spines:
                ax.spines[spine].set_color(c_lines)

            ax.get_yaxis().set_tick_params(color=c_plot_tick)
            ax.get_xaxis().set_tick_params(color=c_plot_tick)

        ax_foms.set_facecolor(c_plot_bg)
        ax_foms.grid(True, color=c_plot_grid, linewidth=0.5)
    else:
        c_text_main = '#000000'

    # title contains info of current iteration
    f.suptitle(''.join(['Iteration %d, ' % (Niter),
                     'FOMs = '] + ['%0.4f  ' % (fom) for fom in current_foms]),
            fontsize=14, color=c_text_main)
    lbd_p, lbd_s, lbd_i = lbds
    ax_field.text(0.6,0.2+H/3*2,'λi= %2.3fum' %lbd_i, fontsize=14)
    ax_field.text(0.6,0.2,'λp = %2.3fum' %lbd_p, fontsize=14)
    ax_field.text(0.6,0.2+H/3,'λs = %2.3fum' %lbd_s, fontsize=14)

    if(fname != ''):
        plt.tight_layout()
        if(dark):
            plt.savefig(fname, dpi=100, facecolor=c_bg_main)
        else:
            plt.savefig(fname, dpi=100)

    if(show_now):
        # plt.tight_layout()
        plt.show()