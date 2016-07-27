T#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""
Module for image merging
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

logger = logging.getLogger(__name__)

__author__ = "Rafael Vescovi"
__credits__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['']


# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:57:31 2016

@author: ravescovi
"""


import glob, time, itertools
import numpy as np
import tomopy
import dxchange



def recon_block(grid, shift_grid, grid_lines, slices, sino_step)
    grid_line_ini = grid_lines[0]
    grid_line_end = grid_lines[1]
    sino_ini= slices[0]
    sino_end = slices[1]

    for grid_line in range(grid_line_ini,grid_line_end+1):
        grid_wd = sample_name + '*_y' + str(grid_line).zfill(2) +'*.h5'
    
        for sino_n in range (sino_ini,sino_end,sino_step):
            print('############################################')
            print('RECONSTRUCTION GRID '+str(grid_line)+' SINO '+str(sino_n))
            recon_slice(grid, shift_grid, grid_line, sino_n, save_folder)    

def load_sino(grid, shift_grid, grid_line, slice_n, save_folder):
    sino, flt, drk = tomopy.read_aps_32id(file_list[kfile], sino=(sino_n,sino_n+1))
    sino = tomopy.normalize(sino, flt, drk)
    return = np.squeeze(sino)



def register_recon(grid, center_grid, ray_size, grid_line, slice_n, save_folder):


    t = time.time()
    sinos = [[]] * len(file_list)
    for x_pos in np.arange(grid.shape[1])):
	t 

    print('file read:           ' + str(time.time() - t))

    # data stitching
    t = time.time()

    for kfile, k in itertools.izip(range(len(file_list)-1,0,-1),range(len(file_list)-1)):
        buff= blend(buff,sinos[kfile-1],[0,(k+1)*x_shift])
    sino = buff.reshape([buff.shape[0],1,buff.shape[1]])
    sino = sino[:,:,:22496].astype('float32')
    print('stitch:           ' + str(time.time() - t))
    print('final size:       ' + str(sino.shape))

    # data downsampling
    t = time.time()
    sino = tomopy.downsample(sino, level=ds_level)
    print('downsample:           ' + str(time.time() - t))
    print('new shape :           ' + str(sino.shape))

    # remove stripes
    t = time.time()
    sino = tomopy.remove_stripe_fw(sino,2)
    print('strip removal:           ' + str(time.time() - t))
    # Minus Log
    sino[np.abs(sino)< 1e-3] = 1
    sino[sino > 1] = 1
    sino = -np.log(sino)
    sino[np.where( np.isnan(sino) == True )] = 0

        #save the sinogram
    tomopy.io.writer.write_tiff(np.squeeze(sino), fname=save_folder+'/sinos/sino_'+str(grid_line)+'_'+str(sino_n))

    t = time.time()
    ang = tomopy.angles(sino.shape[0])


        #GRIDREC
    rec = tomopy.recon(sino, ang, center=center_pos, algorithm='gridrec', filter_name='parzen')
    tomopy.io.writer.write_tiff(rec, fname=save_folder+'/recon/gridrec_'+str(grid_line)+'_'+str(sino_n))

	#SIRT-FBP
	#import astra, sirtfbp
	#astra.plugin.register(sirtfbp.plugin)
	#extra_options = {'filter_dir':'./filters'}
	#num_iter = 100
	#print 'iterations:           ' + str(num_iter)
	#rec = tomopy.recon(sino, ang, center=center_pos, algorithm=tomopy.astra,
	#	           options={'proj_type':'cuda','method':'SIRT-FBP','extra_options':extra_options,'num_iter':num_iter})
	#tomopy.io.writer.write_tiff_stack(rec, fname=save_folder+'/sirtfbp'+str(num_iter)+'_'+str(grid_line)+'_'+str(sino_n))

	#TV
	#import astra, tvtomo
	#num_iter=100
	#print 'iterations:           ' + str(num_iter)
	#astra.plugin.register(tvtomo.plugin)
	#print astra.plugin.get_help('TV-FISTA')
	#rec = tomopy.recon(sino, ang, center=center_pos, algorithm=tomopy.astra,
	#	           options={'method':'TV-FISTA', 'proj_type':'cuda', 'num_iter':num_iter,
	#	                    'extra_options':{'tv_reg':0.000005,'bmin':0.0,'print_progress':True}})
	#tomopy.io.writer.write_tiff_stack(rec, fname=save_folder+'/fista'+str(num_iter)+'_'+str(grid_line)+'_'+str(sino_n))

	print('recon:           ' + str(time.time() - t))


