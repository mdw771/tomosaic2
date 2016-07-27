#!/usr/bin/env python
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
Module for input of tomosaic
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

logger = logging.getLogger(__name__)

__author__ = "Rafael Vescovi"
__credits__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['get_files',
	   'get_index',
	   'save_partial_frames',
	   'save_partial_flats',
           'save_partial_darks',
	   'build_panorama']


import os, glob, re
import h5py
import numpy as np
import tomopy
import dxchange
from tomosaic.merge.merge import *

def get_files(folder, prefix, type='.h5'):
    os.chdir(folder)
    file_list = glob.glob(prefix + '*' + type)
    return file_list


def get_index(file_list):
    regex = re.compile(r".+_x(\d\d)_y(\d\d).+")
    ind_buff = [m.group(1, 2) for l in file_list for m in [regex.search(l)] if m]
    return np.asarray(ind_buff).astype('int')

def save_partial_frames(file_grid, save_folder, prefix, frame=0):
    for (y, x), value in np.ndenumerate(file_grid):
	print(value)
        if (value != None):
            prj, flt, drk = dxchange.read_aps_32id(value, proj=(frame, frame + 1))
            prj = tomopy.normalize(prj, flt, drk)
	    prj = -np.log(prj).astype('float32')
            fname = prefix + 'Y' + str(y).zfill(2) + '_X' + str(x).zfill(2)
            dxchange.write_tiff(np.squeeze(prj), fname=os.path.join(save_folder, 'partial_frames', fname))

def save_partial_flats(file_grid, save_folder, prefix):
    for (y, x), value in np.ndenumerate(file_grid):
        if (value != None):
            _, flt, _ = dxchange.read_aps_32id(value, proj=(0,1))
            fname = prefix + 'Y' + str(y).zfill(2) + '_X' + str(x).zfill(2)
	    flt = flt.mean(axis=0).astype('float16')
            dxchange.write_tiff(np.squeeze(flt), fname=os.path.join(save_folder, 'partial_flats', fname))

def save_partial_darks(file_grid, save_folder, prefix):
    for (y, x), value in np.ndenumerate(file_grid):
        if (value != None):
            _, _, drk = dxchange.read_aps_32id(value, proj=(0,1))
            fname = prefix + 'Y' + str(y).zfill(2) + '_X' + str(x).zfill(2)
	    drk = drk.mean(axis=0).astype('float16')
            dxchange.write_tiff(np.squeeze(drk), fname=os.path.join(save_folder, 'partial_darks', fname))

g_shapes = lambda fname: h5py.File(fname, "r")['exchange/data'].shape

def build_panorama(file_grid, shift_grid, frame=0, cam_size=[2048, 2448],method='max'):
    img_size = shift_grid[-1, -1] + cam_size
    buff = np.zeros(img_size, dtype='float16')
    for (y, x), value in np.ndenumerate(file_grid):
        if (value != None and frame < g_shapes(value)[0]):
            prj, flt, drk = dxchange.read_aps_32id(value, proj=(frame, frame + 1))
            prj = tomopy.normalize(prj, flt[:, :, :], drk)
            prj[np.abs(prj) < 2e-3] = 2e-3
            prj[prj > 1] = 1
            prj = -np.log(prj).astype('float16')
            prj[np.where(np.isnan(prj) == True)] = 0
            buff = blend(buff, np.squeeze(prj), shift_grid[y, x, :],method=method)
    return buff


def grid2file(grid, file_name):
   
    with file(file_name, 'w') as outfile:        outfile.write('# Grid shape: {0}\n'.format(data.shape))
        outfile.write('#Vertical Values')             for data_slice in data:
                np.savetxt(outfile, data_slice, fmt='%-7.2f')        outfile.write('# New slice\n')
   return
   
def file2grid(file):
   return grid

