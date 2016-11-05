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
Module for grid creation
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

logger = logging.getLogger(__name__)

__author__ = "Rafael Vescovi, Ming Du"
__credits__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['start_file_grid',
           'start_shift_grid',
           'shift2center_grid',
           'refine_shift_grid',
           'absolute_shift_grid']

import numpy as np
import h5py
import tomopy
import dxchange
from tomosaic.register.morph import *
from tomosaic.register.register import *
from tomosaic.register.register_translation import register_translation
from tomosaic.util.util import *
from scipy import ndimage
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()


def start_file_grid(file_list, ver_dir=0, hor_dir=0, pattern=0):
    ind_list = get_index(file_list, pattern)
    if pattern == 0:
        x_max, y_max = ind_list.max(0)
        x_min, y_min = ind_list.min(0)
    elif pattern == 1:
        x_max, y_max = ind_list.max(0) + 1
        x_min, y_min = ind_list.min(0) + 1
    grid = np.empty((y_max, x_max), dtype=object)
    for k_file in range(len(file_list)):
        if pattern == 0:
            grid[ind_list[k_file, 1] - 1, ind_list[k_file, 0] - 1] = file_list[k_file]
        elif pattern == 1:
            grid[ind_list[k_file, 1], ind_list[k_file, 0]] = file_list[k_file]
    if ver_dir:
        grid = np.flipud(grid)
    if hor_dir:
        grid = np.fliplr(grid)
    return grid


def start_shift_grid(file_grid, x_shift, y_shift):
    size_x = file_grid.shape[1]
    size_y = file_grid.shape[0]
    shift_grid = np.zeros([size_y, size_x, 2])
    for x_pos in range(size_x):
        shift_grid[:, x_pos, 1] = x_pos * x_shift
    ##
    for y_pos in range(size_y):
        shift_grid[y_pos, :, 0] = y_pos * y_shift
    ##
    return shift_grid


def shift2center_grid(shift_grid,center):
    center_grid = shift_grid[:,:,1] - center
    return center_grid



def find_pairs(file_grid):
    size_y, size_x = file_grid.shape
    pairs = np.empty((size_y * size_x, 4), dtype=object)
    for (y, x), value in np.ndenumerate(file_grid):
        nb1 = (y, x + 1)
        nb2 = (y + 1, x)
        nb3 = (y + 1, x + 1)
        if (x == size_x - 1):
            nb1 = None
            nb3 = None
        if (y == size_y - 1):
            nb2 = None
            nb3 = None
        pairs[size_x * y + x, :] = (y, x), nb1, nb2, nb3
    return pairs


g_shapes = lambda fname: h5py.File(fname, "r")['exchange/data'].shape

def refine_shift_grid(grid, shift_grid, step=200, upsample=100, y_mask=[-5,5], x_mask=[-5,5]):

    if (grid.shape[0] != shift_grid.shape[0] or
        grid.shape[1] != shift_grid.shape[1]):
        return
    pairs = find_pairs(grid)
    n_pairs = pairs.shape[0]

    pairs_shift = np.zeros([n_pairs, 6])

    file_per_rank = int(n_pairs / size)
    remainder = n_pairs % size
    if remainder:
        if rank == 0:
            print('You will have {:d} files that cannot be processed in parallel. Consider optimizing number of ranks. '
                  'Press anykey to continue.'
                  .format(remainder))
            anykey = raw_input()
    comm.Barrier()

    for stage in [0, 1]:
        if stage == 1 and rank != 0:
            pass
        else:
            fstart = rank * file_per_rank
            fend = (rank+1) * file_per_rank
            if stage == 1:
                fstart = size * file_per_rank
                fend = n_pairs
            for line in range(fstart, fend):
                if (grid[pairs[line, 0]] == None):
                    print ("###Block Inexistent")
                    continue
                print('###Line ' + str(line))
                main_pos = pairs[line, 0]
                main_shape = g_shapes(grid[main_pos])
                right_pos = pairs[line, 1]
                if (right_pos != None):
                    right_shape = g_shapes(grid[right_pos])
                else:
                    right_shape = [0,0,0]
                bottom_pos = pairs[line, 2]
                if (bottom_pos != None):
                    bottom_shape = g_shapes(grid[bottom_pos])
                else:
                    bottom_shape = [0,0,0]
                size_max = max(main_shape[0],right_shape[0],bottom_shape[0])
                prj, flt, drk = dxchange.read_aps_32id(grid[main_pos], proj=(0,size_max,step))
                prj = tomopy.normalize(prj, flt[10:15, :, :], drk)
                prj[np.abs(prj) < 2e-3] = 2e-3
                prj[prj > 1] = 1
                prj = -np.log(prj)
                prj[np.where(np.isnan(prj) == True)] = 0
                main_prj = vig_image(prj)
                pairs_shift[line, 0:2] = main_pos

                if (right_pos != None):
                    prj, flt, drk = dxchange.read_aps_32id(grid[right_pos], proj=(0, size_max, step))
                    prj = tomopy.normalize(prj, flt[10:15, :, :], drk)
                    prj[np.abs(prj) < 2e-3] = 2e-3
                    prj[prj > 1] = 1
                    prj = -np.log(prj)
                    prj[np.where(np.isnan(prj) == True)] = 0
                    right_prj = vig_image(prj)
                    shift_ini = shift_grid[right_pos] - shift_grid[main_pos]
                    rangeX = shift_ini[1] + x_mask
                    rangeY = shift_ini[0] + y_mask
                    right_vec = create_stitch_shift(main_prj, right_prj, rangeX, rangeY, down=0, upsample=upsample)
                    pairs_shift[line, 2:4] = right_vec


                if (bottom_pos != None):
                    prj, flt, drk = dxchange.read_aps_32id(grid[bottom_pos], proj=(0,size_max,step))
                    prj = tomopy.normalize(prj, flt[10:15, :, :], drk)
                    prj[np.abs(prj) < 2e-3] = 2e-3
                    prj[prj > 1] = 1
                    prj = -np.log(prj)
                    prj[np.where(np.isnan(prj) == True)] = 0
                    bottom_prj = vig_image(prj)
                    shift_ini = shift_grid[bottom_pos] - shift_grid[main_pos]
                    rangeX = shift_ini[1] + x_mask
                    rangeY = shift_ini[0] + y_mask
                    right_vec = create_stitch_shift(main_prj, bottom_prj, rangeX, rangeY, down=1, upsample=upsample)
                    pairs_shift[line, 4:6] = right_vec

        print(pairs_shift)
    #    new_grid = absolute_shift_grid(pairs_shift, grid)

    return pairs_shift


def create_stitch_shift(block1, block2, rangeX=None, rangeY=None, down=0, upsample=100):
    shift_vec = np.zeros([block1.shape[0], 2])
    shift_vec[0, :] = register_translation(block1.mean(0), block2.mean(0), rangeX=rangeX,
                                                   rangeY=rangeY, down=down, upsample_factor=upsample)

    #for frame in range(block1.shape[0]):
    #    shift_vec[frame, :] = register_translation(block1[frame, :, :], block2[frame, :, :], rangeX=rangeX,
    #                                               rangeY=rangeY, down=down, upsample_factor=100)
    #shift_vec2 = [reject_outliers(shift_vec[0]),reject_outliers(shift_vec[1])]

    #shift = np.mean(shift_vec, axis=0)
    shift = shift_vec[0, :]
    return shift


def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

# Generate absolute shift grid from a relative shift grid.
# Default building method is first right then down.
def absolute_shift_grid(pairs_shift, file_grid):
    shape = file_grid.shape
    abs_shift_grid = np.zeros([shape[0], shape[1], 2], dtype='float32')
    for (y, x), _ in np.ndenumerate(file_grid):
        if y == 0 and x == 0:
            pass
        else:
            for x_ind in range(0, x):
                abs_shift_grid[y, x, :] += pairs_shift[0, x_ind, 0:2]
            for y_ind in range(0, y):
                abs_shift_grid[y, x, :] += pairs_shift[y_ind, x, 2:4]
    return abs_shift_grid

