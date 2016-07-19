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

import numpy as np
import tomopy
import dxchange


def start_file_grid(file_list, ver_dir=0, hor_dir=0):
    ind_list = get_index(file_list)
    x_max, y_max = ind_list.max(0)
    x_min, y_min = ind_list.min(0)
    grid = np.empty((y_max, x_max), dtype=object)
    for k_file in range(len(file_list)):
        grid[ind_list[k_file, 1] - 1, ind_list[k_file, 0] - 1] = file_list[k_file]
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


def shift2center_grid():
    return 0



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
        pairs[size_x * y + x, :] = (y, x), nb1, nb2, nb3
    return pairs


g_shapes = lambda fname: h5py.File(fname, "r")['exchange/data'].shape

def refine_shift_grid(grid, shift_grid, step=100):
    if (grid.shape[0] != shift_grid.shape[0] or
                grid.shape[1] != shift_grid.shape[1]):
        return

    frame = 0
    pairs = find_pairs(grid)
    pairs_shift = pairs
    n_pairs = pairs.shape[0]

    for line in np.arange(n_pairs):
        print 'line ' + str(line), pairs[line, 0], pairs[line, 1], pairs[line, 2]
        main_pos = pairs[line, 0]
        main_shape = get_shape(grid[main_pos])
        right_pos = pairs[line, 1]
        right_shape = get_shape(grid[right_pos])
        bottom_pos = pairs[line, 2]
        bottom_shape = get_shape(grid[bottom_pos])
        diag_pos = pairs[line, 3]
        prj, flt, drk = dxchange.read_aps_32id(grid[main_pos], proj=(frame, frame + 1))
        if (main_pos[1] < 6):
            _, flt, _ = dxchange.read_aps_32id(grid[main_pos[0], 6], proj=(frame, frame + 1))
        prj = tomopy.normalize(prj, flt[20:, :, :], drk)
        prj[np.abs(prj) < 2e-3] = 2e-3
        prj[prj > 1] = 1
        prj = -np.log(prj)
        prj[np.where(np.isnan(prj) == True)] = 0
        main_prj = vig_image(prj)

        if (right_pos != None):
    prj, flt, drk = dxchange.read_aps_32id(grid[right_pos], proj=(frame, frame + 1))
            if (right_pos[0] < 6):
                _, flt, _ = dxchange.read_aps_32id(grid[right_pos[0], 6], proj=(frame, frame + 1))
    prj = tomopy.normalize(prj, flt[20:, :, :], drk)
    prj[np.abs(prj) < 2e-3] = 2e-3
            prj[prj > 1] = 1
            prj = -np.log(prj)
    prj[np.where(np.isnan(prj) == True)] = 0
    right_prj = vig_image(prj)
    shift_ini = shift_grid[right_pos] - shift_grid[main_pos]
    rangeX = shift_ini[1] + [-10, 10]
    rangeY = shift_ini[0] + [0, 5]
    right_vec = create_stitch_shift(main_prj, right_prj, rangeX, rangeY)
    pairs_shift[line, 1] = right_vec


if (bottom_pos != None):
    prj, flt, drk = dxchange.read_aps_32id(grid[bottom_pos], proj=(frame, frame + 1))
            if (bottom_pos[0] < 6):
                _, flt, _ = dxchange.read_aps_32id(grid[bottom_pos[0], 6], proj=(frame, frame + 1))
    prj = tomopy.normalize(prj, flt[20:, :, :], drk)
    prj[np.abs(prj) < 2e-3] = 2e-3
            prj[prj > 1] = 1
            prj = -np.log(prj)
    prj[np.where(np.isnan(prj) == True)] = 0
    bottom_prj = vig_image(prj)
    shift_ini = shift_grid[bottom_pos] - shift_grid[main_pos]
    rangeX = shift_ini[1] + [0, 10]
    rangeY = shift_ini[0] + [-5, 5]
    right_vec = create_stitch_shift(main_prj, bottom_prj, rangeX, rangeY)
    pairs_shift[line, 2] = right_vec
    return pairs_shift

def create_stitch_shift(block1, block2, rangeX=None, rangeY=None):
    shift_vec = np.zeros([block1.shape[0], 2])
    for frame in range(block1.shape[0]):
        shift_vec[frame, :] = cross_correlation_pcm(block1[frame, :, :], block2[frame, :, :], rangeX=rangeX,
                                                    rangeY=rangeY)
    print shift_vec
    shift = np.mean(shift_vec, 0)
    print shift
