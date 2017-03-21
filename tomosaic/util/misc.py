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
import h5py
import numpy as np
import dxchange
import re
import os
import glob


logger = logging.getLogger(__name__)

__author__ = ["Rafael Vescovi", "Ming Du"]
__credits__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['allocate_mpi_subsets',
           'read_aps_32id_adaptive']


def allocate_mpi_subsets(n_task, size, task_list=None):

    if task_list is None:
        task_list = range(n_task)
    sets = []
    per_rank_floor = int(np.floor(n_task/size))
    remainder = n_task % size
    start = 0
    for i in range(size):
        length = per_rank_floor
        if remainder > 0:
            length += 1
            remainder -= 1
        sets.append(task_list[start:start+length])
        start = start + length
    return sets


def which_tile(shift_grid, file_grid, x_coord, y_coord):

    f = h5py.File(file_grid[0, 0])
    y_cam, x_cam = f['exchange/data'].shape[1:3]
    y_prev = 'none'
    x_prev = 'none'
    y = np.searchsorted(shift_grid[:, 0], y_coord) - 1
    if y != 0:
        if y < shift_grid[y-1, 0] + y_cam:
            y_prev = y - 1
        else:
            y_prev = y
    x = np.searchsorted(shift_grid[0, :], x_coord) - 1
    if x != 0:
        if x < shift_grid[0, x-1] + x_cam:
            x_prev = x - 1
        else:
            x_prev = x
    s = file_grid[y, x]
    if x_prev != x or y_prev != y:
        s = s + ' overlapped with ' + file_grid[y_prev, x_prev]
    return s


def entropy(img, range=(-0.002, 0.002)):

    hist, e = np.histogram(img, bins=1024, range=range)
    hist = hist.astype('float32') / img.size + 1e-12
    val = -np.dot(hist, np.log2(hist))
    return val


def minimum_entropy(folder, pattern='*.tiff', range=(-0.002, 0.002)):

    flist = glob.glob(os.path.join(folder, pattern))
    a = []
    s = []
    for fname in flist:
        img = dxchange.read_tiff(fname)
        s.append(entropy(img, range=range))
        a.append(fname)
    return a[np.argmin(s)]




def read_aps_32id_adaptive(fname, proj=None, sino=None):
    """
    Adaptive data reading function that works with dxchange both below and beyond version 0.0.11.
    """
    dxver = dxchange.__version__
    m = re.search(r'(\d+)\.(\d+)\.(\d+)', dxver)
    ver = m.group(1, 2, 3)
    ver = map(int, ver)
    if ver[0] > 0 or ver[1] > 1 or ver[2] > 1:
        dat, flt, drk, _ = dxchange.read_aps_32id(fname, proj=proj, sino=sino)
    else:
        dat, flt, drk = dxchange.read_aps_32id(fname, proj=proj, sino=sino)

    return dat, flt, drk