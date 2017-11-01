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
Module for Phase Retrieval
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import dxchange
import logging
import tomopy

logger = logging.getLogger(__name__)

__author__ = ['Rafael Vescovi', 'Ming Du']
__credits__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['retrieve_phase']


def retrieve_phase(data, method='mba', **kwargs):

    allowed_kwargs = {'mba': ['pixel', 'alpha'],
                      'paganin': ['distance', 'pixel', 'energy', 'alpha']}
    if method not in allowed_kwargs:
        raise ValueError
    for key, value in list(kwargs.items()):
        if key not in allowed_kwargs[method]:
            raise ValueError('Invalid options for selected phase retrieval method.')
    func = _get_func(method)
    data[np.isnan(data)] = 0
    return func(data, **kwargs)


def _get_func(method):
    if method == 'mba':
        func = mba
    if method == 'paganin':
        func = paganin
    return func


def gen_mesh(max, shape):
    """Generate mesh grid.

    Parameters:
    -----------
    lengths : ndarray
        Half-lengths of axes in nm or nm^-1.
    shape : ndarray
        Number of pixels in each dimension.
    """
    yy = np.linspace(-max[0], max[0], shape[0])
    xx = np.linspace(-max[1], max[1], shape[1])
    res = np.meshgrid(xx, yy)
    return res


def mba(input, pixel=1e-4, alpha=0.001):
    """Modified Bronnikov algorithm for phase retrieval.
    """
    assert isinstance(input, np.ndarray)
    g = input - 1
    u_max = 1. / (2. * pixel)
    u, v = gen_mesh([u_max, u_max], input.shape)
    H = 1 / (u ** 2 + v ** 2 + alpha)
    phase = np.real(ifftn(ifftshift(fftshift(fftn(g)) * H)))
    return phase


def paganin(input, pixel=1e-4, distance=50, energy=25, alpha=1e-4):
    assert input.ndim == 2
    input = input.reshape([1, input.shape[0], input.shape[1]]).astype('float32')
    res = tomopy.retrieve_phase(input, pixel_size=pixel, dist=distance, energy=energy, alpha=alpha)
    res = np.squeeze(res)
    return res


def _get_pr_kwargs():

    return {'pixel': 1e-4, 'distance': 50, 'energy': 25, 'alpha_paganin':1e-4}