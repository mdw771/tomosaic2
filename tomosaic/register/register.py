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
import sys

import numpy as np
from numpy.fft import fft2, ifft2
from scipy import ndimage

logger = logging.getLogger(__name__)

__author__ = "Rafael Vescovi"
__credits__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['calculate_matrix',
           'optimize_matrix',
           'cross_correlation_bf,'
           'cross_correlation_pcm']


def cross_correlation_bf(img1, img2, rangeX=None, rangeY=None):
    """
    Find the cartesian shift vector between two images

    Parameters
    ----------

    img1 : ndarray
        Input array.
    img2 : ndarray
        Input array.

    rangeX, rangeY : integer, optional
        Mininum and maximum to search for minima.

    Returns
    -------
    ndarray
        shift array.
    """
    new_x, new_y = map(max, img2.shape, img1.shape)
    new_shape = [new_y, new_x]
    if rangeX is None:
        rangeX = [0, new_x]
    if rangeY is None:
        rangeY = [0, new_y]
    newimg1 = np.zeros(new_shape)
    newimg1[:img1.shape[0], :img1.shape[1]] = img1
    newimg2 = np.zeros(new_shape)
    newimg2[:img2.shape[0], :img2.shape[1]] = img2
    sqmean_img = np.ones(new_shape) * sys.maxint
    for kx in range(rangeX[0], rangeX[1]):
        for ky in range(rangeY[0], rangeY[1]):
            diff = newimg1[ky:, kx:] - newimg2[:new_y - ky, :new_x - kx]
            sqmean_img[ky, kx] = (diff ** 2).mean()
    y, x = np.unravel_index(np.argmin(sqmean_img), sqmean_img.shape)
    return [y, x]


def shift_bit_length(x):
    return 1 << (x - 1).bit_length()


def cross_correlation_pcm(img1, img2, rangeX=None, rangeY=None, blur=3):
    """
    Find the cartesian shift vector between two images

    Parameters
    ----------

    img1 : ndarray
        Input array.
    img2 : ndarray
        Input array.

    rangeX, rangeY : integer, optional
        Mininum and maximum to search for minima.
    blur : integer
        Blur Filter on the Phase Map

    Returns
    -------
    ndarray
        shift array.
    """
    new_size = shift_bit_length(max(map(max, img2.shape, img1.shape)))
    new_shape = [new_size, new_size]
    ##
    if rangeX is None:
        rangeX = [0, new_size]
    if rangeY is None:
        rangeY = [0, new_size]
    ##
    norm_max = np.max([img1.argmax(), img2.argmax()])
    src_image = np.array(img1, dtype=np.complex128, copy=False) / norm_max
    target_image = np.array(img2, dtype=np.complex128, copy=False) / norm_max
    f0 = fft2(src_image)
    f1 = fft2(target_image)
    cross_correlation = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    cross_correlation = ndimage.gaussian_filter(cross_correlation, sigma=blur)
    mask = np.zeros(cross_correlation.shape)
    mask[rangeY[0]:rangeY[1], rangeX[0]:rangeX[1]] = 1
    cross_correlation = cross_correlation * mask
    # Locate maximum
    y, x = np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape)
    return [y, x]
