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
import operator

import numpy as np

logger = logging.getLogger(__name__)

__author__ = "Rafael Vescovi"
__credits__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['img_merge_alpha',
           'img_merge_tv',
           'block_merge']


def blend(img1, img2, shift, method):
    return 0

def img_merge_alpha(img1, img2, shift, alpha=0.5):
    """
    Change dynamic range of values in an array.

    Parameters
    ----------
        Output array.
        :param img1:
        :param img2:
        :param shift:
        :param alpha:

    Returns
    -------
    ndarray
    """
    new_shape = map(max, map(operator.add, img2.shape, shift), img1.shape)
    newimg1 = np.zeros(new_shape)
    newimg1[0:img1.shape[0], 0:img1.shape[1]] = img1
    newimg1[shift[0]:, shift[1]:] = img2

    newimg2 = np.zeros(new_shape)
    newimg2[shift[0]:, shift[1]:] = img2
    newimg2[0:img1.shape[0], 0:img1.shape[1]] = img1

    final_img = alpha * newimg1 + (1 - alpha) * newimg2
    return final_img


def img_merge_max(img1, img2, shift):
    """
    Change dynamic range of values in an array.

    Parameters
    ----------
        :param img1:
        :param img2:
        :param shift:

    Returns
    -------
    ndarray
        Output array.
    """
    new_shape = map(max, map(operator.add, img2.shape, shift), img1.shape)
    newimg1 = np.zeros(new_shape)
    newimg1[0:img1.shape[0], 0:img1.shape[1]] = img1
    newimg1[shift[0]:, shift[1]:] = img2

    newimg2 = np.zeros(new_shape)
    newimg2[shift[0]:, shift[1]:] = img2
    newimg2[0:img1.shape[0], 0:img1.shape[1]] = img1

    buff = np.dstack((newimg1, newimg2))
    final_img = buff.max(2)

    return final_img


# Advanced blending function based on Perez et al., Poisson Image Editing.
# NOTE: DO NEWIMG(NP.ISNAN(NEWIMG)) = NEWIMG(NP.INVERT(NP.ISNAN(NEWIMG))).MIN AFTER STITCHING ALL IMAGES
def img_merge_poisson(img1, img2, shift):
    new_shape = map(max, map(operator.add, img2.shape, shift), img1.shape)
    # img2 covers img1.
    newimg = np.empty(new_shape, dtype='float32')
    # Initialize with NaN to prevent confusion with 0 in images
    newimg[:, :] = np.NaN
    newimg[0:img1.shape[0], 0:img1.shape[1]] = img1
    # Get corner positions for img2 INCLUDING boundary.
    shape = np.squeeze(img2.shape)
    shape_full = np.squeeze(newimg.shape)
    corner = get_corner(shift, shape)
    newimg[corner[0, 0]:corner[1, 0] + 1, corner[0, 1]:corner[1, 1] + 1] = img2
    img2_boo_part = find_bound(shape, corner, newimg)
    img2_boo = np.ones([shape[0], shape[1]], dtype='bool')
    img2_boo[0, :] = False
    img2_boo[:, 0] = False
    img2_boo[-1, :] = False
    img2_boo[:, -1] = False
    # Overwrite overlapping boundary with img1
    bound_y, bound_x = np.nonzero(np.invert(img2_boo_part))
    bound_y += corner[0, 0]
    bound_x += corner[0, 1]
    # for i, j in izip(bound_y, bound_x):
    #     newimg[i, j] = img1[i, j]
    newimg[[bound_y, bound_x]] = img1[[bound_y, bound_x]]
    # Embroider non-overlapping part with blurred img2
    bound_y, bound_x = np.nonzero(np.invert(img2_boo) - np.invert(img2_boo_part))
    img2_blur = cv2.GaussianBlur(img2, (0, 0), 10)
    bound_y += corner[0, 0]
    bound_x += corner[0, 1]
    newimg[[bound_y, bound_x]] = img2_blur[[bound_y - corner[0, 0], bound_x - corner[0, 1]]]
    ##
    spot = newimg[corner[0, 0]:corner[1, 0] + 1, corner[0, 1]:corner[1, 1] + 1]
    print
    "    Blend: Building matrix... ",
    t0 = time.time()
    A = matrix_builder(img2_boo)
    print
    "Done in " + str(time.time() - t0) + " sec."
    print
    "    Blend: Building constant vector... ",
    t0 = time.time()
    b = const_builder(img2_boo, spot, img2)
    print
    "Done in " + str(time.time() - t0) + " sec."
    print
    "    Blend: Solving linear system... ",
    t0 = time.time()
    x = lng.bicg(A, b)[0]
    print
    "Done in " + str(time.time() - t0) + " sec."
    spot[img2_boo] = x
    newimg[corner[0, 0]:corner[1, 0] + 1, corner[0, 1]:corner[1, 1] + 1] = spot
    return newimg


# Return a Boolean matrix with equal size to img2, with True for interior and False for boundary (d-Omega).
# shape: shape array of img2.
# corner: corner pixel indices of img2 in full image space.
def find_bound(shape, corner, newimg):
    img2_boo = np.ones(shape).astype('bool')
    newimg_expand = np.zeros([newimg.shape[0] + 2, newimg.shape[1] + 2])
    newimg_expand[:, :] = np.NaN
    newimg_expand[1:-1, 1:-1] = newimg
    corner = corner + [[1, 1], [1, 1]]
    # Top edge
    for i, j in izip(range(corner[0, 1], corner[1, 1] + 1), range(shape[1])):
        if not np.isnan(newimg_expand[corner[0, 0] - 1, i]):
            img2_boo[0, j] = False
    # Right edge
    for i, j in izip(range(corner[0, 0], corner[1, 0] + 1), range(shape[0])):
        if not np.isnan(newimg_expand[i, corner[1, 1] + 1]):
            img2_boo[j, -1] = False
    # Bottom edge
    for i, j in izip(range(corner[0, 1], corner[1, 1] + 1), range(shape[1])):
        if not np.isnan(newimg_expand[corner[1, 0] + 1, i]):
            img2_boo[-1, j] = False
    # Left edge
    for i, j in izip(range(corner[0, 0], corner[1, 0] + 1), range(shape[0])):
        if not np.isnan(newimg_expand[i, corner[0, 1] - 1]):
            img2_boo[j, 0] = False
    return img2_boo


# Return coordinates of the top right and bottom left pixels of an image in the expanded full image space. Both
# pixels are WITHIN the domain of the pasted image.
def get_corner(shift, img2_shape):
    corner_uly, corner_ulx, corner_bry, corner_brx = (shift[0], shift[1], shift[0] + img2_shape[0] - 1,
                                                      shift[1] + img2_shape[1] - 1)
    return np.squeeze([[corner_uly, corner_ulx], [corner_bry, corner_brx]]).astype('int')


# Build sparse square matrix A in Poisson equation Ax = b.
def matrix_builder(img2_boo):
    n_mat = np.count_nonzero(img2_boo)
    shape = img2_boo.shape
    img2_count = np.zeros([shape[0], shape[1]])
    img2_count[:, :] = 4
    img2_count[:, 0] -= 1
    img2_count[0, :] -= 1
    img2_count[:, -1] -= 1
    img2_count[-1, :] -= 1
    data = img2_count[img2_boo]
    y_ind = np.arange(n_mat)
    x_ind = np.arange(n_mat)
    ##
    img2_count_expand = np.zeros([shape[0] + 2, shape[1] + 2], dtype='int')
    img2_count_expand[1:-1, 1:-1] = img2_boo.astype('int')
    img2_u = np.roll(img2_count_expand, 1, axis=0)
    img2_d = np.roll(img2_count_expand, -1, axis=0)
    img2_l = np.roll(img2_count_expand, 1, axis=1)
    img2_r = np.roll(img2_count_expand, -1, axis=1)
    img2_u = img2_u[1:-1, 1:-1]
    img2_d = img2_d[1:-1, 1:-1]
    img2_l = img2_l[1:-1, 1:-1]
    img2_r = img2_r[1:-1, 1:-1]
    img2_u = img2_u[img2_boo]
    img2_d = img2_d[img2_boo]
    img2_l = img2_l[img2_boo]
    img2_r = img2_r[img2_boo]
    row_int = shape[1] - 2
    count = 0
    y_ind_app = np.squeeze(np.nonzero(img2_u))
    x_ind_app = y_ind_app - row_int
    y_ind = np.append(y_ind, y_ind_app)
    x_ind = np.append(x_ind, x_ind_app)
    count += len(x_ind_app)
    y_ind_app = np.squeeze(np.nonzero(img2_d))
    x_ind_app = y_ind_app + row_int
    y_ind = np.append(y_ind, y_ind_app)
    x_ind = np.append(x_ind, x_ind_app)
    count += len(x_ind_app)
    y_ind_app = np.squeeze(np.nonzero(img2_l))
    x_ind_app = y_ind_app - 1
    y_ind = np.append(y_ind, y_ind_app)
    x_ind = np.append(x_ind, x_ind_app)
    count += len(x_ind_app)
    y_ind_app = np.squeeze(np.nonzero(img2_r))
    x_ind_app = y_ind_app + 1
    y_ind = np.append(y_ind, y_ind_app)
    x_ind = np.append(x_ind, x_ind_app)
    count += len(x_ind_app)
    data_app = np.zeros(count)
    data_app[:] = -1
    data = np.append(data, data_app)
    A = csc_matrix((data, (y_ind, x_ind)), shape=(n_mat, n_mat))
    return A


# Build the constant column b in Ax = b. Panorama is built from left to right, from top tp bottom by default.
# img1_bound can be any matrix with equal size to img2 as long as the boundary position is filled with img1.
def const_builder(img2_boo, img1_bound, img2):
    n_mat = np.count_nonzero(img2_boo)
    shape = img2_boo.shape
    img2_bound_boo = np.invert(img2_boo)
    img2_bound = np.zeros([shape[0], shape[1]])
    img2_bound[img2_bound_boo] = img1_bound[img2_bound_boo]
    img2_bound_expand = np.zeros([shape[0] + 2, shape[1] + 2])
    img2_bound_expand[1:-1, 1:-1] = img2_bound
    img2_bound_expand = circ_neighbor(img2_bound_expand)
    img2_bound = img2_bound_expand[1:-1, 1:-1]
    b = img2_bound[img2_boo]
    ##
    img2_expand = np.zeros([shape[0] + 2, shape[1] + 2])
    img2_expand[1:-1, 1:-1] = img2
    img2_expand = 4 * img2_expand - circ_neighbor(img2_expand)
    img2 = img2_expand[1:-1, 1:-1]
    b += img2[img2_boo]
    return b


# Find the sum of neighbors assuming periodic boundary. Pad the input matrix with 0 when necessary.
def circ_neighbor(mat):
    return np.roll(mat, 1, axis=0) + np.roll(mat, -1, axis=0) + np.roll(mat, 1, axis=1) + np.roll(mat, -1, axis=1)
