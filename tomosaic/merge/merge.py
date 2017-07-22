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
import six
import operator, time
import numpy as np
import scipy
from numpy.linalg import norm
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as lng
from itertools import izip
import gc
import tomosaic.register.morph as morph
import dxchange
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

__author__ = "Rafael Vescovi, Ming Du"
__credits__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['blend',
           'img_merge_alpha',
           'img_merge_overlay',
           'img_merge_max',
           'img_merge_min',
           'img_merge_poisson',
           'img_merge_pyramid',
           'img_merge_pwd']


def blend(img1, img2, shift, method, margin=50, color_correction=True, **kwargs):
    """
    Blend images.
    Parameters
    ----------
    img1, img2 : ndarray
        3D tomographic data.
    shift : array
        Projection angles in radian.

    method : {str, function}
        One of the following string values.
        'alpha'
        'max'
        'min'
        'poison'
        'pyramid'

    filter_par: list, optional
        Filter parameters as a list.
    Returns
    -------
    ndarray
        Reconstructed 3D object.
    """

    allowed_kwargs = {
        'alpha': ['alpha'],
        'max': [],
        'min': [],
        'poisson': [],
        'pyramid': ['blur', 'margin', 'depth'],
        'pwd': ['margin', 'chunk_size'],
        'overlay': ['order']
    }

    generic_kwargs = []
    # Generate kwargs for the algorithm.
    kwargs_defaults = _get_algorithm_kwargs()

    if 'margin' not in kwargs and 'margin' in allowed_kwargs[method]:
        kwargs.update({'margin': margin})

    if isinstance(method, six.string_types):

        # Check whether we have an allowed method
        if method not in allowed_kwargs:
            raise ValueError(
                'Keyword "method" must be one of %s, or a Python method.' %
                (list(allowed_kwargs.keys()),))

        # Make sure have allowed kwargs appropriate for algorithm.
        for key, value in list(kwargs.items()):

            if key not in allowed_kwargs[method]:
                raise ValueError(
                    '%s keyword not in allowed keywords %s' %
                    (key, allowed_kwargs[method]))
            else:
                # Make sure they are numpy arrays.
                if not isinstance(kwargs[key], (np.ndarray, np.generic)) and not isinstance(kwargs[key],
                                                                                            six.string_types):
                    kwargs[key] = np.array(value)

                # Make sure reg_par and filter_par is float32.
                if key == 'alpha':
                    if not isinstance(kwargs[key], np.float32):
                        kwargs[key] = np.array(value, dtype='float32')
                        # if key == 'blur':
                        #     if not isinstance(kwargs[key], np.float32):
                        #         kwargs[key] = np.array(value, dtype='float32')
                        # Set kwarg defaults.

    elif hasattr(method, '__call__'):
        # Set kwarg defaults.
        for kw in generic_kwargs:
            kwargs.setdefault(kw, kwargs_defaults[kw])
    else:
        raise ValueError(
            'Keyword "method" must be one of %s, or a Python method.' %
            (list(allowed_kwargs.keys()),))

    func = _get_func(method)

    if color_correction:
        try:
            if img1.size > 1:
                img2 = correct_luminance(img1, img2, shift, margin=margin)
        except:
            pass

    return func(img1, img2, shift, **kwargs)


def _get_func(method):
    if method == 'alpha':
        func = img_merge_alpha
    elif method == 'overlay':
        func = img_merge_overlay
    elif method == 'max':
        func = img_merge_max
    elif method == 'min':
        func = img_merge_min
    elif method == 'poisson':
        func = img_merge_poisson
    elif method == 'pyramid':
        func = img_merge_pyramid
    elif method == 'pwd':
        func = img_merge_pwd
    return func


def _get_algorithm_kwargs():
    return {'alpha': 0.5, 'blur': 0.4, 'depth': 7, 'order': 1}


def img_merge_alpha(img1, img2, shift, alpha=0.4, margin=100):
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
    newimg, img2 = morph.arrange_image(img1, img2, shift)
    case, rough_shift, corner, buffer1, buffer2, wid_hor, wid_ver = find_overlap(img1, img2, shift, margin=margin)
    buffer = np.dstack((buffer1, buffer2))
    final_img = buffer[:, :, 0] * alpha + buffer[:, :, 1] * (1 - alpha)
    if abs(rough_shift[1]) > margin and abs(rough_shift[0]) > margin:
        newimg[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + mask2.shape[1]] = \
            final_img[:wid_ver, :]
        newimg[corner[0, 0] + wid_ver:corner[0, 0] + mask2.shape[0], corner[0, 1]:corner[0, 1] + wid_hor] = \
            final_img[wid_ver:, :wid_hor]
    else:
        newimg[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + wid_hor] = final_img

    return newimg


def img_merge_overlay(img1, img2, shift):
    """
    Simple overlay of two images. Equivalent to alpha blending with alpha = 1.
    """
    newimg, _ = morph.arrange_image(img1, img2, shift, order=1)
    return newimg


def img_merge_max(img1, img2, shift, margin=100):
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
    newimg, img2 = morph.arrange_image(img1, img2, shift)
    case, rough_shift, corner, buffer1, buffer2, wid_hor, wid_ver = find_overlap(img1, img2, shift, margin=margin)
    buffer = np.dstack((buffer1, buffer2))
    final_img = buffer.max(-1)
    if abs(rough_shift[1]) > margin and abs(rough_shift[0]) > margin:
        newimg[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + mask2.shape[1]] = \
            final_img[:wid_ver, :]
        newimg[corner[0, 0] + wid_ver:corner[0, 0] + mask2.shape[0], corner[0, 1]:corner[0, 1] + wid_hor] = \
            final_img[wid_ver:, :wid_hor]
    else:
        newimg[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + wid_hor] = final_img

    return newimg


def img_merge_min(img1, img2, shift, margin=100):
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
    newimg, img2 = morph.arrange_image(img1, img2, shift)
    case, rough_shift, corner, buffer1, buffer2, wid_hor, wid_ver = find_overlap(img1, img2, shift, margin=margin)
    buffer = np.dstack((buffer1, buffer2))
    final_img = buffer.min(-1)
    if abs(rough_shift[1]) > margin and abs(rough_shift[0]) > margin:
        newimg[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + mask2.shape[1]] = \
            final_img[:wid_ver, :]
        newimg[corner[0, 0] + wid_ver:corner[0, 0] + mask2.shape[0], corner[0, 1]:corner[0, 1] + wid_hor] = \
            final_img[wid_ver:, :wid_hor]
    else:
        newimg[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + wid_hor] = final_img

    return newimg


# Modified for subpixel fourier shift.
def img_merge_poisson(img1, img2, shift):
    newimg, img2 = morph.arrange_image(img1, img2, shift)
    if abs(shift[0]) < 10 and abs(shift[1]) < 10:
        return newimg
    # Get corner positions for img2 INCLUDING boundary.
    shape = np.squeeze(img2.shape)
    corner = _get_corner(morph.get_roughshift(shift), shape)
    img2_boo_part = _find_bound(shape, corner, newimg)
    img2_boo = np.ones([shape[0], shape[1]], dtype='bool')
    img2_boo[0, :] = False
    img2_boo[:, 0] = False
    img2_boo[-1, :] = False
    img2_boo[:, -1] = False
    # Overwrite overlapping boundary with img1
    bound_y, bound_x = np.nonzero(np.invert(img2_boo_part))
    bound_y += corner[0, 0]
    bound_x += corner[0, 1]
    newimg[[bound_y, bound_x]] = img1[[bound_y, bound_x]]
    # Embroider non-overlapping part with blurred img2
    bound_y, bound_x = np.nonzero(np.invert(img2_boo) - np.invert(img2_boo_part))
    img2_blur = scipy.ndimage.filters.gaussian_filter(img2, 10)
    bound_y += corner[0, 0]
    bound_x += corner[0, 1]
    newimg[[bound_y, bound_x]] = img2_blur[[bound_y - corner[0, 0], bound_x - corner[0, 1]]]
    ##
    spot = newimg[corner[0, 0]:corner[1, 0] + 1, corner[0, 1]:corner[1, 1] + 1]
    print("    Blend: Building matrix... ", end="")
    t0 = time.time()
    A = _matrix_builder(img2_boo)
    print("Done in " + str(time.time() - t0) + " sec.")
    print("    Blend: Building constant vector... ", end="")
    t0 = time.time()
    b = _const_builder(img2_boo, spot, img2)
    print("Done in " + str(time.time() - t0) + " sec.")
    print("    Blend: Solving linear system... ", end="")
    t0 = time.time()
    x = lng.bicg(A, b)[0]
    print("Done in " + str(time.time() - t0) + " sec.")
    spot[img2_boo] = x
    newimg[corner[0, 0]:corner[1, 0] + 1, corner[0, 1]:corner[1, 1] + 1] = spot
    return newimg


# Return a Boolean matrix with equal size to img2, with True for interior and False for boundary (d-Omega).
# shape: shape array of img2.
# corner: corner pixel indices of img2 in full image space.
def _find_bound(shape, corner, newimg):
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
def _get_corner(shift, img2_shape):
    corner_uly, corner_ulx, corner_bry, corner_brx = (shift[0], shift[1], shift[0] + img2_shape[0] - 1,
                                                      shift[1] + img2_shape[1] - 1)
    return np.squeeze([[corner_uly, corner_ulx], [corner_bry, corner_brx]]).astype('int')


# Build sparse square matrix A in Poisson equation Ax = b.
def _matrix_builder(img2_boo):
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
def _const_builder(img2_boo, img1_bound, img2):
    n_mat = np.count_nonzero(img2_boo)
    shape = img2_boo.shape
    img2_bound_boo = np.invert(img2_boo)
    img2_bound = np.zeros([shape[0], shape[1]])
    img2_bound[img2_bound_boo] = img1_bound[img2_bound_boo]
    img2_bound_expand = np.zeros([shape[0] + 2, shape[1] + 2])
    img2_bound_expand[1:-1, 1:-1] = img2_bound
    img2_bound_expand = _circ_neighbor(img2_bound_expand)
    img2_bound = img2_bound_expand[1:-1, 1:-1]
    b = img2_bound[img2_boo]
    ##
    img2_expand = np.zeros([shape[0] + 2, shape[1] + 2])
    img2_expand[1:-1, 1:-1] = img2
    img2_expand = 4 * img2_expand - _circ_neighbor(img2_expand)
    img2 = img2_expand[1:-1, 1:-1]
    b += img2[img2_boo]
    return b


# Find the sum of neighbors assuming periodic boundary. Pad the input matrix with 0 when necessary.
def _circ_neighbor(mat):
    return np.roll(mat, 1, axis=0) + np.roll(mat, -1, axis=0) + np.roll(mat, 1, axis=1) + np.roll(mat, -1, axis=1)


def img_merge_pyramid(img1, img2, shift, margin=100, blur=0.4, depth=5):
    """
    Perform pyramid blending. Codes are adapted from Computer Vision Lab, Image blending using pyramid,
    https://compvisionlab.wordpress.com/2013/05/13/image-blending-using-pyramid/.
    Users are strongly suggested to run tests before beginning the actual stitching job using this function to determine
    the biggest depth value that does not give seams due to over-blurring.
    """

    t00 = time.time()
    t0 = time.time()
    # print(    'Starting pyramid blend...')
    newimg, img2 = morph.arrange_image(img1, img2, shift)
    if abs(shift[0]) < margin and abs(shift[1]) < margin:
        return newimg
    # print('    Blend: Image aligned and built in', str(time.time() - t0))

    t0 = time.time()
    case, rough_shift, corner, buffer1, buffer2, wid_hor, wid_ver = find_overlap(img1, img2, shift, margin=margin)
    if case == 'skip':
        return newimg
    mask2 = np.ones(buffer1.shape)
    if abs(rough_shift[1]) > margin:
        mask2[:, :int(wid_hor / 2)] = 0
    if abs(rough_shift[0]) > margin:
        mask2[:int(wid_ver / 2), :] = 0
    ##
    buffer1[np.isnan(buffer1)] = 0
    mask2[np.isnan(mask2)] = 1
    t0 = time.time()
    gauss_mask = _gauss_pyramid(mask2.astype('float'), depth, blur, mask=True)
    gauss1 = _gauss_pyramid(buffer1, depth, blur)
    gauss2 = _gauss_pyramid(buffer2, depth, blur)
    lapl1 = _lapl_pyramid(gauss1, blur)
    lapl2 = _lapl_pyramid(gauss2, blur)
    ovlp_blended = _collapse(_blend(lapl2, lapl1, gauss_mask), blur)
    # print('    Blend: Blending done in', str(time.time() - t0), 'sec.')

    if abs(rough_shift[1]) > margin and abs(rough_shift[0]) > margin:
        newimg[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + mask2.shape[1]] = \
            ovlp_blended[:wid_ver, :]
        newimg[corner[0, 0] + wid_ver:corner[0, 0] + mask2.shape[0], corner[0, 1]:corner[0, 1] + wid_hor] = \
            ovlp_blended[wid_ver:, :wid_hor]
    else:
        newimg[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + wid_hor] = ovlp_blended
    # print('    Blend: Done with this tile in', str(time.time() - t00), 'sec.')
    gc.collect()

    return newimg


def _generating_kernel(a):
    w_1d = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(w_1d, w_1d)


def _ireduce(image, blur):
    kernel = _generating_kernel(blur)
    outimage = scipy.signal.convolve2d(image, kernel, mode='same', boundary='symmetric')
    out = outimage[::2, ::2]
    return out


def _iexpand(image, blur):
    kernel = _generating_kernel(blur)
    outimage = np.zeros((image.shape[0] * 2, image.shape[1] * 2), dtype=np.float64)
    outimage[::2, ::2] = image[:, :]
    out = 4 * scipy.signal.convolve2d(outimage, kernel, mode='same', boundary='symmetric')
    return out


def _gauss_pyramid(image, levels, blur, mask=False):
    output = []
    if mask:
        image = gaussian_filter(image, 20)
    output.append(image)
    tmp = np.copy(image)
    for i in range(0, levels):
        tmp = _ireduce(tmp, blur)
        output.append(tmp)
    return output


def _lapl_pyramid(gauss_pyr, blur):
    output = []
    k = len(gauss_pyr)
    for i in range(0, k - 1):
        gu = gauss_pyr[i]
        egu = _iexpand(gauss_pyr[i + 1], blur)
        if egu.shape[0] > gu.shape[0]:
            egu = np.delete(egu, (-1), axis=0)
        if egu.shape[1] > gu.shape[1]:
            egu = np.delete(egu, (-1), axis=1)
        output.append(gu - egu)
    output.append(gauss_pyr.pop())
    return output


def _blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
    blended_pyr = []
    k = len(gauss_pyr_mask)
    for i in range(0, k):
        p1 = gauss_pyr_mask[i] * lapl_pyr_white[i]
        p2 = (1 - gauss_pyr_mask[i]) * lapl_pyr_black[i]
        blended_pyr.append(p1 + p2)
    return blended_pyr


def _collapse(lapl_pyr, blur):
    output = np.zeros((lapl_pyr[0].shape[0], lapl_pyr[0].shape[1]), dtype=np.float64)
    for i in range(len(lapl_pyr) - 1, 0, -1):
        lap = _iexpand(lapl_pyr[i], blur)
        lapb = lapl_pyr[i - 1]
        if lap.shape[0] > lapb.shape[0]:
            lap = np.delete(lap, (-1), axis=0)
        if lap.shape[1] > lapb.shape[1]:
            lap = np.delete(lap, (-1), axis=1)
        tmp = lap + lapb
        lapl_pyr.pop()
        lapl_pyr.pop()
        lapl_pyr.append(tmp)
        output = tmp
    return output


def img_merge_pwd(img1, img2, shift, margin=100, chunk_size=10000):
    t00 = time.time()
    t0 = time.time()
    newimg, _ = morph.arrange_image(img1, img2, shift, order=2)
    print('Starting PWD blend...')
    if abs(shift[0]) < margin and abs(shift[1]) < margin:
        return newimg
    rough_shift = morph.get_roughshift(shift)
    print('    Blend: Image aligned and built in', str(time.time() - t0))

    # determine scenario
    if abs(rough_shift[1]) > margin and abs(rough_shift[0]) > margin:
        rel_pos = 'lt'
    elif abs(rough_shift[1]) < margin and abs(rough_shift[0]) > margin:
        rel_pos = 't'
    else:
        rel_pos = 'l'

    # get overlapping area
    t0 = time.time()
    corner = _get_corner(rough_shift, img2.shape)
    # for new image with overlap at left and top
    if rel_pos == 'lt':
        abs_width = np.count_nonzero(np.isfinite(img1[-margin, :]))
        abs_height = np.count_nonzero(np.isfinite(img1[:, abs_width - margin]))
        temp0 = img2.shape[0] if corner[1, 0] <= abs_height - 1 else abs_height - corner[0, 0]
        temp1 = img2.shape[1] if corner[1, 1] <= img1.shape[1] - 1 else img1.shape[1] - corner[0, 1]
        mask = np.zeros([temp0, temp1], dtype='bool')
        temp = img1[corner[0, 0]:corner[0, 0] + temp0, corner[0, 1]:corner[0, 1] + temp1]
        temp = np.isfinite(temp)
        wid_ver = np.count_nonzero(temp[:, -1])
        wid_hor = np.count_nonzero(temp[-1, :])
        mask[:wid_ver, :] = True
        mask[:, :wid_hor] = True
        buffer1 = img1[corner[0, 0]:corner[0, 0] + mask.shape[0], corner[0, 1]:corner[0, 1] + mask.shape[1]]
        buffer2 = img2[:mask.shape[0], :mask.shape[1]]
        buffer1[1 - mask] = np.nan
        buffer2[1 - mask] = np.nan
    # for new image with overlap at top only
    elif rel_pos == 't':
        abs_height = np.count_nonzero(np.isfinite(img1[:, margin]))
        wid_ver = abs_height - corner[0, 0]
        wid_hor = img2.shape[1] if img1.shape[1] > img2.shape[1] else img2.shape[1] - corner[0, 1]
        buffer1 = img1[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + wid_hor]
        buffer2 = img2[:wid_ver, :wid_hor]
        buffer1[np.isnan(buffer1)] = 0
        buffer2[np.isnan(buffer2)] = 0
    # for new image with overlap at left only
    else:
        abs_width = np.count_nonzero(np.isfinite(img1[margin, :]))
        wid_ver = img2.shape[0] - corner[0, 0]
        wid_hor = abs_width - corner[0, 1]
        buffer1 = img1[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + wid_hor]
        buffer2 = img2[:wid_ver, :wid_hor]
        buffer1[np.isnan(buffer1)] = 0
        buffer2[np.isnan(buffer2)] = 0

    # find seam


    if rel_pos == 'lt':

        temp_l1 = buffer1[:, :wid_hor]
        temp_l2 = buffer2[:, :wid_hor]
        temp_r1 = buffer1[:wid_ver, :]
        temp_r2 = buffer2[:wid_ver, :]
        if corner[1, 0] <= abs_height - 1:
            dir_l = 'br2tl'
            dv_l = _get_cef_br2tl(temp_l1, temp_l2)
            begin_l = np.array([buffer1.shape[0] - 1, wid_hor - 1])
        else:
            dir_l = 'bl2tr'
            dv_l = _get_cef_bl2tr(temp_l1, temp_l2)
            begin_l = np.array([buffer1.shape[0] - 1, 0])
        if corner[1, 1] <= img1.shape[1] - 1:
            dir_r = 'br2tl'
            dv_r = _get_cef_br2tl(temp_r1, temp_r2)
            begin_r = np.array([wid_ver - 1, buffer1.shape[1] - 1])
        else:
            dir_r = 'bl2tr'
            dv_r = _get_cef_bl2tr(temp_r1, temp_r2)
            begin_r = np.array([0, buffer1.shape[1] - 1])
        dv_l_full = np.zeros(buffer1.shape)
        dv_r_full = np.zeros(buffer1.shape)
        dv_l_full[...] = np.inf
        dv_r_full[...] = np.inf
        dv_l_full[:, :wid_hor] = dv_l
        dv_r_full[:wid_ver, :] = dv_r
        dv = dv_r_full + dv_l_full
        inflect = np.array([wid_ver - 1, wid_hor - 1])
        xx, yy = np.meshgrid(range(dv.shape[1]), range(dv.shape[0]))
        diag_judge = np.abs(inflect[0] * xx - inflect[1] * yy) / np.sqrt(inflect[0] ** 2 + inflect[1] ** 2) < np.sqrt(
            2) / 2
        diag_line = dv * diag_judge
        s = np.unravel_index(np.argmin(diag_line), dv.shape)
        seam = np.zeros(buffer1.shape, dtype='bool')
        seam[s[0], s[1]] = True
        seam = _trace_seam(dv_l_full, seam, s, begin_l, mode=dir_l)
        seam = _trace_seam(dv_r_full, seam, s, begin_r, mode=dir_r)
        mask1 = np.copy(seam)
        for i in mask1.shape[0]:
            ind = np.nonzero(mask1[i, :])[0][0]
            mask1[i, :ind] = True
        mask2 = 1 - mask1
        if np.count_nonzero(mask2.shape == img2.shape) < 2:
            temp = np.ones(img2.shape, dtype='bool')
            temp[:mask2.shape[0], :mask2.shape[1]] = mask2
            mask2 = temp
        full_mask2 = np.zeros(newimg.shape, dtype='bool')
        full_mask2[corner[0, 0]:corner[0, 0] + img2.shape[0], corner[0, 1]:corner[0, 1] + img2.shape[1]] = mask2
        newimg[full_mask2] = img2[mask2]


    elif rel_pos == 't':
        dv = _get_cef_bl2tr(buffer1, buffer2)
        begin = np.array([dv.shape[0] - 1, 0])
        s = np.array([0, dv.shape[1] - 1])
        seam = np.zeros(buffer1.shape, dtype='bool')
        seam[s[0], s[1]] = True
        seam = _trace_seam(dv, seam, s, begin, mode='bl2tr')
        mask1 = np.copy(seam)
        for i in range(mask1.shape[0]):
            ind = np.nonzero(mask1[i, :])[0][0]
            mask1[i, :ind] = True
        mask2 = 1 - mask1
        temp = np.ones(img2.shape, dtype='bool')
        temp[:mask2.shape[0], :mask2.shape[1]] = mask2
        mask2 = temp
        full_mask2 = np.zeros(newimg.shape, dtype='bool')
        full_mask2[corner[0, 0]:corner[0, 0] + img2.shape[0], corner[0, 1]:corner[0, 1] + img2.shape[1]] = mask2
        newimg[full_mask2] = img2[mask2]

    else:
        dv = _get_cef_bl2tr(buffer1, buffer2)
        begin = np.array([dv.shape[0] - 1, 0])
        s = np.array([0, dv.shape[1] - 1])
        seam = np.zeros(buffer1.shape, dtype='bool')
        seam[s[0], s[1]] = True
        seam = _trace_seam(dv, seam, s, begin, mode='bl2tr')
        mask1 = np.copy(seam)
        for i in range(mask1.shape[0]):
            ind = np.nonzero(mask1[i, :])[0][0]
            mask1[i, :ind] = True
        mask2 = 1 - mask1
        temp = np.ones(img2.shape, dtype='bool')
        temp[:mask2.shape[0], :mask2.shape[1]] = mask2
        mask2 = temp
        full_mask2 = np.zeros(newimg.shape, dtype='bool')
        full_mask2[corner[0, 0]:corner[0, 0] + img2.shape[0], corner[0, 1]:corner[0, 1] + img2.shape[1]] = mask2
        newimg[full_mask2] = img2[mask2]
    seam_y, seam_x = np.nonzero(seam)
    seam_coords = np.dstack([seam_y, seam_x])[0] + corner[0, :]
    p_seam = buffer1[seam] - buffer2[seam]
    p_seam = p_seam[p_seam != 0]
    img2_y, img2_x = np.nonzero(full_mask2)
    img2_coords = np.dstack([img2_y, img2_x])[0]
    img2_coords_chunk = []
    st = 0
    while st < img2_coords.shape[0]:
        end = st + chunk_size if st + chunk_size <= img2_coords.shape[0] else img2_coords.shape[0]
        img2_coords_chunk.append(img2_coords[st:end, :])
        st = end
    i_new = np.array([])
    count = 0

    ################## THIS LOOP NEEDS OPTIMIZATION ##################
    for img2_coords_sub in img2_coords_chunk:
        t0 = time.time()
        w_denom = np.sum(1 / _norm(np.swapaxes(seam_coords[:, np.newaxis] - img2_coords_sub, 0, 1)), axis=1)
        w = 1 / _norm(np.swapaxes(seam_coords[:, np.newaxis] - img2_coords_sub, 0, 1))
        w /= w_denom.reshape([w_denom.shape[0], 1])
        p_sub = np.sum(w * p_seam, axis=1)
        i_new = np.append(i_new, p_sub)
    newimg[full_mask2] = newimg[full_mask2] + i_new
    # for q in img2_coords:
    #     w_denom = np.sum(1/norm(seam_coords-q, axis=1))
    #     w = 1 / norm(seam_coords-q, axis=1) / w_denom
    #     p_q = np.sum(w*p_seam)
    #     newimg[q[0], q[1]] += p_q
    print('    Blend: Done with this tile in', str(time.time() - t00), 'sec.')
    gc.collect()
    return newimg


def _get_cef_bl2tr(buffer1, buffer2):
    dv = (buffer1 - buffer2) ** 2
    cor_down = dv.shape[0] - 1
    cor_right = dv.shape[1] - 1
    for x in range(1, cor_right + 1):
        dv[cor_down, x] += dv[cor_down, x - 1]
    for y in range(cor_down - 1, -1, -1):
        for x in range(cor_right + 1):
            if x == 0:
                dv[y, x] += dv[y + 1, x] + dv[y + 1, x + 1]
            elif x == cor_right:
                dv[y, x] += dv[y, x - 1] + dv[y + 1, x - 1] + dv[y + 1, x]
            else:
                dv[y, x] += dv[y, x - 1] + dv[y + 1, x - 1] + dv[y + 1, x] + dv[y + 1, x + 1]
    return dv


def _get_cef_br2tl(buffer1, buffer2):
    dv = (buffer1 - buffer2) ** 2
    cor_down = dv.shape[0] - 1
    cor_right = dv.shape[1] - 1
    for x in range(cor_right - 1, -1, -1):
        dv[cor_down, x] += dv[cor_down, x + 1]
    for y in range(cor_down - 1, -1, -1):
        for x in range(cor_right, -1, -1):
            if x == cor_right:
                dv[y, x] += dv[y + 1, x - 1] + dv[y + 1, x]
            elif x == 0:
                dv[y, x] += dv[y, x + 1] + dv[y + 1, x + 1] + dv[y + 1, x]
            else:
                dv[y, x] += dv[y, x + 1] + dv[y + 1, x - 1] + dv[y + 1, x] + dv[y + 1, x + 1]
    return dv


def _trace_seam(dv, seam, s, begin, mode='br2tl'):
    y, x = s
    if mode == 'br2tl':
        while np.count_nonzero((y, x) == begin) < 2:
            move_ls = [(y, x + 1), (y + 1, x - 1), (y + 1, x), (y + 1, x + 1)]
            if x == 0:
                del move_ls[1]
            if y == dv.shape[0] - 1:
                del move_ls[2]
                del move_ls[3]
                try:
                    del move_ls[1]
                except:
                    pass
            if x == dv.shape[1] - 1:
                del move_ls[0]
                try:
                    del move_ls[3]
                except:
                    pass
            amin_ls = []
            for iy, ix in move_ls:
                amin_ls.append(dv[iy, ix])
            amin = np.argmin(amin_ls)
            y, x = move_ls[amin]
            seam[y, x] = True
    else:
        while np.count_nonzero((y, x) == begin) < 2:
            # print(y, x)
            move_dict = {0: (y, x - 1),
                         1: (y + 1, x - 1),
                         2: (y + 1, x),
                         3: (y + 1, x + 1)}
            if x == dv.shape[1] - 1:
                del move_dict[3]
            if y == dv.shape[0] - 1:
                del move_dict[1]
                del move_dict[2]
                try:
                    del move_dict[3]
                except:
                    pass
            if x == 0:
                del move_dict[0]
                try:
                    del move_dict[1]
                except:
                    pass
            amin_ls = []
            for iy, ix in move_dict.values():
                amin_ls.append(dv[iy, ix])
            amin = np.argmin(amin_ls)
            temp = move_dict.values()
            y, x = temp[amin]
            seam[y, x] = True
    return seam


def _norm(arr):
    res = np.sqrt(arr[:, :, 0] ** 2 + arr[:, :, 1] ** 2)
    return res


def correct_luminance(img1, img2, shift, margin=100, threshold=0.5, max_intercept=1):

    _, _, _, buffer1, buffer2, _, _ = find_overlap(img1, img2, shift, margin=margin)

    mean1 = buffer1[np.isfinite(buffer1)].mean()
    if mean1 < threshold:
        return img2
    mean2 = buffer2[np.isfinite(buffer2)].mean()
    fin1 = np.isfinite(buffer1)
    fin2 = np.isfinite(buffer2)

    # remove singularities
    buffer1[(buffer1 > 10 * mean1) * fin1] = mean1
    buffer2[(buffer2 > 10 * mean2) * fin2] = mean2
    judge = fin1 * fin2 * (buffer1/buffer2 < 1.5) * (buffer2/buffer1 < 1.5)

    # if the number of above average pixels is too small, do nothing and return
    if np.count_nonzero(judge) < 0.3 * buffer1.size:
        return img2

    # build color correction dataset
    orig = buffer2[judge].flatten()
    targ = buffer1[judge].flatten()

    # least square fit
    A = np.vstack([orig, np.ones(len(orig))]).T
    a, b = np.linalg.lstsq(A, targ)[0]

    # reject abnormal mismatch
    if a < 0.5:
        a = 0.5
    elif a > 2:
        a = 2
    if b > max_intercept:
        b = max_intercept
    elif b < -max_intercept:
        b = -max_intercept
    return a * img2 + b


def find_overlap(img1, img2, shift, margin=50):

    rough_shift = morph.get_roughshift(shift)
    corner = _get_corner(rough_shift, img2.shape)
    if abs(rough_shift[1]) > margin and abs(rough_shift[0]) > margin:
        abs_width = np.count_nonzero(np.isfinite(img1[-margin, :]))
        abs_height = np.count_nonzero(np.isfinite(img1[:, abs_width - margin]))
        temp0 = img2.shape[0] if corner[1, 0] <= abs_height - 1 else abs_height - corner[0, 0]
        temp1 = img2.shape[1] if corner[1, 1] <= img1.shape[1] - 1 else img1.shape[1] - corner[0, 1]
        mask = np.zeros([temp0, temp1], dtype='bool')
        temp = img1[corner[0, 0]:corner[0, 0] + temp0, corner[0, 1]:corner[0, 1] + temp1]
        temp = np.isfinite(temp)
        wid_ver = np.count_nonzero(temp[:, -1])
        wid_hor = np.count_nonzero(temp[-1, :])
        mask[:wid_ver, :] = True
        mask[:, :wid_hor] = True
        buffer1 = img1[corner[0, 0]:corner[0, 0] + mask.shape[0], corner[0, 1]:corner[0, 1] + mask.shape[1]]
        buffer2 = img2[:mask.shape[0], :mask.shape[1]]
        buffer1[1 - mask] = np.nan
        buffer2[1 - mask] = np.nan
        case = 'tl'
        if abs_width < corner[0, 1]:
            case = 'skip'
    # for new image with overlap at top only
    elif abs(rough_shift[1]) < margin and abs(rough_shift[0]) > margin:
        abs_height = np.count_nonzero(np.isfinite(img1[:, margin]))
        wid_ver = abs_height - corner[0, 0]
        wid_hor = img2.shape[1] if img1.shape[1] > img2.shape[1] else img2.shape[1] - corner[0, 1]
        buffer1 = img1[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + wid_hor]
        buffer2 = img2[:wid_ver, :wid_hor]
        case = 't'
    # for new image with overlap at left only
    else:
        abs_width = np.count_nonzero(np.isfinite(img1[margin, :]))
        wid_ver = img2.shape[0] - corner[0, 0]
        wid_hor = abs_width - corner[0, 1]
        buffer1 = img1[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + wid_hor]
        buffer2 = img2[:wid_ver, :wid_hor]
        case = 'l'
        if abs_width < corner[0, 1]:
            case = 'skip'
    res1 = np.copy(buffer1)
    res2 = np.copy(buffer2)
    return case, rough_shift, corner, res1, res2, wid_hor, wid_ver
