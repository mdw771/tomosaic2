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
Module for center search
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import numpy as np
import h5py
import tomopy
import pyfftw
import scipy.ndimage as ndimage
from tomopy import downsample
import tomopy.util.dtype as dtype
from six.moves import zip
import gc
import dxchange
import matplotlib.pyplot as plt
import os
import re
import glob
import time
import warnings
try:
    import xlearn
    from xlearn.classify import model
except:
    warnings.warn('Cannot import package xlearn.')
try:
    from mpi4py import MPI
except:
    from tomosaic.util.pseudo import pseudo_comm
from tomosaic.misc import *
from tomosaic.recon import *
from tomosaic.util import *

logger = logging.getLogger(__name__)

__author__ = ['Rafael Vescovi', 'Ming Du']
__credits__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['find_center_vo',
           'find_center_discrete',
           'find_center_merged',
           'find_center_single',
           'write_center']


try:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
except:
    comm = pseudo_comm()
    rank = 0
    size = 1

PI = 3.1415927

def find_center_vo(tomo, ind=None, smin=-50, smax=50, srad=6, step=0.5,
                   ratio=0.5, drop=20):
    """
    Transplanted from TomoPy with minor fixes.
    Find rotation axis location using Nghia Vo's method. :cite:`Vo:14`.
    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    ind : int, optional
        Index of the slice to be used for reconstruction.
    smin, smax : int, optional
        Coarse search radius. Reference to the horizontal center of the sinogram.
    srad : float, optional
        Fine search radius.
    step : float, optional
        Step of fine searching.
    ratio : float, optional
        The ratio between the FOV of the camera and the size of object.
        It's used to generate the mask.
    drop : int, optional
        Drop lines around vertical center of the mask.
    Returns
    -------
    float
        Rotation axis location.
    """
    tomo = dtype.as_float32(tomo)

    if ind is None:
        ind = tomo.shape[1] // 2
    _tomo = tomo[:, ind, :]

    # Enable cache for FFTW.
    pyfftw.interfaces.cache.enable()

    # Reduce noise by smooth filters. Use different filters for coarse and fine search
    _tomo_cs = ndimage.filters.gaussian_filter(_tomo, (3, 1))
    _tomo_fs = ndimage.filters.median_filter(_tomo, (2, 2))

    # Coarse and fine searches for finding the rotation center.
    if _tomo.shape[0] * _tomo.shape[1] > 4e6:  # If data is large (>2kx2k)
        _tomo_coarse = downsample(np.expand_dims(_tomo_cs,1), level=2)[:, 0, :]
        init_cen = _search_coarse(_tomo_coarse, smin/4, smax/4, ratio, drop)
        fine_cen = _search_fine(_tomo_fs, srad, step, init_cen*4, ratio, drop)
    else:
        init_cen = _search_coarse(_tomo_cs, smin, smax, ratio, drop)
        fine_cen = _search_fine(_tomo_fs, srad, step, init_cen, ratio, drop)

    logger.debug('Rotation center search finished: %i', fine_cen)
    return fine_cen


def _search_coarse(sino, smin, smax, ratio, drop):
    """
    Coarse search for finding the rotation center.
    """
    (Nrow, Ncol) = sino.shape
    print(Nrow, Ncol)
    centerfliplr = (Ncol - 1.0) / 2.0

    # Copy the sinogram and flip left right, the purpose is to
    # make a full [0;2Pi] sinogram
    _copy_sino = np.fliplr(sino[1:])

    # This image is used for compensating the shift of sinogram 2
    temp_img = np.zeros((Nrow - 1, Ncol), dtype='float32')
    temp_img[:] = np.flipud(sino)[1:]

    # Start coarse search in which the shift step is 1
    listshift = np.arange(smin, smax + 1)
    print('listshift', listshift)
    listmetric = np.zeros(len(listshift), dtype='float32')
    mask = _create_mask(2 * Nrow - 1, Ncol, 0.5 * ratio * Ncol, drop)
    for i in listshift:
        _sino = np.roll(_copy_sino, i, axis=1)
        if i >= 0:
            _sino[:, 0:i] = temp_img[:, 0:i]
        else:
            _sino[:, i:] = temp_img[:, i:]
        listmetric[i - smin] = np.sum(np.abs(np.fft.fftshift(
            pyfftw.interfaces.numpy_fft.fft2(
                np.vstack((sino, _sino))))) * mask)
    minpos = np.argmin(listmetric)
    print('coarse return', centerfliplr + listshift[minpos] / 2.0)
    return centerfliplr + listshift[minpos] / 2.0


def _search_fine(sino, srad, step, init_cen, ratio, drop):
    """
    Fine search for finding the rotation center.
    """
    Nrow, Ncol = sino.shape
    centerfliplr = (Ncol + 1.0) / 2.0 - 1.0
    # Use to shift the sinogram 2 to the raw CoR.
    shiftsino = np.int16(2 * (init_cen - centerfliplr))
    _copy_sino = np.roll(np.fliplr(sino[1:]), shiftsino, axis=1)
    if init_cen <= centerfliplr:
        lefttake = np.int16(np.ceil(srad + 1))
        righttake = np.int16(np.floor(2 * init_cen - srad - 1))
    else:
        lefttake = np.int16(np.ceil(
            init_cen - (Ncol - 1 - init_cen) + srad + 1))
        righttake = np.int16(np.floor(Ncol - 1 - srad - 1))
    Ncol1 = righttake - lefttake + 1
    mask = _create_mask(2 * Nrow - 1, Ncol1, 0.5 * ratio * Ncol, drop)
    numshift = np.int16((2 * srad) / step) + 1
    listshift = np.linspace(-srad, srad, num=numshift)
    listmetric = np.zeros(len(listshift), dtype='float32')
    factor1 = np.mean(sino[-1, lefttake:righttake])
    factor2 = np.mean(_copy_sino[0,lefttake:righttake])
    _copy_sino = _copy_sino * factor1 / factor2
    num1 = 0
    for i in listshift:
        _sino = ndimage.interpolation.shift(
            _copy_sino, (0, i), prefilter=False)
        sinojoin = np.vstack((sino, _sino))
        listmetric[num1] = np.sum(np.abs(np.fft.fftshift(
            pyfftw.interfaces.numpy_fft.fft2(
                sinojoin[:, lefttake:righttake + 1]))) * mask)
        num1 = num1 + 1
    minpos = np.argmin(listmetric)
    return init_cen + listshift[minpos] / 2.0


def _create_mask(nrow, ncol, radius, drop):
    du = 1.0 / ncol
    dv = (nrow - 1.0) / (nrow * 2.0 * PI)
    centerrow = np.int16(np.ceil(nrow / 2) - 1)
    centercol = np.int16(np.ceil(ncol / 2) - 1)
    mask = np.zeros((nrow, ncol), dtype='float32')
    for i in range(nrow):
        num1 = np.round(((i - centerrow) * dv / radius) / du)
        (p1, p2) = np.int16(np.clip(np.sort(
            (-int(num1) + centercol, num1 + centercol)), 0, ncol - 1))
        mask[i, p1:p2 + 1] = np.ones(p2 - p1 + 1, dtype='float32')
    if drop < centerrow:
        mask[centerrow - drop:centerrow + drop + 1,
             :] = np.zeros((2 * drop + 1, ncol), dtype='float32')
    mask[:,centercol-1:centercol+2] = np.zeros((nrow, 3), dtype='float32')
    return mask


def find_center_dnn(tomo, theta, search_range, level=0, outpath='center', pad_length=0, **kwargs):

    rot_start, rot_end = search_range[:2]
    if len(search_range) == 3:
        search_step = search_range[-1]
    else:
        search_step = 1
    write_center(tomo[:, 0:1, :], theta, dpath=outpath,
                 cen_range=[rot_start / pow(2, level), rot_end / pow(2, level),
                            search_step / pow(2, level)],
                 pad_length=pad_length)
    return search_in_folder_dnn(outpath, **kwargs)


def search_in_folder_dnn(dest_folder, window=((600, 600), (1300, 1300)), dim_img=128, seed=1337, batch_size=50):

    patch_size = (dim_img, dim_img)
    nb_classes = 2
    save_intermediate = False
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    nb_evl = 100

    fnames = glob.glob(os.path.join(dest_folder, '*.tiff'))
    fnames = np.sort(fnames)

    mdl = model(dim_img, nb_filters, nb_conv, nb_classes)

    mdl.load_weights('weight_center.h5')
    start_time = time.time()
    Y_score = np.zeros((len(fnames)))

    for i in range(len(fnames)):
        print(fnames[i])
        img = dxchange.read_tiff(fnames[i])
        X_evl = np.zeros((nb_evl, dim_img, dim_img))

        for j in range(nb_evl):
            X_evl[j] = xlearn.img_window(img[window[0][0]:window[1][0], window[0][1]:window[1][1]], dim_img, reject_bg=True,
                                         threshold=1.2e-4, reset_random_seed=True, random_seed=j)
        X_evl = xlearn.convolve_stack(X_evl, xlearn.get_gradient_kernel())
        X_evl = xlearn.nor_data(X_evl)
        if save_intermediate:
            dxchange.write_tiff(X_evl, os.path.join('debug', 'x_evl', 'x_evl_{}'.format(i)), dtype='float32',
                                overwrite=True)
        X_evl = X_evl.reshape(X_evl.shape[0], 1, dim_img, dim_img)

        Y_evl = mdl.predict(X_evl, batch_size=batch_size)
        Y_score[i] = sum(np.dot(Y_evl, [0, 1]))
        # print('The evaluate score is:', Y_score[i])
        # Y_score = sum(np.round(Y_score))/len(Y_score)

    ind_max = np.argmax(Y_score)
    best_center = float(os.path.splitext(fnames[ind_max])[0])
    print('Center search done in {} s. Optimal center is {}.'.format(time.time() - start_time, best_center))

    return best_center


def find_center_merged(fname, shift_grid, row_range, search_range, search_step=1, slice=600, method='entropy',
                       output_fname='center_pos.txt', read_theta=True, window=None):

    t00 = time.time()
    log = open(output_fname, 'a')
    center_st, center_end = search_range
    row_st, row_end = row_range
    f = h5py.File(fname)
    row_list = range(row_st, row_end)
    sets = allocate_mpi_subsets(len(row_list), size, task_list=row_list)
    full_shape = read_data_adaptive(fname, shape_only=True)
    if window is None:
        window = ((int(full_shape[2] * 0.3), int(full_shape[2] * 0.3)),
                  (int(full_shape[2] * 0.7), int(full_shape[2] * 0.7)))
    if read_theta:
        _, _, _, theta = read_data_adaptive(fname, proj=(0, 1))
    else:
        theta = tomopy.angles(full_shape[0])
    for row in sets[rank]:
        internal_print('Rank {}: starting row {}.'.format(rank, row))
        t0 = time.time()
        sino = int(slice + shift_grid[row, 0, 0])
        sino = f['exchange/data'][:, sino:sino+1, :]
        if method == 'manual' or 'entropy':
            write_center(sino, theta, os.path.join('center', str(row)),
                         cen_range=(center_st, center_end, search_step))
            if method == 'entropy':
                center = minimum_entropy(os.path.join('center', str(row)), reliability_screening=True)
                if center is None:
                    internal_print('Entropy result did not pass reliability screening. Switching to CNN...')
                    rad = int(sino.shape[-1] * 0.3)
                    rec_mid = int(sino.shape[-1] / 2)
                    center = search_in_folder_dnn(os.path.join('center', str(row)),
                                                  window=((rec_mid-rad, rec_mid-rad), (rec_mid+rad, rec_mid+rad)))
                internal_print('For {} center is {}. ({} s)'.format(row, center, time.time() - t0))
                log.write('{} {}\n'.format(row, center))
        elif method == 'vo':
            mid = sino.shape[2] / 2
            smin = (center_st - mid) * 2
            smax = (center_end - mid) * 2
            center = find_center_vo(sino, smin=smin, smax=smax, step=search_step)
            internal_print('For {} center is {}. ({} s)'.format(row, center, time.time() - t0))
            log.write('{} {}\n'.format(row, center))
        elif method == 'dnn':
            center = find_center_dnn(sino, theta, search_range=(center_st, center_end, search_step),
                                     outpath=os.path.join('center', str(row)), window=window)
            internal_print('For {} center is {}. ({} s)'.format(row, center, time.time() - t0))
            log.write('{} {}\n'.format(row, center))
    log.close()
    internal_print('Total time: {} s.'.format(time.time() - t00))


def find_center_discrete(source_folder, file_grid, shift_grid, row_range, search_range, search_step=1, slice=600,
                         method='entropy', data_format='aps_32id', output_fname='center_pos.txt', read_theta=True):

    t00 = time.time()
    log = open(output_fname, 'a')
    row_st, row_end = row_range
    center_st, center_end = search_range
    row_list = range(row_st, row_end)
    sets = allocate_mpi_subsets(len(row_list), size, task_list=row_list)
    full_shape = read_data_adaptive(os.path.join(source_folder, file_grid[0, 0]), shape_only=True)
    if read_theta:
        _, _, _, theta = read_data_adaptive(os.path.join(source_folder, file_grid[0, 0]), proj=(0, 1))
    else:
        theta = tomopy.angles(full_shape[0])
    for row in sets[rank]:
        internal_print('Row {}'.format(row))
        t0 = time.time()
        slice = int(shift_grid[row, 0, 0] + slice)
        # create sinogram
        try:
            sino = dxchange.read_tiff(os.path.join('center_temp', 'sino', 'sino_{:05d}.tiff'.format(slice)))
            sino = sino.reshape([sino.shape[0], 1, sino.shape[1]])
        except:
            center_vec = [center_st] * file_grid.shape[0]
            center_vec = np.array(center_vec)
            sino, _ = create_row_sinogram(file_grid, shift_grid, source_folder, slice, center_vec, 0, blend_method='pyramid',
                                          data_format=data_format)
            dxchange.write_tiff(sino, os.path.join('center_temp', 'sino', 'sino_{:05d}.tiff'.format(slice)))
        sino = tomopy.remove_stripe_ti(sino, alpha=4)
        if method == 'manual' or 'entropy':
            write_center(sino, theta, dpath='center/{}'.format(row),
                                cen_range=(center_st, center_end, search_step))
            if method == 'entropy':
                center = minimum_entropy(os.path.join('center', str(row)), reliability_screening=True)
                if center is None:
                    internal_print('Entropy result did not pass reliability screening. Switching to CNN...')
                    rad = int(sino.shape[-1] * 0.3)
                    rec_mid = int(sino.shape[-1] / 2)
                    center = search_in_folder_dnn(os.path.join('center', str(row)),
                                                  window=((rec_mid-rad, rec_mid-rad), (rec_mid+rad, rec_mid+rad)))
                internal_print('For {} center is {}. ({} s)'.format(row, center, time.time() - t0))
                log.write('{} {}\n'.format(row, center))
        elif method == 'vo':
            mid = sino.shape[2] / 2
            smin = (center_st - mid) * 2
            smax = (center_end - mid) * 2
            center = find_center_vo(sino, smin=smin, smax=smax, step=search_step)
            internal_print('For {} center is {}. ({} s)'.format(row, center, time.time() - t0))
            log.write('{} {}\n'.format(row, center))
    log.close()
    internal_print('Total time: {} s.'.format(time.time() - t00))


def find_center_single(sino_name, search_range, search_step=1, preprocess_single=False, method='entropy',
                       output_fname='center_pos.txt'):

    log = open(output_fname, 'a')
    center_st, center_end = search_range
    sino = dxchange.read_tiff(sino_name)
    if sino.ndim == 2:
        sino = sino.reshape([sino.shape[0], 1, sino.shape[1]])
    if preprocess_single:
        sino = preprocess(np.copy(sino))
    if method == 'manual':
        write_center(sino, tomopy.angles(sino.shape[0]), dpath='center',
                            cen_range=(center_st, center_end, search_step))
    elif method == 'vo':
        mid = sino.shape[2] / 2
        smin = (center_st - mid) * 2
        smax = (center_end - mid) * 2
        center = find_center_vo(sino, smin=smin, smax=smax, step=search_step)
        internal_print('Center is {}.'.format(center))
        log.write('{}\n'.format(center))
        log.close()


def write_center(tomo, theta, dpath='tmp/center', cen_range=None, pad_length=0, remove_padding=True):

    for center in np.arange(*cen_range):
        rec = tomopy.recon(tomo[:, 0:1, :], theta, algorithm='gridrec', center=center)
        if not pad_length == 0 and remove_padding:
            rec = rec[:, pad_length:-pad_length, pad_length:-pad_length]
        dxchange.write_tiff(np.squeeze(rec), os.path.join(dpath, '{:.2f}'.format(center-pad_length)), overwrite=True)

