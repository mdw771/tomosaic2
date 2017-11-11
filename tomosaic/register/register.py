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
import os
import operator
import glob
import re
import time

import numpy as np
from pyfftw.interfaces.numpy_fft import fft2, ifft2
import tomopy
from scipy import ndimage
import dxchange
import h5py
try:
    from mpi4py import MPI
except:
    from tomosaic.util.pseudo import pseudo_comm

from tomosaic.misc import *
from tomosaic.util import *
from tomosaic.merge import blend
from tomosaic.register.register_translation import register_translation


logger = logging.getLogger(__name__)

__author__ = ['Rafael Vescovi', 'Ming Du']
__credits__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['cross_correlation_bf',
           'cross_correlation_pcm',
           'refine_shift_grid',
           'refine_shift_grid_hybrid',
           'refine_tilt',
           'absolute_shift_grid']

try:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
except:
    comm = pseudo_comm()
    rank = 0
    size = 1


def refine_shift_grid(grid, shift_grid, src_folder='.', dest_folder='.', step=800, upsample=10,
                      y_mask=(-5, 5), x_mask=(-5, 5), motor_readout=None, histogram_equalization=False,
                      data_format='aps_32id'):

    root = os.getcwd()
    os.chdir(src_folder)

    if motor_readout is None:
        motor_readout = [shift_grid[1, 0, 0], shift_grid[0, 1, 1]]

    if (grid.shape[0] != shift_grid.shape[0] or grid.shape[1] != shift_grid.shape[1]):
        return
    pairs = find_pairs(grid)
    n_pairs = pairs.shape[0]

    pairs_shift = np.zeros([n_pairs, 6])

    sets = allocate_mpi_subsets(n_pairs, size)

    for line in sets[rank]:
        if (grid[pairs[line, 0]] == None):
            print ("Block Inexistent")
            continue
        main_pos = pairs[line, 0]
        print('Line {} ({})'.format(line, main_pos))
        main_shape = read_data_adaptive(grid[main_pos], shape_only=True)
        right_pos = pairs[line, 1]
        if (right_pos != None):
            right_shape = read_data_adaptive(grid[right_pos], shape_only=True)
        else:
            right_shape = [0,0,0]
        bottom_pos = pairs[line, 2]
        if (bottom_pos != None):
            bottom_shape = read_data_adaptive(grid[bottom_pos], shape_only=True)
        else:
            bottom_shape = [0,0,0]
        size_max = max(main_shape[0],right_shape[0],bottom_shape[0])
        print('    Reading data...')
        prj, flt, drk = read_data_adaptive(grid[main_pos], proj=(0, size_max, step), data_format=data_format, return_theta=False)
        prj = tomopy.normalize(prj, flt, drk)
        prj[np.abs(prj) < 2e-3] = 2e-3
        prj[prj > 1] = 1
        prj = -np.log(prj)
        prj[np.where(np.isnan(prj) == True)] = 0
        main_prj = vig_image(prj)
        pairs_shift[line, 0:2] = main_pos

        if (right_pos != None):
            prj, flt, drk = read_data_adaptive(grid[right_pos], proj=(0, size_max, step), data_format=data_format, return_theta=False)
            prj = tomopy.normalize(prj, flt, drk)
            prj[np.abs(prj) < 2e-3] = 2e-3
            prj[prj > 1] = 1
            prj = -np.log(prj)
            prj[np.where(np.isnan(prj) == True)] = 0
            right_prj = vig_image(prj)
            shift_ini = shift_grid[right_pos] - shift_grid[main_pos]
            rangeX = shift_ini[1] + x_mask
            rangeY = shift_ini[0] + y_mask
            print('    Calculating shift: {}'.format(right_pos))
            right_vec = create_stitch_shift(main_prj, right_prj, rangeX, rangeY, down=0, upsample=upsample,
                                            histogram_equalization=histogram_equalization)
            # if the computed shift drifts out of the mask, use motor readout instead
            if right_vec[0] <= rangeY[0] or right_vec[0] >= rangeY[1]:
                right_vec[0] = motor_readout[0]
            if right_vec[1] <= rangeX[0] or right_vec[1] >= rangeX[1]:
                right_vec[1] = motor_readout[1]
            pairs_shift[line, 2:4] = right_vec

        if (bottom_pos != None):
            prj, flt, drk = read_data_adaptive(grid[bottom_pos], proj=(0, size_max, step), data_format=data_format, return_theta=False)
            prj = tomopy.normalize(prj, flt, drk)
            prj[np.abs(prj) < 2e-3] = 2e-3
            prj[prj > 1] = 1
            prj = -np.log(prj)
            prj[np.where(np.isnan(prj) == True)] = 0
            bottom_prj = vig_image(prj)
            shift_ini = shift_grid[bottom_pos] - shift_grid[main_pos]
            rangeX = shift_ini[1] + x_mask
            rangeY = shift_ini[0] + y_mask
            print('    Calculating shift: {}'.format(bottom_pos))
            right_vec = create_stitch_shift(main_prj, bottom_prj, rangeX, rangeY, down=1, upsample=upsample,
                                            histogram_equalization=histogram_equalization)
            if right_vec[0] <= rangeY[0] or right_vec[0] >= rangeY[1]:
                right_vec[0] = motor_readout[0]
            if right_vec[1] <= rangeX[0] or right_vec[1] >= rangeX[1]:
                right_vec[1] = motor_readout[1]
            pairs_shift[line, 4:6] = right_vec

    comm.Barrier()
    # combine all shifts
    if rank != 0:
        comm.send(pairs_shift, dest=0)
    else:
        print('Combining grids from other ranks...')
        for src in range(1, size):
            temp = comm.recv(source=src)
            pairs_shift = pairs_shift + temp
    comm.Barrier()
    os.chdir(root)
    if rank == 0:
        try:
            print(pairs_shift)
            np.savetxt(os.path.join(dest_folder, 'shifts.txt'), pairs_shift, fmt=str('%4.2f'))
        except:
            print('Warning: failed to save files. Please save pair shifts as shifts.txt manually:')
            print(pairs_shift)
    return pairs_shift


def create_stitch_shift(block1, block2, rangeX=None, rangeY=None, down=0, upsample=100, histogram_equalization=False):
    """
    Find the relative shift between two tiles. If the inputs are image stacks, the correlation function receives the
    maximum intensity projection along the stacking axis.
    """
    feed1 = block1[...]
    feed2 = block2[...]
    if histogram_equalization:
        feed1 = equalize_histogram(feed1, 0, 1, 1024)
        feed2 = equalize_histogram(feed2, 0, 1, 1024)
    if feed1.ndim == 3:
        assert feed1.shape[0] == feed2.shape[0]
        shift_vec = np.zeros([feed1.shape[0], 2])
        for i in range(feed1.shape[0]):
            shift_vec[i, :] = register_translation(feed1[i], feed2[i], rangeX=rangeX, rangeY=rangeY, down=down, upsample_factor=upsample)
        shift = shift_vec.mean(axis=0)
    else:
        shift = register_translation(feed1, feed2, rangeX=rangeX, rangeY=rangeY, down=down, upsample_factor=upsample)

    return shift


def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]


def refine_shift_grid_hybrid(grid, shift_grid, src_folder, rough_shift, mid_tile=None, center_search_range=None,
                             discard_y_shift=False, data_format='aps_32id', refinement_range=(1, 0), mid_center_array=None):

    try:
        os.mkdir('recon_reslice')
    except:
        pass

    pairs_shift = np.zeros([grid.size, 6])

    # determine the order of tiles in a row to be analyzed
    tile_list = [mid_tile]
    i = 1
    while len(tile_list) < shift_grid.shape[1]:
        if mid_tile - i >= 0:
            tile_list.append(mid_tile - i)
        if len(tile_list) < grid.shape[1] and mid_tile + i < grid.shape[1]:
            tile_list.append(mid_tile + i)
        i += 1
    if mid_tile is None:
        mid_tile = int(grid.shape[1] / 2)
    try:
        center_grid = np.loadtxt('center_grid.txt')
        if center_grid.ndim == 1:
            center_grid = center_grid.reshape(grid.shape)
        print(center_grid)
    except:
        center_grid = np.zeros_like(grid, dtype='float')
        center_grid[...] = None
    if mid_center_array is not None:
        center_grid[:, mid_tile] = mid_center_array
    prj_shape = read_data_adaptive(grid[0, 0], proj=(0, 1), data_format=data_format, shape_only=True)
    prj_mid = int(prj_shape[2] / 2)
    fov = prj_shape[2]
    if center_search_range is None:
        center_search_range = (prj_mid-50, prj_mid+50)
    for irow in range(grid.shape[0]):
        edge_y = int(prj_shape[1] / 2)
        for icol in tile_list:
            side = 0 if icol < mid_tile else 1
            if abs(icol - mid_tile) <= refinement_range[side]:
                print('Tile ({}, {}) refining using reslicing'.format(irow, icol))
                pairs_shift, center_grid = refine_pair_shift_reslice(icol, irow, pairs_shift, mid_tile, grid, center_grid, fov, rough_shift,
                                                                     center_search_range, src_folder, prj_shape=prj_shape, data_format=data_format,
                                                                     chunk_size=50)
            else:
                print('Tile ({}, {}) refining using brute-force'.format(irow, icol))
                if icol < mid_tile:
                    neightbor_y = edge_y + np.round(
                        np.sum(pairs_shift[grid.shape[1] * irow:grid.shape[1] * irow + icol, 2:4], axis=0))
                else:
                    neightbor_y = edge_y - np.round(
                        np.sum(pairs_shift[grid.shape[1] * irow:grid.shape[1] * irow + icol, 2:4], axis=0))
                pairs_shift, center_grid = refine_pair_shift_brute()
                pairs_shift[2, 2:4] = [0, 780]

            # save a sinogram
            if icol == 0:
                raise Exception
            print(pairs_shift)
            if icol == mid_tile:
                dat, flt, drk = read_data_adaptive(grid[irow, icol], sino=(edge_y, edge_y+1), return_theta=False)
                dat = tomopy.normalize(dat, flt, drk)
                dat = preprocess(dat)
                dxchange.write_tiff(np.squeeze(dat), os.path.join('refined_sinograms', 'sino_{}.tiff'.format(irow)), dtype='float32', overwrite=True)
            else:
                base_sino = dxchange.read_tiff(os.path.join('refined_sinograms', 'sino_{}.tiff'.format(irow)))
                if base_sino.ndim == 2:
                    base_sino = base_sino[:, np.newaxis, :]
                if icol < mid_tile:
                    this_shift = np.sum(pairs_shift[grid.shape[1] * irow:grid.shape[1] * irow + icol, 2:4], axis=0)
                    y_shift = int(np.round(this_shift[0]))
                    edge_y += y_shift
                    add_sino, flt, drk = read_data_adaptive(grid[irow, icol], sino=(edge_y, edge_y+1), return_theta=False)
                    add_sino = tomopy.normalize(add_sino, flt, drk)
                    add_sino = preprocess(add_sino)
                    new_sino = blend(np.squeeze(add_sino), np.squeeze(base_sino), shift=[0, this_shift[1]], method='pyramid')
                else:
                    this_shift = np.sum(pairs_shift[grid.shape[1] * irow:grid.shape[1] * irow + icol, 2:4], axis=0)
                    y_shift = int(np.round(this_shift[0]))
                    edge_y -= y_shift
                    add_sino, flt, drk = read_data_adaptive(grid[irow, icol], sino=(edge_y, edge_y+1), return_theta=False)
                    add_sino = tomopy.normalize(add_sino, flt, drk)
                    add_sino = preprocess(add_sino)
                    new_sino = blend(np.squeeze(base_sino), np.squeeze(add_sino), shift=[0, this_shift[1]], method='pyramid')
                dxchange.write_tiff(new_sino, os.path.join('refined_sinograms', 'sino_{}.tiff'.format(irow)),
                                    dtype='float32', overwrite=True)



        pairs_shift[grid.shape[1] * (irow + 1) - 1, :2] = [irow, grid.shape[1]-1]

    np.savetxt('shifts.txt', pairs_shift, fmt=str('%4.2f'))

    raise NotImplementedError


def refine_pair_shift_brute(current_tile, irow, pair_shift, mid_tile, file_grid, center_grid, fov, rough_shift, neighbor_y,
                            src_folder, y_range=(-5, 5, 1), x_range=(-10, 10, 1), tilt_range=(-3, 3, 0.5), prj_shape=None,
                            data_format='aps_32id', chunk_size=50):

    try:
        os.mkdir('refiend_sinograms')
    except:
        pass
    try:
        os.mkdir('bf_recons')
    except:
        pass

    y_est, x_est = rough_shift

    if prj_shape is None:
        prj_shape = read_data_adaptive(file_grid[irow, current_tile], proj=(0, 1), data_format=data_format, shape_only=True)

    # read in stitched sinograms with redined shifts. This at least contains the mid-tile.
    base_sino = dxchange.read_tiff(os.path.join('refined_sinograms', str(irow), str(current_tile)))
    for y_tweak in y_range:
        slice_y = neighbor_y - y_tweak
        add_sino, flt, drk, theta = read_data_adaptive(file_grid[irow, current_tile], sino=(slice_y, slice_y+1))
        add_sino = tomopy.normalize(add_sino, flt, drk)
        add_sino = preprocess(add_sino)
        for x_tweak in x_range:
            if current_tile < mid_tile:
                shift_x = x_est + x_tweak
                new_sino = blend(np.squeeze(add_sino), np.squeeze(base_sino), [0, x_est], method='pyramid')
            else:
                shift_x = x_est + x_tweak
                new_sino = blend(np.squeeze(base_sino), np.squeeze(add_sino), [0, x_est], method='pyramid')


def refine_pair_shift_reslice(current_tile, irow, pair_shift, mid_tile, file_grid, center_grid, fov, rough_shift, center_search_range,
                              src_folder, prj_shape=None, data_format='aps_32id', chunk_size=50):

    y_est, x_est = rough_shift
    fov2 = int(fov / 2)
    fov4 = int(fov2 / 2)

    if prj_shape is None:
        prj_shape = read_data_adaptive(file_grid[0, 0], data_format=data_format, shape_only=True)

    mid_center = center_grid[irow, mid_tile]
    this_center = center_grid[irow, current_tile]

    if current_tile == mid_tile:
        pad_length = 1024
    elif current_tile < mid_tile:
        pad_length = max(1024, mid_center + x_est * abs(mid_tile - current_tile) - fov + 10)
    else:
        pad_length = max(1024, x_est * abs(mid_tile - current_tile) + 10)

    # theta = tomopy.angles(prj_shape[0])
    prj, flt, drk, theta = read_data_adaptive(os.path.join(src_folder, file_grid[irow, current_tile]),
                                              sino=(int(prj_shape[1] / 2), int(prj_shape[1] / 2) + 1),
                                              data_format=data_format)
    prj = tomopy.normalize(prj, flt, drk)
    prj = preprocess(prj)
    prj = tomopy.remove_stripe_ti(prj, alpha=4)
    prj = pad_sinogram(prj, pad_length)
    extra_term = pad_length + (mid_tile - current_tile) * x_est
    adapted_range = map(operator.add, center_search_range, [extra_term, extra_term])

    # find a window in the ring-section of the final reconstruction
    img_mid = int((pad_length * 2 + fov) / 2)
    if current_tile < mid_tile:
        window_ymid = img_mid + mid_center + (x_est + 10) * (mid_tile - current_tile) - fov2
    elif current_tile > mid_tile:
        window_ymid = img_mid - (x_est - mid_center + (x_est - 10) * (mid_tile - current_tile - 1)) - fov2
    else:
        window_ymid = img_mid + (np.mean(center_search_range[:2]) - fov2)

    # find center for this tile if not given
    if np.isnan(this_center):
        print('Center of current tile is not given. Finding center using entropy optimization')
        _write_center(prj, theta, os.path.join('partial_center', str(irow), str(current_tile)),
                     cen_range=adapted_range, pad_length=pad_length, remove_padding=False)
        center_y = img_mid - window_ymid + 1 if current_tile == mid_tile else None
        min_s_fname = minimum_entropy(os.path.join('partial_center', str(irow), str(current_tile)),
                                      window=[[window_ymid - fov4, img_mid - fov4],
                                              [window_ymid + fov4, img_mid + fov4]],
                                      ring_removal=True, center_y=center_y)
        this_center = float(re.findall('\d+\.\d+', min_s_fname)[0])
        center_grid[irow, current_tile] = this_center
        print(str(this_center) + '({})'.format(min_s_fname))
        np.savetxt('center_grid.txt', center_grid, fmt=str('%4.2f'))

    dirname = os.path.join('recon_reslice', str(irow), str(current_tile))
    try:
        os.mkdir(dirname)
    except:
        pass

    # recalculate window boundaries using found center
    img_mid = int((pad_length * 2 + fov) / 2)
    window_ymid = img_mid + this_center - fov2

    # do full reconstruction
    temp = glob.glob(os.path.join(dirname, '*.tiff'))
    if len(temp) == 0:
        print('Full reconstruction ({}, {})'.format(irow, current_tile))
        _recon_single(file_grid[irow, current_tile], this_center, dirname, pad_length=pad_length, ring_removal=True,
                      crop=[[window_ymid-fov2, img_mid-fov2], [window_ymid+fov2, img_mid+fov2]], remove_padding=False,
                      chunk_size=chunk_size)

    # do registration if this tile is not the mid-tile
    print('Getting reslice ({}, {}) (this tile)'.format(irow, current_tile))
    this_section = get_reslice(dirname, slice_x=fov2)
    dxchange.write_tiff(this_section, os.path.join(dirname, 'reslice_{}_{}'.format(irow, current_tile)), dtype='float32', overwrite=True)

    if current_tile != mid_tile:
        if current_tile < mid_tile:
            ref_tile = current_tile + 1
            ref_dirname = os.path.join('recon_reslice', str(irow), str(ref_tile))
        else:
            ref_tile = current_tile - 1
            ref_dirname = os.path.join('recon_reslice', str(irow), str(ref_tile))
        print('Getting reslice ({}, {}) (ref tile)'.format(irow, ref_tile))
        ref_section = get_reslice(ref_dirname, slice_x=fov2)

        n_col = file_grid.shape[1]
        print('Registering this tile ({}, {}) and ref tile ({}, {})'.format(irow, current_tile, irow, ref_tile))
        if current_tile < mid_tile:
            shift_vec = create_stitch_shift(this_section,
                                            ref_section,
                                            rangeX=(x_est-20, x_est+20),
                                            rangeY=(-10, 10))
            pair_shift[irow * n_col + current_tile, :] = np.array([irow, current_tile, shift_vec[0], shift_vec[1], y_est, 0])
        else:
            shift_vec = create_stitch_shift(ref_section,
                                            this_section,
                                            rangeX=(x_est-20, x_est+20),
                                            rangeY=(-10, 10))
            pair_shift[irow * n_col + current_tile - 1, :] = np.array([irow, current_tile, shift_vec[0], shift_vec[1], y_est, 0])

        print(shift_vec)

    return pair_shift, center_grid


def refine_tilt(current_tile, irow, mid_tile, file_grid, tilt_range=(-3, 3, 0.5), prj_shape=None, data_format='aps_32id'):

    if prj_shape is None:
        prj_shape = read_data_adaptive(file_grid[0, 0], data_format=data_format, shape_only=True)

    tilt_ls = np.arange(*tilt_range)
    target_slice = int(prj_shape[1] / 2)
    for tilt in tilt_ls:
        print('Getting tilted sinogram for tilt angle {}'.format(tilt))
        sino = get_tilted_sinogram(file_grid[irow, current_tile], target_slice, tilt, preprocess_data=True)
        try:
            _, _, _, theta = read_data_adaptive(file_grid[irow, current_tile], proj=(0, 1), data_format=data_format)
        except:
            theta = tomopy.angles(sino.shape[0])
        rec = tomopy.recon(sino, theta, algorithm='gridrec')
        dxchange.write_tiff(np.squeeze(rec),
                            os.path.join('bf_recons', str(irow), 'tilt', '{:.2f}'.format(tilt)), overwrite=True)





def absolute_shift_grid(pairs_shift, file_grid, mode='vh'):
    """
    Generate absolute shift grid from a relative shift grid. Default building method is first right then down.
    :param pairs_shift:
    :param file_grid:
    :param mode:
    :return:
    """
    shape = file_grid.shape
    abs_shift_grid = np.zeros([shape[0], shape[1], 2], dtype='float32')

    for (y, x), _ in np.ndenumerate(file_grid):
        if y == 0 and x == 0:
            pass
        else:
            if mode == 'hv':
                for x_ind in range(0, x):
                    abs_shift_grid[y, x, :] += pairs_shift[0, x_ind, 0:2]
                for y_ind in range(0, y):
                    abs_shift_grid[y, x, :] += pairs_shift[y_ind, x, 2:4]
            elif mode == 'vh':
                for y_ind in range(0, y):
                    abs_shift_grid[y, x, :] += pairs_shift[y_ind, 0, 2:4]
                for x_ind in range(0, x):
                    abs_shift_grid[y, x, :] += pairs_shift[y, x_ind, 0:2]

    return abs_shift_grid


def _recon_single(fname, center, dest_folder, sino_range=None, chunk_size=50, read_theta=True, pad_length=0,
                  phase_retrieval=False, ring_removal=True, algorithm='gridrec', flattened_radius=40, crop=None,
                  remove_padding=True, **kwargs):

    prj_shape = read_data_adaptive(fname, shape_only=True)
    if read_theta:
        _, _, _, theta = read_data_adaptive(fname, proj=(0, 1), return_theta=True)
    else:
        theta = tomopy.angles(prj_shape[0])
    if sino_range is None:
        sino_st = 0
        sino_end = prj_shape[1]
        sino_step = 1
    else:
        sino_st, sino_end = sino_range[:2]
        if len(sino_range) == 3:
            sino_step = sino_range[-1]
        else:
            sino_step = 1
    chunks = np.arange(0, sino_end, chunk_size * sino_step, dtype='int')
    for chunk_st in chunks:
        t0 = time.time()
        chunk_end = min(chunk_st + chunk_size * sino_step, prj_shape[1])
        data, flt, drk = read_data_adaptive(fname, sino=(chunk_st, chunk_end, sino_step), return_theta=False)
        data = tomopy.normalize(data, flt, drk)
        data = preprocess(data)
        data = tomopy.remove_stripe_ti(data, alpha=4)
        if phase_retrieval:
            data = tomopy.retrieve_phase(data, kwargs['pixel_size'], kwargs['dist'], kwargs['energy'],
                                         kwargs['alpha'])
        if pad_length != 0:
            data = pad_sinogram(data, pad_length)
        if ring_removal:
            data = tomopy.remove_stripe_ti(data, alpha=4)
            rec0 = tomopy.recon(data, theta, center=center+pad_length, algorithm=algorithm, **kwargs)
            rec = tomopy.remove_ring(np.copy(rec0))
            cent = int((rec.shape[1] - 1) / 2)
            xx, yy = np.meshgrid(np.arange(rec.shape[2]), np.arange(rec.shape[1]))
            mask0 = ((xx - cent) ** 2 + (yy - cent) ** 2 <= flattened_radius ** 2)
            mask = np.zeros(rec.shape, dtype='bool')
            for i in range(mask.shape[0]):
                mask[i, :, :] = mask0
            rec[mask] = (rec[mask] + rec0[mask]) / 2
        else:
            rec = tomopy.recon(data, theta, center=center + pad_length, algorithm=algorithm, **kwargs)
        if pad_length != 0 and remove_padding:
            rec = rec[:, pad_length:pad_length + prj_shape[2], pad_length:pad_length + prj_shape[2]]
        rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)
        if crop is not None:
            crop = np.asarray(crop, dtype='int')
            rec = rec[:, crop[0, 0]:crop[1, 0], crop[0, 1]:crop[1, 1]]
        for i in range(rec.shape[0]):
            slice = chunk_st + sino_step * i
            dxchange.write_tiff(rec[i, :, :],
                                fname=os.path.join(dest_folder, 'recon_{:05d}.tiff').format(slice),
                                dtype='float32')
        print('Slice {} - {} finished in {:.2f} s.'.format(chunk_st, chunk_st + rec.shape[0] * sino_step, time.time() - t0))



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


def cross_correlation_pcm(img1, img2, rangeX=None, rangeY=None, blur=3, down=0):
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
    norm_max = np.max([img1.max(), img2.max()])
    src_image = np.array(img1, dtype=np.complex128, copy=False) / norm_max
    target_image = np.array(img2, dtype=np.complex128, copy=False) / norm_max
    f0 = fft2(src_image)
    f1 = fft2(target_image)
    cross_correlation = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    cross_correlation = ndimage.gaussian_filter(cross_correlation, sigma=blur)
    shape = cross_correlation.shape
    mask = np.zeros(cross_correlation.shape)
    if rangeY[0] > 0 and rangeX[0] > 0:
        mask[rangeY[0]:rangeY[1], rangeX[0]:rangeX[1]] = 1
    elif rangeY[0] < 0:
        mask[shape[0] + rangeY[0]:, rangeX[0]:rangeX[1]] = 1
        mask[:rangeY[1], rangeX[0]:rangeX[1]] = 1
    elif rangeX[0] < 0:
        mask[rangeY[0]:rangeY[1], shape[1] + rangeX[0]:] = 1
        mask[rangeY[0]:rangeY[1], :rangeX[1]] = 1
    cross_correlation = cross_correlation * mask
    # Locate maximum
    shifts = np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape)
    if not down:
        if shifts[1] < 0:
            shifts[1] += float(shape[1])
    else:
        if shifts[0] < 0:
            shifts[0] += float(shape[0])
    return shifts


def partial_center_alignment(file_grid, shift_grid, center_vec, src_folder, range_0=-5, range_1=5, data_format='aps_32id'):
    """
    Further refine shift to optimize reconstruction in case of tilted rotation center, using entropy minimization as
    metric.
    """
    n_tiles = file_grid.size
    if size > n_tiles:
        raise ValueError('Number of ranks larger than number of tiles.')
    root_folder = os.getcwd()
    os.chdir(src_folder)
    if rank == 0:
        f = h5py.File(file_grid[0, 0])
        dset = f['exchange/data']
        slice = int(dset.shape[1]/2)
        xcam = dset.shape[2]
        n_angles = dset.shape[0]
        for i in range(1, size):
            comm.send(slice, dest=i)
            comm.send(xcam, dest=i)
            comm.send(n_angles, dest=i)
    else:
        slice = comm.recv(source=0)
        xcam = comm.recv(source=0)
        n_angles = comm.recv(source=0)
    comm.Barrier()
    width = shift_grid[:, -1, 1].max() + xcam
    tile_ls = np.zeros([file_grid.size, 2], dtype='int')
    a = np.unravel_index(range(n_tiles), file_grid.shape)
    tile_ls[:, 0] = a[0]
    tile_ls[:, 1] = a[1]
    theta = tomopy.angles(n_angles)
    shift_corr = np.zeros(shift_grid.shape)
    alloc_set = allocate_mpi_subsets(n_tiles, size)
    for i in alloc_set[rank]:
        y, x = tile_ls[i]
        center = center_vec[y]
        fname = file_grid[y, x]
        sino, flt, drk = read_data_adaptive(fname, sino=(slice, slice + 1), data_format=data_format)
        sino = np.squeeze(tomopy.normalize(sino, flt, drk))
        sino = preprocess(sino)
        s_opt = np.inf
        for delta in range(range_0, range_1+1):
            sino_pad = np.zeros([n_angles, width])
            sino_pad = arrange_image(sino_pad, sino, (0, shift_grid[y, x, 1]+delta))
            sino_pad = sino_pad.reshape(sino_pad.shape[0], 1, sino_pad.shape[1])
            rec = tomopy.recon(sino_pad, theta, center=center)
            rec = np.squeeze(rec)
            s = entropy(rec)
            if s < s_opt:
                s_opt = s
                delta_opt = delta
        shift_corr[y, x, 1] += delta_opt
    comm.Barrier()
    if rank != 0:
        comm.send(shift_corr, dest=0)
    else:
        for i in range(1, size):
            shift_corr = shift_corr + comm.recv(source=i)
    os.chdir(root_folder)
    np.save('shift_corr', shift_corr)

    return


def _write_center(tomo, theta, dpath='tmp/center', cen_range=None, pad_length=0, remove_padding=True):

    for center in np.arange(*cen_range):
        rec = tomopy.recon(tomo[:, 0:1, :], theta, algorithm='gridrec', center=center)
        if not pad_length == 0 and remove_padding:
            rec = rec[:, pad_length:-pad_length, pad_length:-pad_length]
        dxchange.write_tiff(np.squeeze(rec), os.path.join(dpath, '{:.2f}'.format(center-pad_length)), overwrite=True)
