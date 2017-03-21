# !/usr/bin/env python
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
from tomosaic import blend
import tomosaic
import logging
import glob, time, itertools, os
from tomosaic.register.morph import *
from tomosaic.util.misc import allocate_mpi_subsets, read_aps_32id_adaptive
from scipy.ndimage import gaussian_filter
import numpy as np
import tomopy
import dxchange
import h5py
from itertools import izip
try:
    from mpi4py import MPI
except:
    from tomosaic.util.pseudo import pseudo_comm


logger = logging.getLogger(__name__)

__author__ = "Rafael Vescovi, Ming Du"
__credits__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['recon_hdf5',
           'recon_block',
           'recon_slice',
           'prepare_slice',
           'load_sino',
           'register_recon']

try:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
except:
    comm = pseudo_comm()
    rank = 0
    size = 1


def recon_hdf5(src_fanme, dest_folder, sino_range, sino_step, shift_grid, center_vec=None, center_eq=None, dtype='float32',
               algorithm='gridrec', tolerance=1, chunk_size=20, save_sino=False, sino_blur=None, flattened_radius=120,
               mode='180', **kwargs):
    """
    center_eq: a and b parameters in fitted center position equation center = a*slice + b.
    """

    if not os.path.exists(dest_folder):
        try:
            os.mkdir(dest_folder)
        except:
            pass
    sino_ini = int(sino_range[0])
    sino_end = int(sino_range[1])
    sino_ls_all = np.arange(sino_ini, sino_end, sino_step, dtype='int')
    alloc_set = allocate_mpi_subsets(sino_ls_all.size, size, task_list=sino_ls_all)
    sino_ls = alloc_set[rank]

    # prepare metadata
    f = h5py.File(src_fanme)
    dset = f['exchange/data']
    full_shape = dset.shape
    theta = tomopy.angles(full_shape[0])
    if center_eq is not None:
        a, b = center_eq
        center_ls = sino_ls * a + b
        center_ls = np.round(center_ls)
        for iblock in range(int(sino_ls.size/chunk_size)+1):
            print('Beginning block {:d}.'.format(iblock))
            t0 = time.time()
            istart = iblock*chunk_size
            iend = np.min([(iblock+1)*chunk_size, sino_ls.size])
            fstart = sino_ls[istart]
            fend = sino_ls[iend]
            center = center_ls[istart:iend]
            data = dset[:, fstart:fend:sino_step, :]
            data[np.isnan(data)] = 0
            data = data.astype('float32')
            data = tomopy.remove_stripe_ti(data, alpha=4)
            if sino_blur is not None:
                for i in range(data.shape[1]):
                    data[:, i, :] = gaussian_filter(data[:, i, :], sino_blur)
            rec = tomopy.recon(data, theta, center=center, algorithm=algorithm, **kwargs)
            rec = tomopy.remove_ring(rec)
            rec = tomopy.remove_outlier(rec, tolerance)
            rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)
            for i in range(rec.shape[0]):
                slice = fstart + i*sino_step
                dxchange.write_tiff(rec[i, :, :], fname=os.path.join(dest_folder, 'recon/recon_{:05d}_{:05d}.tiff').format(slice, sino_ini))
                if save_sino:
                    dxchange.write_tiff(data[:, i, :], fname=os.path.join(dest_folder, 'sino/recon_{:05d}_{:d}.tiff').format(slice, int(center[i])))
            iblock += 1
            print('Block {:d} finished in {:.2f} s.'.format(iblock, time.time()-t0))
    else:
        # divide chunks
        grid_bins = np.append(np.ceil(shift_grid[:, 0, 0]), full_shape[1])
        chunks = []
        center_ls = []
        istart = 0
        counter = 0
        # irow should be 0 for slice 0
        irow = np.searchsorted(grid_bins, sino_ls[0], side='right')-1

        for i in range(sino_ls.size):
            counter += 1
            sino_next = i+1 if i != sino_ls.size-1 else i
            if counter >= chunk_size or sino_ls[sino_next] >= grid_bins[irow+1] or sino_next == i:
                iend = i+1
                chunks.append((istart, iend))
                istart = iend
                center_ls.append(center_vec[irow])
                if sino_ls[sino_next] >= grid_bins[irow+1]:
                    irow += 1
                counter = 0

        # reconstruct chunks
        # theta = tomopy.angles(4270, ang1=0, ang2=170.8)
        iblock = 1
        for (istart, iend), center in izip(chunks, center_ls):
            print('Beginning block {:d}.'.format(iblock))
            t0 = time.time()
            fstart = sino_ls[istart]
            fend = sino_ls[iend-1]
            print('Reading data...')
            data = dset[:, fstart:fend+1:sino_step, :]
            if mode == '360':
                overlap =  2 * (dset.shape[2] - center)
                data = tomosaic.morph.sino_360_to_180(data, overlap=overlap, rotation='right')
            data[np.isnan(data)] = 0
            data = data.astype('float32')
            if sino_blur is not None:
                for i in range(data.shape[1]):
                    data[:, i, :] = gaussian_filter(data[:, i, :], sino_blur)
            data = tomopy.remove_stripe_ti(data, alpha=4)
            rec0 = tomopy.recon(data, theta, center=center, algorithm=algorithm, **kwargs)
            rec = tomopy.remove_ring(np.copy(rec0))
            cent = int((rec.shape[1]-1) / 2)
            xx, yy = np.meshgrid(np.arange(rec.shape[2]), np.arange(rec.shape[1]))
            mask0 = ((xx-cent)**2+(yy-cent)**2 <= flattened_radius**2)
            mask = np.zeros(rec.shape, dtype='bool')
            for i in range(mask.shape[0]):
                mask[i, :, :] = mask0
            rec[mask] = (rec[mask] + rec0[mask])/2
            rec = tomopy.remove_outlier(rec, tolerance)
            rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)

            for i in range(rec.shape[0]):
                slice = fstart + i*sino_step
                dxchange.write_tiff(rec[i, :, :], fname=os.path.join(dest_folder, 'recon/recon_{:05d}.tiff').format(slice), dtype=dtype)
                if save_sino:
                    dxchange.write_tiff(data[:, i, :], fname=os.path.join(dest_folder, 'sino/recon_{:05d}_{:d}.tiff').format(slice, center), dtype=dtype)
            print('Block {:d} finished in {:.2f} s.'.format(iblock, time.time()-t0))
            iblock += 1
    return


def recon_hdf5_mpi(src_fanme, dest_folder, sino_range, sino_step, center_vec, shift_grid, dtype='float32',
               algorithm='gridrec', tolerance=1, save_sino=False, sino_blur=None, **kwargs):
    """
    Reconstruct a single tile, or fused HDF5 created using util/total_fusion. MPI supported.
    """

    raise DeprecationWarning

    if rank == 0:
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
    sino_ini = int(sino_range[0])
    sino_end = int(sino_range[1])
    f = h5py.File(src_fanme)
    dset = f['exchange/data']
    full_shape = dset.shape
    theta = tomopy.angles(full_shape[0])
    center_vec = np.asarray(center_vec)
    sino_ls = np.arange(sino_ini, sino_end, sino_step, dtype='int')
    grid_bins = np.ceil(shift_grid[:, 0, 0])

    t0 = time.time()
    alloc_set = allocate_mpi_subsets(sino_ls.size, size, task_list=sino_ls)
    for slice in alloc_set[rank]:
        print('    Rank {:d}: reconstructing {:d}'.format(rank, slice))
        grid_line = np.digitize(slice, grid_bins)
        grid_line = grid_line - 1
        center = center_vec[grid_line]
        data = dset[:, slice, :]
        if sino_blur is not None:
            data = gaussian_filter(data, sino_blur)
        data = data.reshape([full_shape[0], 1, full_shape[2]])
        data[np.isnan(data)] = 0
        data = data.astype('float32')
        if save_sino:
            dxchange.write_tiff(data[:, slice, :], fname=os.path.join(dest_folder, 'sino/recon_{:05d}_{:d}.tiff').format(slice, center))
        # data = tomopy.remove_stripe_ti(data)
        rec = tomopy.recon(data, theta, center=center, algorithm=algorithm, **kwargs)
        # rec = tomopy.remove_ring(rec)
        rec = tomopy.remove_outlier(rec, tolerance)
        rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)
        dxchange.write_tiff(rec, fname='{:s}/recon/recon_{:05d}_{:d}'.format(dest_folder, slice, center), dtype=dtype)

    print('Rank {:d} finished in {:.2f} s.'.format(rank, time.time()-t0))
    return


def recon_block(grid, shift_grid, src_folder, dest_folder, slice_range, sino_step, center_vec, ds_level=0, blend_method='max',
                blend_options=None, tolerance=1, sinogram_order=False, algorithm='gridrec', init_recon=None, ncore=None, nchunk=None, dtype='float32',
                crop=None, save_sino=False, assert_width=None, sino_blur=None, color_correction=False, flattened_radius=120, normalize=True,
                test_mode=False, **kwargs):
    """
    Reconstruct dsicrete HDF5 tiles, blending sinograms only.
    """

    raw_folder = os.getcwd()
    os.chdir(src_folder)
    sino_ini = int(slice_range[0])
    sino_end = int(slice_range[1])
    mod_start_slice = 0
    center_vec = np.asarray(center_vec)
    center_pos_cache = 0
    sino_ls = np.arange(sino_ini, sino_end, sino_step, dtype='int')
    pix_shift_grid = np.ceil(shift_grid)
    pix_shift_grid[pix_shift_grid < 0] = 0

    alloc_set = allocate_mpi_subsets(sino_ls.size, size, task_list=sino_ls)
    for i_slice in alloc_set[rank]:
        print('############################################')
        print('Reconstructing ' + str(i_slice))
        # judge from which tile to retrieve sinos
        grid_lines = np.zeros(grid.shape[1], dtype=np.int)
        slice_in_tile = np.zeros(grid.shape[1], dtype=np.int)
        for col in range(grid.shape[1]):
            bins = pix_shift_grid[:, col, 0]
            grid_lines[col] = int(np.squeeze(np.digitize(i_slice, bins)) - 1)
            if grid_lines[col] == -1:
                print("WARNING: The specified starting slice number does not allow for full sinogram construction. Trying next slice...")
                mod_start_slice = 1
                break
            else:
                mod_start_slice = 0
            slice_in_tile[col] = i_slice - bins[grid_lines[col]]
        if mod_start_slice == 1:
            continue
        center_pos = int(np.round(center_vec[grid_lines].mean()))
        if center_pos_cache == 0:
            center_pos_cache = center_pos
        center_diff = center_pos - center_pos_cache
        center_pos_0 = center_pos
        row_sino, center_pos = prepare_slice(grid, shift_grid, grid_lines, slice_in_tile, ds_level=ds_level,
                                             method=blend_method, blend_options=blend_options, rot_center=center_pos,
                                             assert_width=assert_width, sino_blur=sino_blur, color_correction=color_correction,
                                             normalize=normalize)
        rec0 = recon_slice(row_sino, center_pos, sinogram_order=sinogram_order, algorithm=algorithm,
                          init_recon=init_recon, ncore=ncore, nchunk=nchunk, **kwargs)
        rec = tomopy.remove_ring(np.copy(rec0))
        cent = int((rec.shape[1] - 1) / 2)
        xx, yy = np.meshgrid(np.arange(rec.shape[2]), np.arange(rec.shape[1]))
        mask0 = ((xx - cent) ** 2 + (yy - cent) ** 2 <= flattened_radius ** 2)
        mask = np.zeros(rec.shape, dtype='bool')
        for i in range(mask.shape[0]):
            mask[i, :, :] = mask0
        rec[mask] = (rec[mask] + rec0[mask]) / 2
        rec = tomopy.remove_outlier(rec, tolerance)
        rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)

        print('Center:            {:d}'.format(center_pos))
        rec = np.squeeze(rec)
        if center_diff != 0:
            rec = np.roll(rec, -center_diff, axis=0)
        if not crop is None:
            crop = np.asarray(crop)
            rec = rec[crop[0, 0]:crop[1, 0], crop[0, 1]:crop[1, 1]]

        os.chdir(raw_folder)
        if test_mode:
            dxchange.write_tiff(rec, fname=os.path.join(dest_folder, 'recon/recon_{:05d}_{:04d}.tiff'.format(i_slice, center_pos)), dtype=dtype)
        else:
            dxchange.write_tiff(rec, fname=os.path.join(dest_folder, 'recon/recon_{:05d}.tiff'.format(i_slice)), dtype=dtype)
        if save_sino:
            dxchange.write_tiff(np.squeeze(row_sino), fname=os.path.join(dest_folder, 'sino/sino_{:05d}.tiff'.format(i_slice)), overwrite=True)
        os.chdir(src_folder)
    os.chdir(raw_folder)
    return


def to_rgb2(im):
    im = np.squeeze(im)
    ret = np.empty((im.shape[0], im.shape[1], 3), dtype=np.float16)
    ret[:, :, :] = im[:, :, np.newaxis]
    return ret


def prepare_slice(grid, shift_grid, grid_lines, slice_in_tile, ds_level=0, method='max', blend_options=None, pad=None,
                  rot_center=None, assert_width=None, sino_blur=None, color_correction=False, normalize=True):
    sinos = [None] * grid.shape[1]
    for col in range(grid.shape[1]):
        try:
            sinos[col] = load_sino(grid[grid_lines[col], col], slice_in_tile[col], normalize=normalize)
        except:
            pass
    t = time.time()
    row_sino = register_recon(grid, grid_lines, shift_grid, sinos, method=method, blend_options=blend_options,
                              color_correction=color_correction, assert_width=assert_width)
    if not pad is None:
        row_sino, rot_center = pad_sino(row_sino, pad, rot_center)

    print('stitch:           ' + str(time.time() - t))
    print('final size:       ' + str(row_sino.shape))

    t = time.time()
    row_sino = tomopy.downsample(row_sino, level=ds_level)
    print('downsample:           ' + str(time.time() - t))
    print('new shape :           ' + str(row_sino.shape))

    # t = time.time()
    # row_sino = tomopy.remove_stripe_fw(row_sino, 2)
    # print('strip removal:           ' + str(time.time() - t))
    # Minus Log
    row_sino = tomosaic.util.preprecess(row_sino)
    if sino_blur is not None:
        row_sino[:, 0, :] = gaussian_filter(row_sino[:, 0, :], sino_blur)
    return row_sino, rot_center


def pad_sino(row_sino, pad, rot_center):
    if not pad is None and not pad < row_sino.shape[1]:
        if pad % 2 == 0:
            print("Adding 1 to make odd padding width...")
            pad += 1
        temp = np.zeros([row_sino.shape[0], 1, pad])
        temp[:, :, :] = np.min(row_sino)
        pad_center = (pad - 1) / 2
        if not (rot_center > pad_center or (row_sino.shape[1] - rot_center - 1) > pad_center):
            start = pad_center - rot_center
            temp[:, :, start:start + row_sino.shape[2]] = row_sino
            return (temp, int(pad_center))
        else:
            print("WARNING: The specified padding width cannot accomodate current sinograms.")
            return (row_sino, int(rot_center))
    elif pad < row_sino.shape[2]:
        print("WARNING: Specified padding width is smaller than projection length. ")
        return (row_sino, int(rot_center))


def recon_slice(row_sino, center_pos, sinogram_order=False, algorithm=None,
        init_recon=None, ncore=None, nchunk=None, **kwargs):
    t = time.time()
    ang = tomopy.angles(row_sino.shape[0])
    print(row_sino.shape)
    row_sino = row_sino.astype('float32')
    # row_sino = tomopy.normalize_bg(row_sino) # WARNING: normalize_bg can unpredicatably give bad results for some slices
    row_sino = tomopy.remove_stripe_ti(row_sino, alpha=4)
    rec = tomopy.recon(row_sino, ang, center=center_pos, sinogram_order=sinogram_order, algorithm=algorithm,
        init_recon=init_recon, ncore=ncore, nchunk=nchunk, **kwargs)

    print('recon:           ' + str(time.time() - t))
    return rec


def load_sino(filename, sino_n, normalize=True):
    print('Loading {:s}, slice {:d}'.format(filename, sino_n))
    sino_n = int(sino_n)
    sino, flt, drk = read_aps_32id_adaptive(filename, sino=(sino_n, sino_n + 1))
    if not normalize:
        flt[:, :, :] = flt.max()
        drk[:, :, :] = 0
    sino = tomopy.normalize(sino, flt, drk)
    # 1st slice of each tile of some samples contains mostly abnormally large values which should be removed.
    if sino.max() > 1e2:
        sino[np.abs(sino) > 1] = 1
    return np.squeeze(sino)


def register_recon(grid, grid_lines, shift_grid, sinos, method='max', blend_options=None, color_correction=False, assert_width=None):
    t = time.time()
    file_list = [grid[grid_lines[col], col] for col in range(grid.shape[1])]
    buff = np.zeros([1, 1], dtype='float32')
    for col in range(len(file_list)):
        # try:
        x_shift = shift_grid[grid_lines[col], col, 1]
        temp = np.copy(sinos[col]).astype(np.float32)
        if blend_options is not None:
            opt = blend_options
        else:
            opt = {}
        buff = blend(buff, temp, [0, x_shift], method=method, color_correction=color_correction, **opt)
        # except:
        #     continue
    row_sino = buff.reshape([buff.shape[0], 1, buff.shape[1]])
    i = 0
    if assert_width is None:
        while True:
            try:
                if shift_grid[grid_lines[-1], -1, 1] > 0:
                    assert_width = shift_grid[grid_lines[-1], -1, 1] + sinos[i].shape[-1]
                else:
                    assert_width = row_sino.shape[-1]
                break
            except:
                i += 1
                if i == len(sinos):
                    raise ValueError('No valid data contained in sinos.')
    if assert_width > row_sino.shape[-1]:
        temp = np.zeros([row_sino.shape[0], 1, assert_width])
        temp[:, :, :row_sino.shape[-1]] = row_sino
        row_sino = temp
    else:
        row_sino = row_sino[:, :, :assert_width].astype('float32')
    print('stitch:           ' + str(time.time() - t))
    print('final size:       ' + str(row_sino.shape))
    return row_sino


