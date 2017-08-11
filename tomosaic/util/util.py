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

logger = logging.getLogger(__name__)

__author__ = "Rafael Vescovi"
__credits__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['get_files',
           'get_index',
           'save_partial_frames',
           'save_partial_raw',
           'build_panorama',
           'total_fusion',
           'file2grid',
           'hdf5_retrieve_phase',
	       'preprocess']

import os, glob, re
import h5py
import numpy as np
import tomopy
import dxchange
from tomosaic.util.phase import retrieve_phase
from tomosaic.misc.misc import allocate_mpi_subsets, read_aps_32id_adaptive
from tomosaic.merge.merge import blend
from tomosaic.register.morph import arrange_image
import shutil
from scipy.ndimage import gaussian_filter
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
from tomopy import downsample
import time
import gc

try:
    from mpi4py import MPI
except:
    from tomosaic.util.pseudo import pseudo_comm


try:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
except:
    comm = pseudo_comm()
    rank = 0
    size = 1


def get_files(folder, prefix, type='.h5'):
    root = os.getcwd()
    os.chdir(folder)
    file_list = glob.glob(prefix + '*' + type)
    os.chdir(root)
    return file_list


def get_index(file_list, pattern=0):
    if pattern == 0:
        regex = re.compile(r".+_x(\d+)_y(\d+).+")
        ind_buff = [m.group(1, 2) for l in file_list for m in [regex.search(l)] if m]
    elif pattern == 1:
        regex = re.compile(r".+_y(\d+)_x(\d+).+")
        ind_buff = [m.group(2, 1) for l in file_list for m in [regex.search(l)] if m]
    return np.asarray(ind_buff).astype('int')


def save_partial_frames(file_grid, save_folder, prefix, frame=0):
    for (y, x), value in np.ndenumerate(file_grid):
        print(value)
        if (value != None):
            prj, flt, drk = read_aps_32id_adaptive(value, proj=(frame, frame + 1))
            prj = tomopy.normalize(prj, flt, drk)
            prj = preprocess(prj)
            fname = prefix + 'Y' + str(y).zfill(2) + '_X' + str(x).zfill(2)
            dxchange.write_tiff(np.squeeze(prj), fname=os.path.join(save_folder, 'partial_frames', fname))


def save_partial_raw(file_list, save_folder):
    for value in file_list:
        if (value != None):
            prj, flt, drk = read_aps_32id_adaptive(value, proj=(0, 1))
            fname = value
            dxchange.write_tiff_stack(np.squeeze(flt), fname=os.path.join(save_folder, 'partial_flats', fname))
            dxchange.write_tiff_stack(np.squeeze(drk), fname=os.path.join(save_folder, 'partial_darks', fname))
            prj = prj.astype('float32')
            dxchange.write_tiff(np.squeeze(prj), fname=os.path.join(save_folder, 'partial_frames_raw', fname))


g_shapes = lambda fname: h5py.File(fname, "r")['exchange/data'].shape


def build_panorama(src_folder, file_grid, shift_grid, frame=0, method='max', method2=None, blend_options={}, blend_options2={},
                   blur=None, color_correction=False, margin=100):

    t00 = time.time()
    root = os.getcwd()
    os.chdir(src_folder)
    cam_size = g_shapes(file_grid[0, 0])
    cam_size = cam_size[1:3]
    img_size = shift_grid[-1, -1] + cam_size
    buff = np.zeros([1, 1])
    last_none = False
    if method2 is None:
        for (y, x), value in np.ndenumerate(file_grid):
            if (value != None and frame < g_shapes(value)[0]):
                prj, flt, drk = read_aps_32id_adaptive(value, proj=(frame, frame + 1))
                prj = tomopy.normalize(prj, flt, drk)
                prj = preprocess(prj, blur=blur)
                t0 = time.time()
                buff = blend(buff, np.squeeze(prj), shift_grid[y, x, :], method=method, color_correction=color_correction, **blend_options)
                print('Rank: {:d}; Frame: {:d}; Pos: ({:d}, {:d}); Method: {:s}; Color Corr.:{:b}; Tile stitched in '
                      '{:.2f} s.'.format(rank, frame, y, x, method, color_correction, time.time()-t0))
                if last_none:
                    buff[margin:, margin:-margin][np.isnan(buff[margin:, margin:-margin])] = 0
                    last_none = False
            else:
                last_none = True
    else:
        for y in range(file_grid.shape[0]):
            temp_grid = file_grid[y:y+1, :]
            temp_shift = np.copy(shift_grid[y:y+1, :, :])
            offset = np.min(temp_shift[:, :, 0])
            temp_shift[:, :, 0] = temp_shift[:, :, 0] - offset
            row_buff = np.zeros([1, 1])
            prj, flt, drk = read_aps_32id_adaptive(temp_grid[0, 0], proj=(frame, frame + 1))
            prj = tomopy.normalize(prj, flt, drk)
            prj = preprocess(prj, blur=blur)
            row_buff, _ = arrange_image(row_buff, np.squeeze(prj), temp_shift[0, 0, :], order=1)
            for x in range(1, temp_grid.shape[1]):
                value = temp_grid[0, x]
                if (value != None and frame < g_shapes(value)[0]):
                    prj, flt, drk = read_aps_32id_adaptive(value, proj=(frame, frame + 1))
                    prj = tomopy.normalize(prj, flt, drk)
                    prj = preprocess(prj, blur=blur)
                    t0 = time.time()
                    row_buff = blend(row_buff, np.squeeze(prj), temp_shift[0, x, :], method=method, color_correction=color_correction, **blend_options)
                    print('Rank: {:d}; Frame: {:d}; Pos: ({:d}, {:d}); Method: {:s}; Color Corr.:{:b}; Tile stitched in '
                          '{:.2f} s.'.format(rank, frame, y, x, method, color_correction, time.time() - t0))
                    if last_none:
                        row_buff[margin:, margin:-margin][np.isnan(row_buff[margin:, margin:-margin])] = 0
                        last_none = False
                else:
                    last_none = True
            t0 = time.time()
            buff = blend(buff, row_buff, [offset, 0], method=method2, color_correction=False, **blend_options2)
            print('Rank: {:d}; Frame: {:d}; Row: {:d}; Row stitched in {:.2f} s.'.format(rank, frame, y, time.time()-t0))
    print('Rank: {:d}; Frame: {:d}; Panorama built in {:.2f} s.'.format(rank, frame, time.time()-t00))
    os.chdir(root)
    return buff


def grid2file(grid, file_name):
    with file(file_name, 'w') as outfile:
        # for data_slice in grid:
        ncol = len(grid[0, 0, :])
        nval = grid.shape[0] * grid.shape[1]
        y_lst = np.zeros(nval)
        x_lst = np.zeros(nval)
        values = np.zeros([ncol, nval])
        ind = 0
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                y_lst[ind] = y
                x_lst[ind] = x
                temp = grid[y, x, :]
                values[:, ind] = temp
                ind += 1
        outarr = [y_lst, x_lst]
        outarr = np.append(outarr, values, axis=0)
        outarr = np.transpose(outarr)
        outarr = np.squeeze(outarr)
        np.savetxt(outfile, outarr, fmt=str('%4.2f'))
    return


def file2grid(file_name):

    with file(file_name, 'r') as infile:
        grid0 = np.loadtxt(file_name)
        grid_shape = [grid0[-1, 0] + 1, grid0[-1, 1] + 1]
        grid_shape = map(int, grid_shape)
        ncol = len(grid0[0, :]) - 2
        grid = np.zeros([grid_shape[0], grid_shape[1], ncol])
        for line in grid0:
            y, x = map(int, (line[0], line[1]))
            grid[y, x, :] = line[2:]
    return grid


def normalize(img):

    img = (img - img.min()) / (img.max() - img.min())
    return img


def reorganize_dir(file_list, raw_ds=(2,4), dtype='float16', **kwargs):
    """
    Reorganize hdf5 files and reorganize directory as:
    ----------------
    /data_raw_1x/1920x1200x4500 x* 12x11
                  /shift_matrix
                 /center_matrix
    /data_raw_2x/960x600x4500     x* 12x11

    /raw_4x/480x300x4500     x* 12x11
    ---------------
    /recon_gridrec_1x

    x* means grid of hdf5 files

    Parameters:
    -----------
    file_list : ndarray
        List of h5 files in the directory.
    convert : int, optional
        Bit of integer the data are to be converted into.
    """

    # downsample
    try:
        f = h5py.File(file_list[0])
        full_shape = f['exchange/data'].shape
    except:
        f = h5py.File(os.path.join('data_raw_1x', file_list[0]))
        full_shape = f['exchange/data'].shape
    comm.Barrier()

    for fname in file_list:
        print('Now processing '+str(fname))
        # make downsampled subdirectories
        for ds in raw_ds:
            # create downsample folder if not existing
            folder_name = 'data_raw_'+str(ds)+'x'
            if rank == 0:
                if not os.path.exists(folder_name):
                    os.mkdir(folder_name)
            comm.Barrier()
            # move file if downsample level is 1
            if ds == 1:
                if rank == 0:
                    if not os.path.isfile(folder_name+'/'+fname):
                        shutil.move(fname, folder_name+'/'+fname)
                comm.Barrier()
            # otherwise perform downsampling
            else:
                if rank == 0:
                    if not os.path.exists(folder_name):
                        os.mkdir(folder_name)
                comm.Barrier()
                try:
                    o = h5py.File('data_raw_1x/' + fname, 'r')
                except:
                    o = h5py.File(fname, 'r')
                raw = o['exchange/data']
                if rank == 0:
                    if os.path.exists(folder_name+'/'+fname):
                        print('Warning: File already exists. Continue anyway? (y/n) ')
                        cont = raw_input()
                        if cont in ['n', 'N']:
                            exit()
                        else:
                            print('Old file will be overwritten. '+folder_name+'/' + fname)
                            os.remove(folder_name+'/' + fname)
                    f = h5py.File(folder_name+'/'+fname)
                comm.Barrier()
                if rank != 0:
                    f = h5py.File(folder_name+'/'+fname)
                dat_grp = f.create_group('exchange')
                dat = dat_grp.create_dataset('data', (full_shape[0], np.floor(full_shape[1]/ds),
                                                          np.floor(full_shape[2]/ds)), dtype=dtype)
                # write downsampled data frame-by-frame
                n_frames = full_shape[0]
                alloc_sets = allocate_mpi_subsets(n_frames, size)
                for frame in alloc_sets[rank]:
                    temp = raw[frame:frame+1, :, :]
                    temp = image_downsample(temp, ds)
                    dat[frame:frame+1, :, :] = temp
                    print('\r    Rank: {:d}, DS: {:d}, at frame {:d}'.format(rank, ds, frame))
                print(' ')
                # downsample flat/dark field data
                comm.Barrier()
                raw = o['exchange/data_white']
                aux_shape = raw.shape
                dat = dat_grp.create_dataset('data_white', (aux_shape[0], np.floor(aux_shape[1]/ds),
                                                                  np.floor(aux_shape[2]/ds)), dtype=dtype)
                comm.Barrier()
                print('    Downsampling whites and darks')
                n_whites = raw.shape[0]
                alloc_sets = allocate_mpi_subsets(n_whites, size)
                for frame in alloc_sets[rank]:
                    temp = raw[frame:frame+1, :, :]
                    temp = image_downsample(temp, ds)
                    dat[frame:frame+1, :, :] = temp
                    print('\r    Rank: {:d}, DS: {:d}, at frame {:d}'.format(rank, ds, frame))

                raw = o['exchange/data_dark']
                aux_shape = raw.shape
                dat = dat_grp.create_dataset('data_dark', (aux_shape[0], np.floor(aux_shape[1]/ds),
                                                           np.floor(aux_shape[2]/ds)), dtype=dtype)
                comm.Barrier()
                n_darks = raw.shape[0]
                alloc_sets = allocate_mpi_subsets(n_darks, size)
                for frame in alloc_sets[rank]:
                    temp = raw[frame:frame+1, :, :]
                    temp = image_downsample(temp, ds)
                    dat[frame:frame+1, :, :] = temp
                    print('\r    Rank: {:d}, DS: {:d}, at frame {:d}'.format(rank, ds, frame))
                comm.Barrier()
                f.close()
                comm.Barrier()
                print('Done file: {:s} DS: {:d}'.format(fname, ds))
        # delete file after all done
        try:
            os.remove(fname)
        except:
            pass


def reorganize_tiffs():
    tiff_list = glob.glob('*.tiff')
    for fname in tiff_list:
        print('Now processing '+str(fname))
        # make downsampled subdirectories
        for ds in [1, 2, 4]:
            # create downsample folder if not existing
            folder_name = 'tiff_'+str(ds)+'x'
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            # copy file if downsample level is 1
            if ds == 1:
                shutil.copyfile(fname, folder_name+'/'+fname)
            # otherwise perform downsampling
            else:
                temp = imread(fname, flatten=True)
                temp = image_downsample(temp, ds)
                imsave(folder_name+'/'+fname, temp, format='tiff')


def global_histogram(dmin, dmax, n_bins, plot=True):
    tiff_list = glob.glob('*.tiff')
    mybins = np.linspace(dmin, dmax, n_bins + 1)
    myhist = np.zeros(n_bins, dtype='int32')
    bin_width = (dmax - dmin) / n_bins
    for fname in tiff_list:
        print('Now analyzing'+fname)
        temp = imread(fname, flatten=True)
        temp = np.ndarray.flatten(temp)
        myhist = myhist + np.histogram(temp, bins=mybins)[0]
    if plot:
        plt.bar(mybins[:-1], myhist, width=bin_width)
        plt.show()
    return myhist, mybins


def img_cast(image, display_min, display_max, dtype='uint16'):
    bit = int(re.findall(r'\d+', dtype)[0])
    divider = 2 ** bit
    image.clip(display_min, display_max, out=image)
    image -= display_min
    image = image / (display_max - display_min + 1) * float(divider)
    return image.astype(dtype)


def image_downsample(img, ds):
    temp = downsample(downsample(img, level=ds-1, axis=1), level=ds-1, axis=2)
    return temp


def check_fname_ext(fname, ext):
    ext_len = len(ext)
    if fname[-ext_len-1:] != '.' + ext:
        fname += ('.' + ext)
    return fname


def hdf5_cast(fname, display_min=None, display_max=None, dtype='uint16'):
    f = h5py.File(fname)
    dset = f['exchange/data']
    n_slices = dset.shape[0]
    if display_min is None:
        if rank == 0:
            display_min = dset.min()
            for i in range(1, size):
                comm.send(display_min, dest=i)
        else:
            display_min = comm.recv(source=0)
    comm.Barrier()
    if display_max is None:
        if rank == 0:
            display_max = dset.max()
            for i in range(1, size):
                comm.send(display_max, dest=i)
        else:
            display_max = comm.recv(source=0)
    comm.Barrier()
    alloc_set = allocate_mpi_subsets(n_slices, size)
    for i in alloc_set[rank]:
        temp = dset[i, :, :]
        temp = img_cast(temp, display_min, display_max, dtype=dtype)
        dset[i, :, :] = temp
        print('    Rank: {:d}, slice: {:d}'.format(rank, i))
    dset = dset.astype(dtype)
    try:
        dset = f['exchange/data_white']
        alloc_set = allocate_mpi_subsets(dset.shape[0], size)
        for i in alloc_set[rank]:
            temp = dset[i, :, :]
            temp = img_cast(temp, display_min, display_max, dtype=dtype)
            dset[i, :, :] = temp
        dset = dset.astype(dtype)
    except:
        pass
    try:
        dset = f['exchange/data_dark']
        alloc_set = allocate_mpi_subsets(dset.shape[0], size)
        for i in alloc_set[rank]:
            temp = dset[i, :, :]
            temp = img_cast(temp, display_min, display_max, dtype=dtype)
            dset[i, :, :] = temp
        dset = dset.astype(dtype)
    except:
        pass
    return


def tiff2hdf5(src_folder, dest_folder, dest_fname, pattern='recon_*.tiff', display_min=None, display_max=None,
              dtype='int8'):

    dest_fname = check_fname_ext(dest_fname, 'h5')
    filelist = glob.glob(os.path.join(src_folder, pattern))
    filelist = sorted(filelist)
    n_files = len(filelist)
    temp = imread(filelist[0])
    full_shape = np.array([n_files, temp.shape[0], temp.shape[1]])
    if rank == 0:
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        f = h5py.File(os.path.join(dest_folder, dest_fname))
    comm.Barrier()
    if rank != 0:
        f = h5py.File(os.path.join(dest_folder, dest_fname))
    grp = f.create_group('exchange')
    grp.create_dataset('data', full_shape, dtype='float32')
    alloc_set = allocate_mpi_subsets(n_files, size)
    dset = f['exchange/data']
    for i in alloc_set[rank]:
        img = imread(filelist[i])
        dset[i, :, :] = img
        print('    Rank: {:d}, file: {:d}'.format(rank, i))
    comm.Barrier()
    hdf5_cast(os.path.join(dest_folder, dest_fname), display_min=display_min, display_max=display_max, dtype=dtype)
    return


def make_empty_hdf5(folder, fname, full_shape, dtype):
    fname = check_fname_ext(fname, 'h5')
    f = h5py.File('{:s}/{:s}'.format(folder, fname))
    grp = f.create_group('exchange')
    grp.create_dataset('data', full_shape, dtype=dtype)
    dset_flat = grp.create_dataset('data_white', (1, full_shape[1], full_shape[2]), dtype=dtype)
    dset_dark = grp.create_dataset('data_dark', (1, full_shape[1], full_shape[2]), dtype=dtype)
    dset_flat[:, :, :] = np.ones(dset_flat.shape, dtype=dtype)
    dset_dark[:, :, :] = np.zeros(dset_dark.shape, dtype=dtype)
    return f


def hdf5_retrieve_phase(src_folder, src_fname, dest_folder, dest_fname, method='paganin', corr_flat=False,
                        dtype='float16', sino_range=None, **kwargs):

    src_fname = check_fname_ext(src_fname, 'h5')
    dest_fname = check_fname_ext(dest_fname, 'h5')
    o = h5py.File('{:s}/{:s}'.format(src_folder, src_fname))
    dset_src = o['exchange/data']
    n_frames = dset_src.shape[0]

    if rank == 0:
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        if os.path.exists(dest_folder + '/' + dest_fname):
            print('Warning: File already exists. Continue anyway? (y/n) ')
            cont = raw_input()
            if cont in ['n', 'N']:
                exit()
            else:
                print('Old file will be overwritten.')
                os.remove(dest_folder+'/' + dest_fname)
        #f = make_empty_hdf5(dest_folder, dest_fname, dset_src.shape, dtype=dtype)
        f = h5py.File(dest_folder+'/'+dest_fname)
    comm.Barrier()
    if rank != 0:
        f = h5py.File(dest_folder+'/'+dest_fname)
    full_shape = dset_src.shape
    grp = f.create_group('exchange')
    if sino_range is None:
        dset_dest = grp.create_dataset('data', full_shape, dtype=dtype)
        dset_flat = grp.create_dataset('data_white', (1, full_shape[1], full_shape[2]), dtype=dtype)
        dset_dark = grp.create_dataset('data_dark', (1, full_shape[1], full_shape[2]), dtype=dtype)
    else:
        sino_start = sino_range[0]
        sino_end = sino_range[1]
        dset_dest = grp.create_dataset('data', (full_shape[0], (sino_end-sino_start), full_shape[2]), dtype=dtype)
        dset_flat = grp.create_dataset('data_white', (1, (sino_end-sino_start), full_shape[2]), dtype=dtype)
        dset_dark = grp.create_dataset('data_dark', (1, (sino_end-sino_start), full_shape[2]), dtype=dtype)
    dset_flat[:, :, :] = np.ones(dset_flat.shape, dtype=dtype)
    dset_dark[:, :, :] = np.zeros(dset_dark.shape, dtype=dtype)
    comm.Barrier()
    flt = o['exchange/data_white'].value
    drk = o['exchange/data_dark'].value
    print('Method: {:s}'.format(method), kwargs)

    alloc_set = allocate_mpi_subsets(n_frames, size)
    for frame in alloc_set[rank]:
        t0 = time.time()
        print('    Rank: {:d}; current frame: {:d}.'.format(rank, frame))
        if sino_range is None:
            temp = dset_src[frame, :, :]
        else:
            sino_start = sino_range[0]
            sino_end = sino_range[1]
            temp = dset_src[frame, sino_start:sino_end, :]
        if corr_flat:
            temp = temp.reshape([1, temp.shape[0], temp.shape[1]])
            temp = tomopy.normalize(temp, flt, drk)
            temp = preprocess(temp)
            temp = np.squeeze(temp)
        temp = retrieve_phase(temp, method=method, **kwargs)
        dset_dest[frame, :, :] = temp.astype(dtype)
        print('    Done in {:.2f}s. '.format(time.time()-t0))

    f.close()
    comm.Barrier()
    return


def total_fusion(src_folder, dest_folder, dest_fname, file_grid, shift_grid, blend_method='pyramid', blend_method2=None,
                 blend_options={}, blend_options2={}, blur=None, color_correction=False, dtype='float16'):
    """
    Fuse hdf5 of all tiles in to one single file. MPI is supported.

    Parameters
    ----------
    blend_method: blending algorithm. If blend_method2 is None, the specified algorithm will be applied to both x and y
                  directions by default.
    blend_method2: secondary blending algorithm. If this option is not None, it will be applied for blending in y-
                   direction, while blend_method will be applied for x.
    """

    dest_fname = check_fname_ext(dest_fname, 'h5')
    if rank == 0:
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        if os.path.exists(dest_folder + '/' + dest_fname):
            print('Warning: File already exists. Continue anyway? (y/n) ')
            cont = raw_input()
            if cont in ['n', 'N']:
                exit()
            else:
                print('Old file will be overwritten.')
                os.remove(dest_folder + '/' + dest_fname)
        f = h5py.File(dest_folder + '/' + dest_fname)
    comm.Barrier()
    if rank != 0:
        assert os.path.exists(dest_folder + '/' + dest_fname)
        f = h5py.File(dest_folder + '/' + dest_fname)

    origin_dir = os.getcwd()
    os.chdir(src_folder)

    o = h5py.File(file_grid[0, 0])
    n_frames, y_cam, x_cam = o['exchange/data'].shape
    frames_per_rank = int(n_frames/size)
    grp = f.create_group('exchange')
    full_width = np.max(shift_grid[:, -1, 1]) + x_cam + 10
    full_height = np.max(shift_grid[-1, :, 0]) + y_cam + 10
    full_shape = (n_frames, full_height, full_width)
    dset_data = grp.create_dataset('data', full_shape, dtype=dtype)
    dset_flat = grp.create_dataset('data_white', (1, full_height, full_width), dtype=dtype)
    dset_dark = grp.create_dataset('data_dark', (1, full_height, full_width), dtype=dtype)
    dset_flat[:, :, :] = np.ones(dset_flat.shape, dtype=dtype)
    dset_dark[:, :, :] = np.zeros(dset_dark.shape, dtype=dtype)

    print('Started to build full hdf5.')
    t0 = time.time()
    alloc_set = allocate_mpi_subsets(n_frames, size)
    for frame in alloc_set[rank]:
        print('alloc set {:d}'.format(rank))
        print('    Rank: {:d}; current frame: {:d}..'.format(rank, frame))
        t00 = time.time()
        pano = np.zeros((full_height, full_width), dtype=dtype)
        # save_stdout = sys.stdout
        # sys.stdout = open('log', 'w')
        temp = build_panorama(src_folder, file_grid, shift_grid, frame=frame, method=blend_method, method2=blend_method2,
                              blend_options=blend_options, blend_options2=blend_options2, blur=blur, color_correction=color_correction)
        temp[np.isnan(temp)] = 0
        # sys.stdout = save_stdout
        pano[:temp.shape[0], :temp.shape[1]] = temp.astype(dtype)
        dset_data[frame, :, :] = pano
        print('    Frame {:d} done in {:.3f} s.'.format(frame, time.time() - t00))
    print('Data built and written in {:.3f} s.'.format(time.time() - t0))
    # try:
    #     os.remove('trash')
    # except:
    #     print('Please remove trash manually.')

    os.chdir(origin_dir)


def partial_center_alignment(file_grid, shift_grid, center_vec, src_folder, range_0=-5, range_1=5):
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
        sino, flt, drk = read_aps_32id_adaptive(fname, sino=(slice, slice+1))
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


def preprocess(dat, blur=None):

    dat[np.abs(dat) < 2e-3] = 2e-3
    dat[dat > 1] = 1
    dat = -np.log(dat)
    dat[np.where(np.isnan(dat) == True)] = 0
    if blur is not None:
        dat = gaussian_filter(dat, blur)

    return dat


def blur_hdf5(fname, sigma):
    """
    Apply Gaussian filter to each projection of a HDF5 for noise reduction.
    """
    f = h5py.File(fname)
    dset = f['exchange/data']
    nframes = dset.shape[0]
    alloc_sets = allocate_mpi_subsets(nframes, size)
    for frame in alloc_sets[rank]:
        print('Rank: {:d}; Frame: {:d}.'.format(rank, frame))
        dset[frame, :, :] = gaussian_filter(dset[frame, :, :], sigma)
    f.close()
    gc.collect()
