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

__author__ = ['Rafael Vescovi', 'Ming Du']
__credits__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['get_files',
           'get_index',
           'save_partial_frames',
           'save_partial_raw',
           'file2grid',
           'hdf5_retrieve_phase',
	       'preprocess',
           'g_shapes',
           'equalize_histogram',
           'pad_sinogram',
           'read_center_pos',
           'check_fname_ext',
           'image_downsample',
           'get_tilted_sinogram',
           'most_neighbor_clustering']

import os, glob, re
import warnings
import h5py
try:
    import netCDF4 as cdf
except:
    warnings.warn('netCDF4 cannot be imported.')
import numpy as np
import tomopy
import dxchange
from tomosaic.util.phase import retrieve_phase
from tomosaic.misc.misc import allocate_mpi_subsets, read_data_adaptive
import shutil
from scipy.ndimage import gaussian_filter
from scipy.misc import imread, imsave
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
from tomopy import downsample
import time
import six.moves
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


def get_files(folder, prefix, type='.h5', strict_matching=True):
    if not type.startswith('.'):
        type = '.' + type
    root = os.getcwd()
    os.chdir(folder)
    file_list = []
    for f in glob.glob(prefix + '*' + type):
        if strict_matching:
            if re.match(prefix + '.+[x,y]\d+' + type, f):
                file_list.append(f)
        else:
            file_list.append(f)
    if len(file_list) == 0:
        file_list = glob.glob(prefix + '*' + type)
    os.chdir(root)
    return file_list


def get_index(file_list, pattern=0):
    '''
    Get tile indices.
    :param file_list: list of files.
    :param pattern: pattern of naming. For files named with x_*_y_*, use
                    pattern=0. For files named with y_*_x_*, use pattern=1.
    :return: 
    '''
    if pattern == 0:
        regex = re.compile(r".+_x(\d+)_y(\d+).+")
        ind_buff = [m.group(1, 2) for l in file_list for m in [regex.search(l)] if m]
    elif pattern == 1:
        regex = re.compile(r".+_y(\d+)_x(\d+).+")
        ind_buff = [m.group(2, 1) for l in file_list for m in [regex.search(l)] if m]
    return np.asarray(ind_buff).astype('int')


def save_partial_frames(file_grid, save_folder, prefix, frame=0, data_format='aps_32id'):
    for (y, x), value in np.ndenumerate(file_grid):
        print(value)
        if (value != None):
            prj, flt, drk, _ = read_data_adaptive(value, proj=(frame, frame + 1), data_format=data_format)
            prj = tomopy.normalize(prj, flt, drk)
            prj = preprocess(prj)
            fname = prefix + 'Y' + str(y).zfill(2) + '_X' + str(x).zfill(2)
            dxchange.write_tiff(np.squeeze(prj), fname=os.path.join(save_folder, 'partial_frames', fname))


def save_partial_raw(file_list, save_folder, data_format='aps_32id'):
    for value in file_list:
        if (value != None):
            prj, flt, drk, _ = read_data_adaptive(value, proj=(0, 1), data_format=data_format)
            fname = value
            dxchange.write_tiff_stack(np.squeeze(flt), fname=os.path.join(save_folder, 'partial_flats', fname))
            dxchange.write_tiff_stack(np.squeeze(drk), fname=os.path.join(save_folder, 'partial_darks', fname))
            prj = prj.astype('float32')
            dxchange.write_tiff(np.squeeze(prj), fname=os.path.join(save_folder, 'partial_frames_raw', fname))


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

    with open(file_name, 'r') as infile:
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


def g_shapes(fname):
    try:
        return h5py.File(fname, "r")['exchange/data'].shape
    except:
        return cdf.Dataset(fname)['array_data'].shape


def image_downsample(img, ds):
    temp = downsample(downsample(img, level=int(np.log2(ds)), axis=1), level=int(np.log2(ds)), axis=2)
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
            cont = six.moves.input()
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


def preprocess(dat, blur=None, normalize_bg=False):

    dat[np.abs(dat) < 2e-3] = 2e-3
    dat[dat > 1] = 1
    if normalize_bg:
        dat = tomopy.normalize_bg(dat)
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


def get_histogram(img, bin_min, bin_max, n_bin=256):

    bins = np.linspace(bin_min, bin_max, n_bin)
    counts = np.zeros(n_bin+1)
    ind = np.squeeze(np.searchsorted(bins, img))
    for i in ind:
        counts[i] += 1
    return counts / img.size


def equalize_histogram(img, bin_min, bin_max, n_bin=256):

    histogram = get_histogram(img, bin_min, bin_max, n_bin=n_bin)
    bins = np.linspace(bin_min, bin_max, n_bin)
    e_table = np.zeros(n_bin + 1)
    res = np.zeros(img.shape)
    s_max = float(np.max(img))
    for i in range(bins.size):
        e_table[i] = s_max * np.sum(histogram[:i+1])
    ind = np.searchsorted(bins, img)
    for (y, x), i in np.ndenumerate(ind):
        res[y, x] = e_table[i]
    return res


def pad_sinogram(sino, length, mean_length=40, mode='edge'):

    if sino.ndim == 3:
        length = int(length)
        res = np.zeros([sino.shape[0], sino.shape[1], sino.shape[2] + length * 2])
        res[:, :, length:length+sino.shape[2]] = sino
        if mode == 'edge':
            for i in range(sino.shape[1]):
                mean_left = np.mean(sino[:, i, :mean_length], axis=1).reshape([sino.shape[0], 1])
                mean_right = np.mean(sino[:, i, -mean_length:], axis=1).reshape([sino.shape[0], 1])
                res[:, i, :length] = mean_left
                res[:, i, -length:] = mean_right
    else:
        res = np.zeros([sino.shape[0], sino.shape[1] + length * 2])
        mean_left = np.mean(sino[:, :mean_length], axis=1).reshape([sino.shape[0], 1])
        mean_right = np.mean(sino[:, -mean_length:], axis=1).reshape([sino.shape[0], 1])
        res[:, :length] = mean_left
        res[:, -length:] = mean_right
        res[:, length:length+sino.shape[1]] = sino
    return res


def read_center_pos(fname='center_pos.txt'):

    f = open(fname)
    lines = f.readlines()
    center_vec = np.zeros(len(lines))
    for line in lines:
        row, pos = line.split()
        center_vec[int(row)] = float(pos)
    return center_vec


def get_tilted_sinogram(fname, target_slice, tilt, preprocess_data=True):
    """
    Get a sinogram that is tilted about the theta-axis from the specified dataset.
    The tilting axis is assumed to be along the center of the FOV width.
    :param fname: name of the dataset.
    :param target_slice: the slice number where the tilting axis lies.
    :param tilt: tilting angle in degree. Positive = anticlockwise; negative = clockwise.
    :return: tilted sinogram.
    """
    shape = read_data_adaptive(fname, shape_only=True)
    fov2 = shape[2] / 2.
    tilt = np.deg2rad(tilt)
    range_l = target_slice + fov2 * np.tan(tilt)
    range_r = target_slice - fov2 * np.tan(tilt)
    if tilt > 0:
        range_l = int(np.ceil(range_l)) + 1
        range_r = int(np.floor(range_r))
    else:
        range_l = int(np.floor(range_l))
        range_r = int(np.ceil(range_r)) + 1
    dat, flt, drk, _ = read_data_adaptive(fname,
                                          sino=(min([range_l, range_r]), max([range_l, range_r])))
    if preprocess_data:
        dat = tomopy.normalize(dat, flt, drk)
        dat = preprocess(dat)
    else:
        dat[np.isnan(dat)] = 0

    tilted_block = rotate(dat, tilt, axes=(1, 2), reshape=True)
    mid_slice = int(dat.shape[1] / 2)

    return tilted_block[:, mid_slice:mid_slice+1, :]

    # values = dat.flatten()
    # dx, dy, dz = dat.shape
    # points = np.array([[x, y, z] for x in range(dx) for y in range(dy) for z in range(dz)])
    # print('interp st')
    # f = LinearNDInterpolator(points, values)
    # print('interp done')
    # range_l = target_slice + fov2 * np.tan(tilt)
    # range_r = target_slice - fov2 * np.tan(tilt)
    # xx = zz = []
    # for i in range(dx):
    #     xx += [i] * dz
    # xx = np.array(xx)
    # for i in range(dx):
    #     zz += range(dz)
    # zz = np.array(zz)
    # yy = np.tile(np.linspace(range_l, range_r, dz), dx)
    # print('gen')
    # sino = f(xx, yy, zz).reshape(dx, 1, dz)


def most_neighbor_clustering(data, radius):

    data = np.array(data)
    counter = np.zeros(len(data))
    for ind, i in enumerate(data):
        for j in data:
            if j != i and abs(j - i) < radius:
                counter[ind] += 1
    return data[np.where(counter == counter.max())]

