#!/usr/bin/env python# -*- coding: utf-8 -*-# ########################################################################## Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         ##                                                                         ## Copyright 2015. UChicago Argonne, LLC. This software was produced       ## under U.S. Government contract DE-AC02-06CH11357 for Argonne National   ## Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    ## U.S. Department of Energy. The U.S. Government has rights to use,       ## reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    ## UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        ## ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     ## modified to produce derivative works, such modified software should     ## be clearly marked, so as not to confuse it with the version available   ## from ANL.                                                               ##                                                                         ## Additionally, redistribution and use in source and binary forms, with   ## or without modification, are permitted provided that the following      ## conditions are met:                                                     ##                                                                         ##     * Redistributions of source code must retain the above copyright    ##       notice, this list of conditions and the following disclaimer.     ##                                                                         ##     * Redistributions in binary form must reproduce the above copyright ##       notice, this list of conditions and the following disclaimer in   ##       the documentation and/or other materials provided with the        ##       distribution.                                                     ##                                                                         ##     * Neither the name of UChicago Argonne, LLC, Argonne National       ##       Laboratory, ANL, the U.S. Government, nor the names of its        ##       contributors may be used to endorse or promote products derived   ##       from this software without specific prior written permission.     ##                                                                         ## THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     ## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       ## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       ## FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     ## Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        ## INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    ## BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        ## LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        ## CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      ## LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       ## ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         ## POSSIBILITY OF SUCH DAMAGE.                                             ## #########################################################################"""Module for image merging"""from __future__ import (absolute_import, division, print_function,                        unicode_literals)import loggingfrom scipy.ndimage import fourier_shiftimport numpy as npimport operatorimport dxchangelogger = logging.getLogger(__name__)__author__ = "Rafael Vescovi"__credits__ = "Doga Gursoy"__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."__docformat__ = 'restructuredtext en'__all__ = ['realign_image',           'realign_block',           'vig_image']def arrange_image(img1, img2, shift, order=1):    """    Place properly aligned image in buff    Parameters    ----------    img1 : ndarray        Substrate image array.    img2 : ndarray        Image being added on.    shift : float        Subpixel shift.    order : int        Order that images are arranged. If order is 1, img1 is written first and img2 is placed on the top. If order is        2, img2 is written first and img1 is placed on the top.    Returns    -------    newimg : ndarray        Output array.    """    rough_shift = get_roughshift(shift)    img2 = realign_image(img2, shift - rough_shift.astype('float'))    new_shape = map(int, map(max, map(operator.add, img2.shape, rough_shift), img1.shape))    newimg = np.empty(new_shape)    newimg[:, :] = np.NaN    if order == 1:        newimg[0:img1.shape[0], 0:img1.shape[1]] = img1        newimg[rough_shift[0]:rough_shift[0] + img2.shape[0], rough_shift[1]:rough_shift[1] + img2.shape[1]] = img2    elif order == 2:        newimg[rough_shift[0]:rough_shift[0] + img2.shape[0], rough_shift[1]:rough_shift[1] + img2.shape[1]] = img2        newimg[0:img1.shape[0], 0:img1.shape[1]] = img1    else:        print('Warning: images are not arranged due to misspecified order.')    return newimgdef realign_image(arr, shift, angle=0):    """    Translate and rotate image via Fourier    Parameters    ----------    arr : ndarray        Image array.    shift: float        Mininum and maximum values to rescale data.    angle: float, optional        Mininum and maximum values to rescale data.    Returns    -------    ndarray        Output array.    """    temp = fourier_shift(np.fft.fftn(arr), shift)    temp = np.fft.ifftn(temp)    temp = np.abs(temp).astype('float32')    return tempdef realign_block(arr, shift_vector, angle_vector=None, axis=0):    """    Translate and rotate image via Fourier    Parameters    ----------    arr : ndarray        Image array.    shift: float, optional        Mininum and maximum values to rescale data.    angle: float, optional        Mininum and maximum values to rescale data.    Returns    -------    ndarray        Output array.        :param arr:        :param shift_vector:        :param angle_vector:        :param axis:    """def get_roughshift(shift):    rough_shift = np.ceil(shift)    rough_shift[rough_shift < 0] = 0    return rough_shiftdef vig_image(img, vig=150):    temp = np.copy(img)    if img.ndim is 3:        for i in range(img.shape[0]):            temp[i, :, :] = vig_2d(temp[i, :, :], vig=vig)    else:        temp = vig_2d(temp, vig=vig)    return tempdef vig_2d(img, vig=150):    temp = np.copy(img)    step = 1.0/vig    for k in np.arange(vig):        pos = vig - k        temp[pos,:]  = temp[pos,:] * step * pos        temp[-pos,:] = temp[-pos,:] * step * pos        temp[:,pos]  = temp[:,pos] * step * pos        temp[:,-pos] = temp[:,-pos] * step * pos    return tempdef test_out(img, save_folder):    img[np.abs(img) < 1e-3] = 1e-3    img[img > 1] = 1    img = -np.log(img)    img[np.where(np.isnan(img) == True)] = 0    dxchange.write_tiff(np.squeeze(img), fname=save_folder+'test')