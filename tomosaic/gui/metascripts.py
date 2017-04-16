import os, glob

import dxchange
import tomopy
from Tkinter import *

from tomosaic.misc import *
from tomosaic.util import *


def get_filelist(ui):

    return get_files(ui.raw_folder, ui.prefix)

def get_filegrid(ui):

    return start_file_grid(ui.filelist, pattern=1)

def get_rough_shiftgrid(ui):

    return start_shift_grid(ui.filegrid, ui.x_shift, ui.y_shift)


def write_first_frames(ui):

    root = os.getcwd()
    os.chdir(ui.raw_folder)
    try:
        os.mkdir('first_frames')
    except:
        pass
    for i in ui.filelist:
        ui.boxMetaOut.insert(END, i + '\n')
        prj, flt, drk = dxchange.read_aps_32id(i, proj=(0, 1))
        prj = tomopy.normalize(prj, flt, drk)
        dxchange.write_tiff(prj, os.path.join('first_frames', os.path.splitext(i)[0]))
