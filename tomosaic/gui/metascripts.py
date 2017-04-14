import os, glob

import dxchange
import tomopy

from tomosaic.misc import *
from tomosaic.util import *


def get_filelist(ui):

    return get_files(ui.raw_folder, ui.prefix)

def write_first_frames(ui):

    root = os.getcwd()
    os.chdir(ui.raw_folder)
    os.mkdir('first_frames')
    for i in ui.filelist:
        prj, flt, drk = dxchange.read_aps_32id(i, proj=(0, 1))
        prj = tomopy.normalize(prj, flt, drk)
        dxchange.write_tiff(prj, os.path.join('first_frames', os.path.splitext(i)[0]))
