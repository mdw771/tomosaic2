import os, glob

import dxchange
import tomopy
from Tkinter import *

from merg_ui import updateOpt
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

def write_pars(ui, dict):

    for key in dict.keys():
        if dict[key] is None:
            dict[key] = ''
    ui.entRawPath.delete(0, END)
    ui.entRawPath.insert(0, dict['raw_folder'])
    ui.entPrefix.delete(0, END)
    ui.entPrefix.insert(0, dict['prefix'])
    ui.entRoughX.delete(0, END)
    ui.entRoughX.insert(0, dict['x_shift'])
    ui.entRoughY.delete(0, END)
    ui.entRoughY.insert(0, dict['y_shift'])
    ui.entShiftPath.delete(0, END)
    ui.entShiftPath.insert(0, dict['shift_path'])
    ui.entRegiNCore.delete(0, END)
    ui.entRegiNCore.insert(0, dict['mpi_ncore'])
    ui.entMergSrc.delete(0, END)
    ui.entMergSrc.insert(0, dict['merge_src'])
    ui.entMergDest.delete(0, END)
    ui.entMergDest.insert(0, os.path.join(dict['merge_dest_folder'], dict['merge_dest_fname']))
    ui.entMergNCore.delete(0, END)
    ui.entMergNCore.insert(0, dict['merge_mpi_ncore'])
    ui.varMergMeth1.set(ui.merge_meth1)
    ui.varMergMeth2.set(ui.merge_meth2)
    if dict['ifmpi']:
        ui.ifmpi.set(True)
    else:
        ui.ifmpi.set(False)
    updateOpt(ui, 0, ui.varMergMeth1.get())
    updateOpt(ui, 1, ui.varMergMeth2.get())



