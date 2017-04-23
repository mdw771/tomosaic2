import os, glob

import dxchange
import tomopy
from Tkinter import *

import merg_ui
import reco_ui
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
    ui.entCentSrc.delete(0, END)
    ui.entCentSrc.insert(0, dict['cent_src'])
    ui.entCentDest.delete(0, END)
    ui.entCentDest.insert(0, dict['cent_dest'])
    ui.entCentStart.delete(0, END)
    ui.entCentStart.insert(0, dict['cent_start'])
    ui.entCentEnd.delete(0, END)
    ui.entCentEnd.insert(0, dict['cent_end'])
    ui.entCentStep.delete(0, END)
    ui.entCentStep.insert(0, dict['cent_step'])
    ui.entCentDs.delete(0, END)
    ui.entCentDs.insert(0, dict['cent_ds'])
    ui.entCentSlice.delete(0, END)
    ui.entCentSlice.insert(0, dict['cent_slice'])
    ui.entCentAlgo.delete(0, END)
    ui.entCentAlgo.insert(0, dict['cent_algo'])
    ui.entCentNCore.delete(0, END)
    ui.entCentNCore.insert(0, dict['cent_mpi_ncore'])
    ui.varCentMode.set(ui.cent_mode)
    ui.entRecoSrc.delete(0, END)
    ui.entRecoSrc.insert(0, dict['reco_src'])
    ui.entRecoDest.delete(0, END)
    ui.entRecoDest.insert(0, dict['reco_dest'])
    ui.entRecoCent.delete(0, END)
    ui.entRecoCent.insert(0, dict['reco_cent'])
    ui.entRecoStart.delete(0, END)
    ui.entRecoStart.insert(0, dict['reco_start'])
    ui.entRecoEnd.delete(0, END)
    ui.entRecoEnd.insert(0, dict['reco_end'])
    ui.entRecoStep.delete(0, END)
    ui.entRecoStep.insert(0, dict['reco_step'])
    ui.varRecoAlgo.set(dict['reco_algo'])
    ui.varRecoMode.set(dict['reco_mode'])
    ui.entRecoDs.delete(0, END)
    ui.entRecoDs.insert(0, dict['reco_ds'])
    ui.varRecoPr.set(str(dict['reco_pr']))
    ui.entRecoNCore.delete(0, END)
    ui.entRecoNCore.insert(0, dict['reco_mpi_ncore'])
    ui.varBlendMeth.set(dict['reco_blend'])
    ui.entRecoChunk.delete(0, END)
    ui.entRecoChunk.insert(0, dict['reco_chunk'])
    if dict['ifmpi']:
        ui.ifmpi.set(True)
    else:
        ui.ifmpi.set(False)
    if dict['cent_type'] == 'dis':
        ui.varCentType.set('dis')
    elif dict['cent_type'] == 'sin':
        ui.varCentType.set('sin')
    if dict['reco_type'] == 'dis':
        ui.varRecoType.set('dis')
    elif dict['reco_type'] == 'sin':
        ui.varRecoType.set('sin')
    merg_ui.updateOpt(ui, 0, ui.varMergMeth1.get())
    merg_ui.updateOpt(ui, 1, ui.varMergMeth2.get())
    reco_ui.updateAlgoOpts(ui, ui.varRecoAlgo.get())
    reco_ui.updatePrOpts(ui, ui.varRecoPr.get())
    reco_ui.updateBlendOpt(ui, ui.varBlendMeth.get())

