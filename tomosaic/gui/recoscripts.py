import os, glob
import shutil
import subprocess
import time

import dxchange
import tomopy
from Tkinter import *
from tkMessageBox import showerror
import numpy as np

from tomosaic.misc import *
from tomosaic.util import *
from tomosaic.recon import *


def recon_mpi(ui):

    ds = int(ui.cent_ds)
    center_vec = np.loadtxt(ui.reco_cent, 'float32')
    center_vec = center_vec.reshape([1, center_vec.size])
    nchunk = int(ui.reco_chunk)

    if ui.ifmpi.get() == False:
            if ui.reco_type == 'dis':
                recon_block(ui.filegrid, ui.shiftgrid/ds, ui.reco_src, ui.reco_dest, (ui.reco_start, ui.reco_end),
                            ui.reco_step, center_vec, blend_method=ui.reco_blend, blend_options=ui.reco_blend_opts,
                            algorithm=ui.reco_algo, mode=ui.reco_mode, phase_retrieval=ui.reco_pr, **ui.reco_pr_opts)
            elif ui.reco_type == 'sin':
                recon_hdf5(ui.reco_src, ui.reco_dest, (ui.reco_start, ui.reco_end), ui.reco_step, ui.shiftgrid/ds,
                           center_vec=center_vec, algorithm=ui.reco_algo, mode=ui.reco_mode, phase_retrieval=ui.reco_pr,
                           chunk_size=nchunk, **ui.reco_pr_opts)
    else:
        mpi_script_writer_recon(ui)
        temp_path = os.path.join(ui.raw_folder, 'recon.py')
        flag = None
        flag = os.system('mpirun -n ' + str(ui.reco_mpi_ncore) + ' python ' + temp_path)
        while True:
            if flag is not None:
                # os.remove(temp_path)
                break
            else:
                time.sleep(5)
    ui.boxMergOut.insert(END, 'Done.\n')
    return


def mpi_script_writer_recon(ui):

    shutil.copyfile('mpi_common_head', os.path.join(ui.raw_folder, 'recon.py'))
    f = open(os.path.join(ui.raw_folder, 'recon.py'), 'a')
    script = ['raw_folder = "' + ui.raw_folder + '"\n',
              'os.chdir(raw_folder)\n',
              'prefix = "' + ui.prefix + '"\n',
              'file_list = tomosaic.get_files(raw_folder, prefix, type="h5")\n',
              'file_grid = tomosaic.start_file_grid(file_list, pattern=1)\n',
              'x_shift = ' + str(ui.x_shift) + '\n',
              'y_shift = ' + str(ui.y_shift) + '\n',
              '\n',
              'relative_shift = tomosaic.util.file2grid("shifts.txt")\n',
              'shift_grid = tomosaic.absolute_shift_grid(relative_shift, file_grid)\n',
              'ds = {:s}\n'.format(ui.reco_ds),
              'center_vec = np.loadtxt("{:s}", "float32")\n'.format(ui.reco_cent),
              'center_vec = center_vec.reshape([1, center_vec.size])\n'
              'src = "{:s}"\n'.format(ui.reco_src),
              'dest = "{:s}"\n'.format(ui.reco_dest),
              'reco_start = {:d}\n'.format(ui.reco_start),
              'reco_end = {:d}\n'.format(ui.reco_end),
              'reco_step = {:d}\n'.format(ui.reco_step),
              'pr_opts = {:s}\n'.format(ui.reco_pr_opts)
             ]
    pr = 'None' if ui.reco_pr is None else '"' + ui.reco_pr + '"'
    if ui.reco_type == 'dis':
        script.append('tomosaic.recon_block(file_grid, shift_grid/ds, src, dest, \
        (reco_start, reco_end), reco_step, center_vec, blend_method="{:s}", blend_options={:s}, algorithm="{:s}", \
        mode="{:s}", phase_retrieval={:s}, **pr_opts)\n'.format(ui.reco_blend, ui.reco_blend_opts, ui.reco_algo,
                                                                ui.reco_mode, ui.reco_pr))
    elif ui.reco_type == 'sin':
        script.append('tomosaic.recon_hdf5(src, dest, (reco_start, reco_end), reco_step, shift_grid, \
        center_vec=center_vec, algorithm="{:s}", mode="{:s}", phase_retrieval={:s}, chunk_size={:d}, **pr_opts)\n')\
            .format(ui.reco_algo, ui.reco_mode, ui.reco_pr, int(ui.reco_chunk))
    script.append('\n')
    f.writelines(script)
    f.close()
