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


def center_mpi(ui):

    ds = int(ui.cent_ds)
    y_shift = int(ui.shiftgrid[:, 0, 0].mean()) / ds
    y_shift = y_shift if y_shift > 0 else 1
    n_rows = ui.shiftgrid.shape[0]

    if ui.ifmpi.get() == False:
            center_vec = np.zeros(n_rows)
            for center in range(ui.cent_start, ui.cent_end, ui.cent_step):
                center_vec[:] = center
                if ui.cent_type == 'dis':
                    recon_block(ui.file_grid, ui.shiftgrid/ds, ui.cent_src, ui.cent_dest,
                                (ui.cent_slice, ui.cent_slice+(n_rows-1)*y_shift+1), y_shift, center_vec,
                                blend_method='pyramid', algorithm=ui.cent_algo, mode=ui.cent_mode, test_mode=True)
                elif ui.cent_type == 'sin':
                    recon_hdf5(ui.cent_src, ui.cent_dest, (ui.cent_slice, ui.cent_slice+(n_rows-1)*y_shift+1), y_shift,
                               ui.shiftgrid/ds, center_vec=center_vec, algorithm=ui.cent_algo, mode=ui.cent_mode, test_mode=True)
    else:
        mpi_script_writer_center(ui)
        temp_path = os.path.join(ui.raw_folder, 'center.py')
        flag = None
        flag = os.system('mpirun -n ' + str(ui.merge_mpi_ncore) + ' python ' + temp_path)
        while True:
            if flag is not None:
                # os.remove(temp_path)
                break
            else:
                time.sleep(5)
    ui.boxMergOut.insert(END, 'Done.\n')
    return


def mpi_script_writer_center(ui):

    shutil.copyfile('mpi_common_head', os.path.join(ui.raw_folder, 'center.py'))
    f = open(os.path.join(ui.raw_folder, 'center.py'), 'a')
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
              'ds = {:d}\n'.format(ui.cent_ds),
              'y_shift = int(np.mean(shift_grid[:, 0, 0])) / ds\n',
              'y_shift = y_shift if y_shift > 0 else 1\n',
              'n_rows = shift_grid.shape[0]\n',
              'src = "{:s}"\n'.format(ui.cent_src),
              'dest = "{:s}"\n'.format(ui.cent_dest),
              'center_vec = np.zeros(n_rows)\n',
              'cent_start = {:d}\n'.format(ui.cent_start),
              'cent_end = {:d}\n'.format(ui.cent_end),
              'cent_step = {:d}\n'.format(ui.cent_step),
              'slice = {:d}\n'.format(ui.cent_slice),
              'for center in range(cent_start, cent_end, cent_step):\n',
              '    center_vec[:] = center\n'
             ]
    if ui.cent_type == 'dis':
        script.append('    tomosaic.recon_block(file_grid, shift_grid/ds, src, dest, \
        (slice, slice+(n_rows-1)*y_shift+1), y_shift, center_vec, blend_method="pyramid", algorithm="{:s}", \
        mode="{:s}", test_mode=True)'.format(ui.cent_algo, ui.cent_mode))
    elif ui.cent_type == 'sin':
        script.append('    tomosaic.recon_hdf5(src, dest, (slice, slice+(n_rows-1)*y_shift+1), y_shift, shift_grid/ds, \
        center_vec=center_vec, algorithm="{:s}", mode="{:s}", test_mode=True)').format(ui.cent_algo, ui.cent_mode)
    script.append('\n')
    f.writelines(script)
    f.close()
