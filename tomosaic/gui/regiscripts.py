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


def read_shifts(ui):

    shift_grid = file2grid("shifts.txt")
    shift_grid = absolute_shift_grid(shift_grid, ui.file_grid)
    return shift_grid


def find_shifts_mpi(ui):

    if ui.ifmpi == False:
        ui.boxRegiOut.insert(END, 'Refining shifts...\n')
        relative_shift = refine_shift_grid(ui.file_grid, ui.shift_grid, motor_readout=(ui.y_shift, ui.x_shift))
        shift_grid = absolute_shift_grid(relative_shift, ui.file_grid)
    else:
        ui.boxRegiOut.insert(END, 'Generating temporary script file...\n')
        mpi_script_writer(ui)
        temp_path = os.path.join(ui.raw_folder, 'temp.py')
        ui.boxRegiOut.insert(END, 'Refining shifts...\n')
        ui.boxRegiOut.insert(END, 'Refer to initial terminal window for intermediate output.')
        flag = None
        flag = os.system('mpirun -n ' + str(ui.mpi_ncore) + ' python ' + temp_path)
        print flag
        while True:
            try:
                relative_shift = file2grid(os.path.join(ui.raw_folder, "shifts.txt"))
                shift_grid = absolute_shift_grid(relative_shift, ui.file_grid)
                ui.boxRegiOut.insert(END, 'Removing temporary script...\n')
                os.remove(temp_path)
                break
            except:
                time.sleep(5)
    return shift_grid, relative_shift


def mpi_script_writer(ui):

    shutil.copyfile('mpi_common_head', os.path.join(ui.raw_folder, 'temp.py'))
    f = open(os.path.join(ui.raw_folder, 'temp.py'), 'a')
    f.writelines(['raw_folder = "' + ui.raw_folder + '"\n',
                  'os.chdir(raw_folder)\n'
                  'prefix = "' + ui.prefix + '"\n',
                  'file_list = tomosaic.get_files(raw_folder, prefix, type="h5")\n',
                  'file_grid = tomosaic.start_file_grid(file_list, pattern=1)\n',
                  'x_shift = ' + str(ui.x_shift) + '\n',
                  'y_shift = ' + str(ui.y_shift) + '\n',
                  'shift_grid = tomosaic.start_shift_grid(file_grid, x_shift, y_shift)\n'
                  'tomosaic.refine_shift_grid(file_grid, shift_grid, motor_readout=(y_shift, x_shift))\n'])
    f.close()


def resave_shifts(ui):

    if ui.relative_shift is None:
        showerror(message='Relative shifts must be read or computed before resaving.')
    np.savetxt(ui._savepath, ui.relative_shift, fmt=str('%4.2f'))