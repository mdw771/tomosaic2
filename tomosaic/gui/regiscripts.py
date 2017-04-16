import os, glob
import shutil
import subprocess

import dxchange
import tomopy
from Tkinter import *

from tomosaic.misc import *
from tomosaic.util import *


def read_shifts(ui):

    shift_grid = file2grid("shifts.txt")
    shift_grid = absolute_shift_grid(shift_grid, ui.file_grid)
    return shift_grid


def find_shifts_mpi(ui):

    if ui.ifmpi == False:
        refined_shift = refine_shift_grid(ui.file_grid, ui.shift_grid, motor_readout=(ui.y_shift, ui.x_shift))
    else:
        mpi_script_writer(ui)
        # os.system('mpirun -n ' + str(ui.mpi_ncore))

def mpi_script_writer(ui):

    shutil.copyfile('mpi_common_head', os.path.join(ui.raw_folder, 'temp.py'))
    f = open(os.path.join(ui.raw_folder, 'temp.py'), 'w')
    f.writelines(['raw_folder = ' + ui.raw_folder,
                  'prefix = ' + ui.prefix,
                  "file_list = tomosaic.get_files('.', prefix, type='h5')",
                  "file_grid = tomosaic.start_file_grid(file_list, pattern=1)",
                  'x_shift = ' + str(ui.x_shift),
                  'y_shift = ' + str(ui.y_shift),
                  "refine_shift_grid(file_grid, shift_grid, motor_readout=(y_shift, x_shift))"])
    f.close()


