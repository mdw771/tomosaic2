import os, glob
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

    if ui.use_mpi == False:

        refined_shift = refine_shift_grid(ui.file_grid, ui.shift_grid, motor_readout=(ui.y_shift, ui.x_shift))

    else:

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        name = MPI.Get_processor_name()