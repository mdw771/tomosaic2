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


def merge_mpi(ui):

    if ui.ifmpi.get() == False:
        total_fusion(ui.merge_src, ui.merge_dest_folder, ui.merge_dest_fname, ui.filegrid, ui.shiftgrid,
                     blend_method=ui.merge_meth1, blend_options=ui.merge_opts1,
                     blend_method2=ui.merge_meth2, blend_options2=ui.merge_opts2)
    else:
        mpi_script_writer_merge(ui)
        temp_path = os.path.join(ui.raw_folder, 'merge.py')
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


def mpi_script_writer_merge(ui):

    shutil.copyfile('mpi_common_head', os.path.join(ui.raw_folder, 'merge.py'))
    f = open(os.path.join(ui.raw_folder, 'merge.py'), 'a')
    opt2 = '"' + ui.merge_meth2 + '"' if ui.merge_meth2 != 'Same as X' else 'None'
    f.writelines(['raw_folder = "' + ui.raw_folder + '"\n',
                  'os.chdir(raw_folder)\n',
                  'prefix = "' + ui.prefix + '"\n',
                  'file_list = tomosaic.get_files(raw_folder, prefix, type="h5")\n',
                  'file_grid = tomosaic.start_file_grid(file_list, pattern=1)\n',
                  'x_shift = ' + str(ui.x_shift) + '\n',
                  'y_shift = ' + str(ui.y_shift) + '\n',
                  '\n',
                  'relative_shift = tomosaic.util.file2grid("shifts.txt")\n',
                  'shift_grid = tomosaic.absolute_shift_grid(relative_shift, file_grid)\n',
                  'blend_options1 = ' + str(ui.merge_opts1) + '\n',
                  'blend_options2 = ' + str(ui.merge_opts2) + '\n',
                  'tomosaic.total_fusion("{:s}", "{:s}", "{:s}", file_grid, shift_grid, blend_method="{:s}", \
                  blend_method2={:s}, blend_options=blend_options1, blend_options2=blend_options2)\n'\
                  .format(ui.merge_src, ui.merge_dest_folder, ui.merge_dest_fname, ui.merge_meth1, ui.merge_meth2,
                          str(ui.merge_opts1), str(ui.merge_opts2)),
                  '\n'
                  ])
    f.close()
