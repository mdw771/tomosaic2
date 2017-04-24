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


def phase_mpi(ui):

    if ui.ifmpi.get() == False:
            hdf5_retrieve_phase(ui.phas_src_folder, ui.phas_src_fname, ui.phas_dest_folder, ui.phas_dest_fname,
                                     method=ui.phas_meth)
    else:
        mpi_script_writer_phase(ui)
        temp_path = os.path.join(ui.raw_folder, 'phase.py')
        flag = None
        flag = os.system('mpirun -n ' + str(ui.phas_mpi_ncore) + ' python ' + temp_path)
        while True:
            if flag is not None:
                # os.remove(temp_path)
                break
            else:
                time.sleep(5)
    return


def mpi_script_writer_phase(ui):

    shutil.copyfile('mpi_common_head', os.path.join(ui.raw_folder, 'phase.py'))
    f = open(os.path.join(ui.raw_folder, 'phase.py'), 'a')
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
              'src_folder = "{:s}"\n'.format(ui.phas_src_folder),
              'src_fname = "{:s}"\n'.format(ui.phas_src_fname),
              'dest_folder = "{:s}"\n'.format(ui.phas_dest_folder),
              'dest_fname = "{:s}"\n'.format(ui.phas_dest_fname),
              'method = "{:s}"\n'.format(ui.phas_meth),
              'pr_opts = {:s}\n'.format(ui.phas_pr_opts),
              'hdf5_retrieve_phase(src_folder, src_fname, dest_folder, dest_fname, method=method)\n'
              ]
    script.append('\n')
    f.writelines(script)
    f.close()
