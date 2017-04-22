from functools import partial
import copy

from Tkinter import *
from ttk import Notebook
from tkFileDialog import *
from tkMessageBox import showerror, showwarning, showinfo

from metascripts import *
from centscripts import *


def centtab_ui(ui):

    formCent = Frame(ui.tabCent)
    bottCent = Frame(ui.tabCent)

    # type line

    frameCentType = Frame(formCent)
    labCentType = Label(frameCentType, text='Operation type:')
    labCentType.pack(side=LEFT)
    ui.varCentType = StringVar()
    ui.varCentType.set('dis')
    radCentDis = Radiobutton(frameCentType, variable=ui.varCentType, text='Discrete files', value='dis')
    radCentDis.pack(side=LEFT)
    radCentSin = Radiobutton(frameCentType, variable=ui.varCentType, text='Single HDF5', value='sin')
    radCentSin.pack(side=LEFT)

    # source line

    frameCentSrc = Frame(formCent)
    labCentSrc = Label(frameCentSrc, text='Source folder or file:')
    labCentSrc.pack(side=LEFT)
    ui.entCentSrc = Entry(frameCentSrc)
    ui.entCentSrc.pack(side=LEFT, fill=X, expand=True)
    buttCentSrc = Button(frameCentSrc, text='Browse...', command=partial(getCentSrc, ui))
    buttCentSrc.pack(side=LEFT)
    buttCentRaw = Button(frameCentSrc, text='Same as raw folder', command=partial(getRawFolder, ui))
    buttCentRaw.pack(side=LEFT)

    # destination line

    frameCentDest = Frame(formCent)
    labCentDest = Label(frameCentDest, text='Destination directory:')
    labCentDest.pack(side=LEFT)
    ui.entCentDest = Entry(frameCentDest)
    ui.entCentDest.pack(side=LEFT, fill=X, expand=True)
    buttCentDestBrowse = Button(frameCentDest, text='Browse...', command=partial(getCentDest, ui))
    buttCentDestBrowse.pack(side=LEFT)
    buttCentDestDefault = Button(frameCentDest, text='Use default', command=partial(getCentDestDefault, ui))
    buttCentDestDefault.pack(side=LEFT)

    # range line

    frameCentRange = Frame(formCent)
    labelCentRange = Label(frameCentRange, text='Test range:')
    labelCentRange.pack(side=LEFT)
    ui.entCentStart = Entry(frameCentRange)
    ui.entCentStart.pack(side=LEFT)
    labelCentHyphen = Label(frameCentRange, text='-')
    labelCentHyphen.pack(side=LEFT)
    ui.entCentEnd = Entry(frameCentRange)
    ui.entCentEnd.pack(side=LEFT)
    labelCentStep = Label(frameCentRange, text='Step:')
    labelCentStep.pack(side=LEFT)
    ui.entCentStep = Entry(frameCentRange)
    ui.entCentStep.insert(0, '1')
    ui.entCentStep.pack(side=LEFT)

    # misc line

    frameCentMisc = Frame(formCent)
    labSliceNo = Label(frameCentMisc, text='Slice position:')
    labSliceNo.pack(side=LEFT)
    ui.entCentSlice = Entry(frameCentMisc)
    ui.entCentSlice.pack(side=LEFT, fill=X, expand=True)
    labCentDs = Label(frameCentMisc, text=' Downsampling:')
    labCentDs.pack(side=LEFT)
    ui.entCentDs = Entry(frameCentMisc)
    ui.entCentDs.insert(0, '1')
    ui.entCentDs.pack(side=LEFT)

    # algo line
    frameCentAlgo = Frame(formCent)
    labCentAlgo = Label(frameCentAlgo, text='Reconstruction algorithm:')
    labCentAlgo.pack(side=LEFT)
    ui.entCentAlgo = Entry(frameCentAlgo)
    ui.entCentAlgo.insert(0, 'gridrec')
    ui.entCentAlgo.pack(side=LEFT, fill=X, expand=True)
    labCentMode = Label(frameCentAlgo, text='Mode:')
    labCentMode.pack(side=LEFT)
    ui.varCentMode = StringVar()
    ui.varCentMode.set('180')
    opts = ('180', '360')
    optCentMode = OptionMenu(frameCentAlgo, ui.varCentMode, *opts)
    optCentMode.pack(side=LEFT)


    # mpi line

    frameCentMPI = Frame(formCent)
    labCentMPI = Label(frameCentMPI, text='Use MPI:')
    labCentMPI.pack(side=LEFT)
    ui.ifmpi = BooleanVar()
    radMPIY = Radiobutton(frameCentMPI, variable=ui.ifmpi, text='Yes', value=True)
    radMPIY.pack(side=LEFT)
    radMPIN = Radiobutton(frameCentMPI, variable=ui.ifmpi, text='No', value=False)
    radMPIN.pack(side=LEFT, padx=10)
    labCentNCore = Label(frameCentMPI, text='Number of processes to initiate:')
    labCentNCore.pack(side=LEFT)
    ui.entCentNCore = Entry(frameCentMPI)
    ui.entCentNCore.insert(0, '5')
    ui.entCentNCore.pack(side=LEFT, fill=X, expand=True)

    # box line

    frameCentOut = Frame(formCent, height=210)
    frameCentOut.pack_propagate(False)
    ui.boxCentOut = Text(frameCentOut)
    ui.boxCentOut.insert(END, 'Center optimization\n')
    ui.boxCentOut.insert(END, 'Refer to initial terminal window for intermediate output.\n--------------\n')
    ui.boxCentOut.pack(side=LEFT, fill=BOTH, expand=YES)

    # button line

    buttCentLaunch = Button(bottCent, text='Launch', command=partial(launchCenter, ui))
    buttCentLaunch.grid(row=0, column=0, sticky=W+E)
    buttCentConfirm = Button(bottCent, text='Confirm parameters', command=partial(readCentPars, ui))
    buttCentConfirm.grid(row=0, column=1, sticky=W+E)

    frameCentType.pack(fill=X)
    frameCentSrc.pack(fill=X)
    frameCentDest.pack(fill=X)
    frameCentRange.pack(fill=X)
    frameCentMisc.pack(fill=X)
    frameCentAlgo.pack(fill=X)
    frameCentMPI.pack(fill=X)
    frameCentOut.pack(fill=X)
    formCent.pack(fill=X)
    bottCent.pack(side=BOTTOM)


def getCentSrc(ui):

    if ui.varCentType.get() == 'dis':
        src = askdirectory()
    elif ui.varCentType.get() == 'sin':
        src = askopenfilename()
    ui.entCentSrc.insert(0, src)


def getRawFolder(ui):

    try:
        ui.entCentSrc.insert(0, ui.raw_folder)
    except:
        showerror(message='Raw folder must be specified in metadata tab.')


def getCentDest(ui):

    src = askdirectory()
    ui.entCentDest.insert(0, src)


def getCentDestDefault(ui):

    ui.entCentDest.insert(0, os.path.join(ui.raw_folder, 'center'))


def readCentPars(ui):

    ui.cent_type = ui.varCentType.get()
    ui.cent_src = ui.entCentSrc.get()
    ui.cent_dest = ui.entCentDest.get()
    ui.cent_start = int(ui.entCentStart.get())
    ui.cent_end = int(ui.entCentEnd.get())
    ui.cent_step = int(ui.entCentStep.get())
    ui.cent_slice = int(ui.entCentSlice.get())
    ui.cent_algo = ui.entCentAlgo.get()
    ui.cent_mpi_ncore = ui.entCentNCore.get()
    ui.cent_ds = int(ui.entCentDs.get())
    ui.boxCentOut.insert(END, 'Parameters read.\n')


def launchCenter(ui):

    readCentPars(ui)
    center_mpi(ui)
    ui.boxCentOut.insert(END, 'Done.\n')

