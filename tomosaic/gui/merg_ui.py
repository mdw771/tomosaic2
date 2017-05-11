from functools import partial
import copy

from Tkinter import *
from ttk import Notebook
from tkFileDialog import *
from tkMessageBox import showerror, showwarning, showinfo

from metascripts import *
from regiscripts import *
from mergscripts import *
from tomosaic.merge.merge import _get_algorithm_kwargs


def mergtab_ui(ui):

    formMerg = Frame(ui.tabMerg)
    bottMerg = Frame(ui.tabMerg)

    # source line

    frameMergSrc = Frame(formMerg)
    labMergSrc = Label(frameMergSrc, text='Source folder:')
    labMergSrc.pack(side=LEFT)
    ui.entMergSrc = Entry(frameMergSrc)
    ui.entMergSrc.pack(side=LEFT, fill=X, expand=True)
    buttMergSrcBrowse = Button(frameMergSrc, text='Browse...', command=partial(getMergSrcFolder, ui))
    buttMergSrcBrowse.pack(side=LEFT)
    buttMergSrcDefault = Button(frameMergSrc, text='Same as raw folder', command=partial(getRawFolder, ui))
    buttMergSrcDefault.pack(side=LEFT)

    # dest line

    frameMergDest = Frame(formMerg)
    labMergDest = Label(frameMergDest, text='Destination filename:')
    labMergDest.pack(side=LEFT)
    ui.entMergDest = Entry(frameMergDest)
    ui.entMergDest.pack(side=LEFT, fill=X, expand=True)
    buttMergDestBrowse = Button(frameMergDest, text='Browse...', command=partial(getMergDestFile, ui))
    buttMergDestBrowse.pack(side=LEFT)

    # method line

    frameMergMethod = Frame(formMerg)
    labMergMeth1 = Label(frameMergMethod, text='Blending method:')
    labMergMeth1.pack(side=LEFT)
    ui.varMergMeth1 = StringVar()
    ui.varMergMeth1.set('Max')
    lsMergMeth1 = ('overlay', 'max', 'min', 'alpha', 'pyramid', 'poisson')
    ui.optMergMeth1 = OptionMenu(frameMergMethod, ui.varMergMeth1, command=partial(updateOpt, ui, 0), *lsMergMeth1)
    ui.optMergMeth1.pack(side=LEFT, fill=X, expand=True)
    labMergMeth2 = Label(frameMergMethod, text=' Blending method (Y, opt):')
    labMergMeth2.pack(side=LEFT)
    ui.varMergMeth2 = StringVar()
    ui.varMergMeth2.set('Same as X')
    lsMergMeth2 = ('Same as X', 'overlay', 'max', 'min', 'alpha', 'pyramid', 'poisson')
    ui.optMergMeth2 = OptionMenu(frameMergMethod, ui.varMergMeth2, command=partial(updateOpt, ui, 1), *lsMergMeth2)
    ui.optMergMeth2.pack(side=LEFT, fill=X, expand=True)

    # option 1 line

    ui.frameMergOpt1 = Frame(formMerg)
    labMergOpt1 = Label(ui.frameMergOpt1, text='Options: ')
    labMergOpt1.pack(side=LEFT)
    ui.frameOpt1Inp = Frame(ui.frameMergOpt1)
    labOpt1Default = Label(ui.frameOpt1Inp, text='Select a method.')
    labOpt1Default.pack(side=LEFT)

    # option 2 line

    ui.frameMergOpt2 = Frame(formMerg)
    labMergOpt2 = Label(ui.frameMergOpt2, text='Options: ')
    labMergOpt2.pack(side=LEFT)
    ui.frameOpt2Inp = Frame(ui.frameMergOpt2)
    labOpt2Default = Label(ui.frameOpt2Inp, text='Select a method.')
    labOpt2Default.pack(side=LEFT)

    # mpi line

    frameMergMPI = Frame(formMerg)
    labMergMPI = Label(frameMergMPI, text='Use MPI:')
    labMergMPI.pack(side=LEFT)
    radMPIY = Radiobutton(frameMergMPI, variable=ui.ifmpi, text='Yes', value=True)
    radMPIY.pack(side=LEFT)
    radMPIN = Radiobutton(frameMergMPI, variable=ui.ifmpi, text='No', value=False)
    radMPIN.pack(side=LEFT, padx=10)
    labMergNCore = Label(frameMergMPI, text='Number of processes to initiate:')
    labMergNCore.pack(side=LEFT)
    ui.entMergNCore = Entry(frameMergMPI)
    ui.entMergNCore.insert(0, '5')
    ui.entMergNCore.pack(side=LEFT, fill=X, expand=True)

    # out line

    frameMergOut = Frame(formMerg, height=245)
    frameMergOut.pack_propagate(False)
    ui.boxMergOut = Text(frameMergOut)
    ui.boxMergOut.insert(END, 'Merging\n')
    ui.boxMergOut.insert(END, 'Refer to initial terminal window for intermediate output.\n--------------\n')
    ui.boxMergOut.pack(side=LEFT, fill=BOTH, expand=YES)

    # button line

    buttMergLaunch = Button(bottMerg, text='Launch', command=partial(launchMerging, ui))
    buttMergLaunch.grid(row=0, column=0, sticky=W+E)
    buttMergConfirm = Button(bottMerg, text='Confirm parameters', command=partial(readMergPars, ui))
    buttMergConfirm.grid(row=0, column=1, sticky=W+E)

    ui.frameOpt1Inp.pack(side=LEFT, fill=X)
    ui.frameOpt2Inp.pack(side=LEFT, fill=X)
    frameMergSrc.pack(fill=X)
    frameMergDest.pack(fill=X)
    frameMergMethod.pack(fill=X)
    ui.frameMergOpt1.pack(fill=X)
    ui.frameMergOpt2.pack(fill=X)
    frameMergMPI.pack(fill=X)
    frameMergOut.pack(fill=X)
    formMerg.pack(fill=X)
    bottMerg.pack(side=BOTTOM)


def updateOpt(ui, uid, meth):

    if uid == 0:
        field = ui.frameOpt1Inp
    elif uid == 1:
        field = ui.frameOpt2Inp

    ui.lstAlpha = [None, None]
    ui.lstDepth = [None, None]
    ui.lstBlur = [None, None]
    ui.lstOrder = [None, None]

    for w in field.winfo_children():
        w.destroy()

    default_opts = _get_algorithm_kwargs()

    if meth in ('max', 'min', 'poisson'):
        lab0 = Label(field, text='No options available for the selected method.')
        lab0.pack(side=LEFT)
    elif meth == 'overlay':
        lab0 = Label(field, text='Image on the top (1 or 2): ')
        lab0.pack(side=LEFT)
        ui.lstOrder[uid] = Entry(field)
        ui.lstOrder[uid].insert(0, '2')
        ui.lstOrder[uid].pack(side=LEFT, fill=X)
    elif meth == 'alpha':
        lab0 = Label(field, text='Alpha: ')
        lab0.pack(side=LEFT)
        ui.lstAlpha[uid] = Entry(field)
        ui.lstAlpha[uid].insert(0, default_opts['alpha'])
        ui.lstAlpha[uid].pack(side=LEFT, fill=X)
    elif meth == 'pyramid':
        lab0 = Label(field, text='Depth: ')
        lab0.pack(side=LEFT)
        ui.lstDepth[uid] = Entry(field)
        ui.lstDepth[uid].insert(0, default_opts['depth'])
        ui.lstDepth[uid].pack(side=LEFT)
        lab1 = Label(field, text=' Blur: ')
        lab1.pack(side=LEFT)
        ui.lstBlur[uid] = Entry(field)
        ui.lstBlur[uid].insert(0, default_opts['blur'])
        ui.lstBlur[uid].pack(side=LEFT)
    else:
        lab0 = Label(field, text='No options available for the selected method.')
        lab0.pack(side=LEFT)


def getMergSrcFolder(ui):

    src = askdirectory()
    ui.entMergSrc.insert(0, src)


def getRawFolder(ui):

    try:
        ui.entMergSrc.insert(0, ui.raw_folder)
    except:
        showerror(message='Raw folder must be specified in metadata tab.')


def getMergDestFile(ui):

    dest = asksaveasfilename()
    ui.entMergDest.insert(0, dest)


def launchMerging(ui):

    readMergPars(ui)
    merge_mpi(ui)


def buildMergOpts(ui, meth, dict, uid):

    if meth == 'overlay':
        dict['order'] = 1 if ui.lstOrder[uid].get() == 2 else 2
    elif meth == 'alpha':
        dict['alpha'] = float(ui.lstAlpha[uid].get())
    elif meth == 'pyramid':
        dict['depth'] = int(ui.lstDepth[uid].get())
        dict['blur'] = float(ui.lstBlur[uid].get())


def readMergPars(ui):

    ui.merge_src = ui.entMergSrc.get()
    ui.merge_dest_fname = os.path.basename(ui.entMergDest.get())
    ui.merge_dest_folder = ui.entMergDest.get().split(ui.merge_dest_fname)[0]
    ui.merge_meth1 = ui.varMergMeth1.get()
    ui.merge_meth2 = ui.varMergMeth2.get()
    buildMergOpts(ui, ui.merge_meth1, ui.merge_opts1, 0)
    buildMergOpts(ui, ui.merge_meth2, ui.merge_opts2, 1)
    ui.merge_mpi_ncore = int(ui.entMergNCore.get())
    ui.boxMergOut.insert(END, 'Parameters read.\n')
