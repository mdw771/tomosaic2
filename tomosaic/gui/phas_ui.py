from functools import partial
import copy

from Tkinter import *
from ttk import Notebook
from tkFileDialog import *
from tkMessageBox import showerror, showwarning, showinfo

from metascripts import *
from phasscripts import *
from tomosaic.util.phase import _get_pr_kwargs


def phastab_ui(ui):
    
    formPhas = Frame(ui.tabPhas)
    bottPhas = Frame(ui.tabPhas)

    # source line

    framePhasSrc = Frame(formPhas)
    labPhasSrc = Label(framePhasSrc, text='Source file:')
    labPhasSrc.pack(side=LEFT)
    ui.entPhasSrc = Entry(framePhasSrc)
    ui.entPhasSrc.pack(side=LEFT, fill=X, expand=True)
    buttPhasSrcBrowse = Button(framePhasSrc, text='Browse...', command=partial(getPhasSrc, ui))
    buttPhasSrcBrowse.pack(side=LEFT)
    buttPhasSrcDefault = Button(framePhasSrc, text='Fill in raw folder', command=partial(getRawFolder, ui))
    buttPhasSrcDefault.pack(side=LEFT)

    # dest line

    framePhasDest = Frame(formPhas)
    labPhasDest = Label(framePhasDest, text='Destination filename:')
    labPhasDest.pack(side=LEFT)
    ui.entPhasDest = Entry(framePhasDest)
    ui.entPhasDest.pack(side=LEFT, fill=X, expand=True)
    buttPhasDestBrowse = Button(framePhasDest, text='Browse...', command=partial(getPhasDestFile, ui))
    buttPhasDestBrowse.pack(side=LEFT)

    # pr line

    framePhasPr = Frame(formPhas)
    labPr = Label(framePhasPr, text='Phase retrieval')
    labPr.pack(side=LEFT)
    pr_opts = ('paganin',)
    ui.varPhasMeth = StringVar()
    ui.varPhasMeth.set('paganin')
    optPhasPr = OptionMenu(framePhasPr, ui.varPhasMeth, command=partial(updatePrOpts, ui), *pr_opts)
    optPhasPr.pack(side=LEFT)
    labUnit = Label(framePhasPr, text='Dimensions: length (cm), energy (keV)')
    labUnit.pack(side=LEFT)

    # pr options

    ui.framePhasPrOpts = Frame(formPhas)
    labPrOpts = Label(ui.framePhasPrOpts, text='Phase retrieval options will be shown here if a method is selected.')
    labPrOpts.pack(side=LEFT)
    updatePrOpts(ui, ui.varPhasMeth.get())

    # mpi line

    framePhasMPI = Frame(formPhas)
    labPhasMPI = Label(framePhasMPI, text='Use MPI:')
    labPhasMPI.pack(side=LEFT)
    ui.phas_ifmpi = BooleanVar()
    radMPIY = Radiobutton(framePhasMPI, variable=ui.phas_ifmpi, text='Yes', value=True)
    radMPIY.pack(side=LEFT)
    radMPIN = Radiobutton(framePhasMPI, variable=ui.phas_ifmpi, text='No', value=False)
    radMPIN.pack(side=LEFT, padx=10)
    labPhasNCore = Label(framePhasMPI, text='Number of processes to initiate:')
    labPhasNCore.pack(side=LEFT)
    ui.entPhasNCore = Entry(framePhasMPI)
    ui.entPhasNCore.insert(0, '5')
    ui.entPhasNCore.pack(side=LEFT, fill=X, expand=True)

    # out line

    framePhasOut = Frame(formPhas, height=260)
    framePhasOut.pack_propagate(False)
    ui.boxPhasOut = Text(framePhasOut)
    ui.boxPhasOut.insert(END, 'Phasing\n')
    ui.boxPhasOut.insert(END, 'Refer to initial terminal window for intermediate output.\n--------------\n')
    ui.boxPhasOut.pack(side=LEFT, fill=BOTH, expand=YES)

    # button line

    buttPhasLaunch = Button(bottPhas, text='Launch', command=partial(launchPhasing, ui))
    buttPhasLaunch.grid(row=0, column=0, sticky=W + E)
    buttPhasConfirm = Button(bottPhas, text='Confirm parameters', command=partial(readPhasPars, ui))
    buttPhasConfirm.grid(row=0, column=1, sticky=W + E)

    framePhasSrc.pack(fill=X)
    framePhasDest.pack(fill=X)
    framePhasPr.pack(fill=X)
    ui.framePhasPrOpts.pack(fill=X)
    framePhasMPI.pack(fill=X)
    framePhasOut.pack(fill=X)
    formPhas.pack(fill=X)
    bottPhas.pack(side=BOTTOM)


def getPhasSrc(ui):

    src = askopenfilename()
    ui.entPhasSrc.insert(0, src)


def getRawFolder(ui):

    try:
        ui.entPhasSrc.insert(0, ui.raw_folder)
    except:
        showerror(message='Raw folder must be specified in metadata tab.')


def getPhasDestFile(ui):

    dest = asksaveasfilename()
    ui.entPhasDest.insert(0, dest)


def launchPhasing(ui):

    readPhasPars(ui)
    phas_mpi(ui)
    ui.boxPhasOut.insert(END, 'Done.\n')


def buildPhasOpts(ui, meth, dict, uid):
    if meth == 'overlay':
        dict['order'] = 1 if ui.lstOrder[uid].get() == 2 else 2
    elif meth == 'alpha':
        dict['alpha'] = float(ui.lstAlpha[uid].get())
    elif meth == 'pyramid':
        dict['depth'] = int(ui.lstDepth[uid].get())
        dict['blur'] = float(ui.lstBlur[uid].get())


def readPhasPars(ui):

    ui.phas_src_fanme = os.path.basename(ui.entPhasSrc.get())
    ui.phas_src_folder = ui.entPhasSrc.get().split(ui.phas_src_fname)[0]
    ui.phas_dest_fname = os.path.basename(ui.entPhasDest.get())
    ui.phas_dest_folder = ui.entPhasDest.get().split(ui.phas_dest_fname)[0]
    ui.phas_meth = ui.varPhasMeth.get()
    buildPhasOpts(ui, ui.phas_meth, ui.phas_opts, 0)
    ui.phas_mpi_ncore = int(ui.entPhasNCore.get())
    ui.boxPhasOut.insert(END, 'Parameters read.\n')


def updatePrOpts(ui, meth):

    for w in ui.framePhasPrOpts.winfo_children():
        w.destroy()

    default_opts = _get_pr_kwargs()

    if meth == 'paganin':
        width = 10
        ui.lab1 = Label(ui.framePhasPrOpts, text='Px size:')
        ui.lab1.grid(row=0, column=0)
        ui.ent1 = Entry(ui.framePhasPrOpts)
        ui.ent1.grid(row=0, column=1)
        ui.ent1.insert(0, default_opts['pixel'])
        ui.lab2 = Label(ui.framePhasPrOpts, text='Dist:')
        ui.lab2.grid(row=0, column=2)
        ui.ent2 = Entry(ui.framePhasPrOpts)
        ui.ent2.grid(row=0, column=3)
        ui.ent2.insert(0, default_opts['distance'])
        ui.lab3 = Label(ui.framePhasPrOpts, text='E:')
        ui.lab3.grid(row=0, column=4)
        ui.ent3 = Entry(ui.framePhasPrOpts)
        ui.ent3.grid(row=0, column=5)
        ui.ent3.insert(0, default_opts['energy'])
        ui.lab4 = Label(ui.framePhasPrOpts, text='Alpha:')
        ui.lab4.grid(row=0, column=6)
        ui.ent4 = Entry(ui.framePhasPrOpts)
        ui.ent4.grid(row=0, column=7)
        ui.ent4.insert(0, default_opts['alpha_paganin'])
        ui.ent1['width'] = width
        ui.ent2['width'] = width
        ui.ent3['width'] = width
        ui.ent4['width'] = width

    else:
        ui.labNone = Label(ui.framePhasPrOpts, text='No options available.')
        ui.labNone.pack(side=LEFT)