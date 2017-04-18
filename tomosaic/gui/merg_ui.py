from functools import partial
import copy

from Tkinter import *
from ttk import Notebook
from tkFileDialog import *
from tkMessageBox import showerror, showwarning, showinfo

from metascripts import *
from regiscripts import *
from mergscripts import *


def mergtab_ui(ui):

    formMerg = Frame(ui.tabMerg)
    bottMerg = Frame(ui.tabMerg)

    # source line

    frameMergSrc = Frame(formMerg)
    labMergSrc = Label(frameMergSrc, text='Source folder:')
    labMergSrc.pack(side=LEFT)
    ui.entMergSrc = Entry(frameMergSrc)
    ui.entMergSrc.pack(side=LEFT, fill=X, expand=True)
    buttMergSrcBrowse = Button(frameMergSrc, text='Browse...')
    buttMergSrcBrowse.pack(side=LEFT)
    buttMergSrcDefault = Button(frameMergSrc, text='Same as raw folder')
    buttMergSrcDefault.pack(side=LEFT)

    # dest line

    frameMergDest = Frame(formMerg)
    labMergDest = Label(frameMergDest, text='Destination filename:')
    labMergDest.pack(side=LEFT)
    ui.entMergDest = Entry(frameMergDest)
    ui.entMergDest.pack(side=LEFT, fill=X, expand=True)
    buttMergDestBrowse = Button(frameMergDest, text='Browse...')
    buttMergDestBrowse.pack(side=LEFT)

    # method line

    frameMergMethod = Frame(formMerg)
    labMergMeth1 = Label(frameMergMethod, text='Blending method:')
    labMergMeth1.pack(side=LEFT)
    ui.varMergMeth1 = StringVar()
    lsMergMeth1 = ('Overlay', 'Max', 'Min', 'Alpha', 'Pyramid', 'Poisson')
    print type(ui)
    ui.optMergMeth1 = OptionMenu(frameMergMethod, ui.varMergMeth1, command=partial(updateOpt, ui, 1), *lsMergMeth1)
    ui.optMergMeth1.pack(side=LEFT, fill=X, expand=True)
    labMergMeth2 = Label(frameMergMethod, text=' Blending method (Y, opt):')
    labMergMeth2.pack(side=LEFT)
    ui.varMergMeth2 = StringVar()
    lsMergMeth2 = ('Same as X', 'Overlay', 'Max', 'Min', 'Alpha', 'Pyramid', 'Poisson')
    ui.optMergMeth2 = OptionMenu(frameMergMethod, ui.varMergMeth2, command=partial(updateOpt, ui, 2), *lsMergMeth2)
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
    ui.ifmpi = BooleanVar()
    radMPIY = Radiobutton(frameMergMPI, variable=ui.ifmpi, text='Yes', value=True)
    radMPIY.pack(side=LEFT)
    radMPIN = Radiobutton(frameMergMPI, variable=ui.ifmpi, text='No', value=False)
    radMPIN.pack(side=LEFT, padx=10)
    labMergNCore = Label(frameMergMPI, text='Number of processes to initiate:')
    labMergNCore.pack(side=LEFT)
    ui.entMergNCore = Entry(frameMergMPI)
    ui.entMergNCore.pack(side=LEFT, fill=X, expand=True)





    ui.frameOpt1Inp.pack(side=LEFT, fill=X)
    ui.frameOpt2Inp.pack(side=LEFT, fill=X)
    frameMergSrc.pack(fill=X)
    frameMergDest.pack(fill=X)
    frameMergMethod.pack(fill=X)
    ui.frameMergOpt1.pack(fill=X)
    ui.frameMergOpt2.pack(fill=X)
    frameMergMPI.pack(fill=X)
    formMerg.pack(fill=X)
    bottMerg.pack(fill=X)

def updateOpt(ui, uid, meth):

    print ui, meth

    if uid == 1:
        field = ui.frameOpt1Inp
    elif uid == 2:
        field = ui.frameOpt2Inp

    for w in field.winfo_children():
        w.destroy()

    if meth in ('Max', 'Min', 'Poisson'):
        lab0 = Label(field, text='No options available for the selected method.')
        lab0.pack(side=LEFT)
    elif meth == 'Overlay':
        lab0 = Label(field, text='Image on the top: ')
        lab0.pack(side=LEFT)
        ui.entOverlay = Entry(field)
        ui.entOverlay.pack(side=LEFT, fill=X)
    elif meth == 'Alpha':
        lab0 = Label(field, text='Alpha: ')
        lab0.pack(side=LEFT)
        ui.entAlpha = Entry(field)
        ui.entAlpha.pack(side=LEFT, fill=X)
    elif meth == 'Pyramid':
        lab0 = Label(field, text='Depth: ')
        lab0.pack(side=LEFT)
        entDepth = Entry(field)
        entDepth.pack(side=LEFT)
        lab1 = Label(field, text='Blur: ')
        lab1.pack(side=LEFT)
        entPyrBlur = Entry(field)
        entPyrBlur.pack(side=LEFT)


