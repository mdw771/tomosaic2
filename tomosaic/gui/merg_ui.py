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
    lsMergMeth1 = ('Max', 'Min', 'Alpha', 'Pyramid', 'Poisson')
    optMergMeth1 = OptionMenu(frameMergMethod, ui.varMergMeth1, *lsMergMeth1)
    optMergMeth1.pack(side=LEFT, fill=X, expand=True)
    labMergMeth2 = Label(frameMergMethod, text=' Blending method (Y, opt):')
    labMergMeth2.pack(side=LEFT)
    ui.varMergMeth2 = StringVar()
    lsMergMeth2 = ('None', 'Max', 'Min', 'Alpha', 'Pyramid', 'Poisson')
    optMergMeth2 = OptionMenu(frameMergMethod, ui.varMergMeth2, *lsMergMeth2)
    optMergMeth2.pack(side=LEFT, fill=X, expand=True)


    frameMergSrc.pack(fill=X)
    frameMergDest.pack(fill=X)
    frameMergMethod.pack(fill=X)
    formMerg.pack(fill=X)
    bottMerg.pack(fill=X)