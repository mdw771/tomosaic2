from functools import partial
import copy

from Tkinter import *
from ttk import Notebook
from tkFileDialog import *
from tkMessageBox import showerror, showwarning, showinfo

from metascripts import *
from centscripts import *


def recotab_ui(ui):

    formReco = Frame(ui.tabReco)
    bottReco = Frame(ui.tabReco)

    # type line

    frameRecoType = Frame(formReco)
    labRecoType = Label(frameRecoType, text='Operation type:')
    labRecoType.pack(side=LEFT)
    ui.varRecoType = StringVar()
    ui.varRecoType.set('dis')
    radRecoDis = Radiobutton(frameRecoType, variable=ui.varRecoType, text='Discrete files', value='dis')
    radRecoDis.pack(side=LEFT)
    radRecoSin = Radiobutton(frameRecoType, variable=ui.varRecoType, text='Single HDF5', value='sin')
    radRecoSin.pack(side=LEFT)

    # source line

    frameRecoSrc = Frame(formReco)
    labRecoSrc = Label(frameRecoSrc, text='Source folder or file:')
    labRecoSrc.pack(side=LEFT)
    ui.entRecoSrc = Entry(frameRecoSrc)
    ui.entRecoSrc.pack(side=LEFT, fill=X, expand=True)
    buttRecoSrc = Button(frameRecoSrc, text='Browse...', command=partial(getRecoSrc, ui))
    buttRecoSrc.pack(side=LEFT)
    buttRecoRaw = Button(frameRecoSrc, text='Same as raw folder', command=partial(getRawFolder, ui))
    buttRecoRaw.pack(side=LEFT)

    # destination line

    frameRecoDest = Frame(formReco)
    labRecoDest = Label(frameRecoDest, text='Destination directory:')
    labRecoDest.pack(side=LEFT)
    ui.entRecoDest = Entry(frameRecoDest)
    ui.entRecoDest.pack(side=LEFT, fill=X, expand=True)
    buttRecoDestBrowse = Button(frameRecoDest, text='Browse...', command=partial(getRecoDest, ui))
    buttRecoDestBrowse.pack(side=LEFT)
    buttRecoDestDefault = Button(frameRecoDest, text='Use default', command=partial(getRecoDestDefault, ui))
    buttRecoDestDefault.pack(side=LEFT)




    formReco.pack(fill=X)
    bottReco.pack(side=BOTTOM)


def getRecoSrc(ui):

    pass


def getRawFolder(ui):

    pass


def getRecoDest(ui):

    pass


def getRecoDestDefault(ui):

    pass