from functools import partial
import copy

from Tkinter import *
from ttk import Notebook
from tkFileDialog import *
from tkMessageBox import showerror, showwarning, showinfo

from metascripts import *
from regiscripts import *


def metatab_ui(ui):
    
    # ======================================================
    # metadata tab

    rowPrefix = 2
    rowFirstFrame = 3
    rowShift = 4
    rowOutbox = 5

    formMeta = Frame(ui.tabMeta)
    bottMeta = Frame(ui.tabMeta)

    # path line
    framePath = Frame(formMeta)
    labRawPath = Label(framePath, text='Data path:')
    labRawPath.pack(side=LEFT)
    ui.entRawPath = Entry(framePath)
    ui.entRawPath.pack(side=LEFT, fill=X, expand=True)
    buttRawBrowse = Button(framePath, text='Browse...', command=partial(getRawDirectory, ui))
    buttRawBrowse.pack(side=LEFT)

    # prefix line
    labPrefix = Label(formMeta, text='Prefix:')
    labPrefix.grid(row=rowPrefix, column=0, sticky=W)
    ui.entPrefix = Entry(formMeta)
    ui.entPrefix.grid(row=rowPrefix, column=1, columnspan=3, sticky=W+E)

    # writer line
    buttFirstFrame = Button(formMeta, text='Write first projection frame for all files',
                            command=partial(writeFirstFrames, ui))
    buttFirstFrame.grid(row=rowFirstFrame, columnspan=4)

    # shift line
    labRoughY = Label(formMeta, text='Estimated shift Y:')
    labRoughY.grid(row=rowShift, column=0, sticky=W)
    ui.entRoughY = Entry(formMeta)
    ui.entRoughY.grid(row=rowShift, column=1)
    labRoughX = Label(formMeta, text='X:')
    labRoughX.grid(row=rowShift, column=2, sticky=W)
    ui.entRoughX = Entry(formMeta)
    ui.entRoughX.grid(row=rowShift, column=3)

    # outbox line
    ui.boxMetaOut = Text(formMeta)
    ui.boxMetaOut.insert(END, 'Tomosaic GUI (Beta)\n--------------\n')
    ui.boxMetaOut.grid(row=rowOutbox, column=0, rowspan=4, columnspan=4, sticky=N + S + W + E)

    # confirm button line
    buttMetaSave = Button(bottMeta, text='Save all parameters...', command=ui.saveAllAttr)
    buttMetaSave.grid(row=0, column=0, sticky=W+E)
    buttMetaConfirm = Button(bottMeta, text='Confirm', command=partial(readMeta, ui))
    buttMetaConfirm.grid(row=0, column=1, sticky=W+E)

    framePath.grid(row=0, column=0, columnspan=4, sticky=W+E)
    formMeta.pack()
    bottMeta.pack(side=BOTTOM)
    
    
def getRawDirectory(ui):

    ui.raw_folder = askdirectory()
    ui.entRawPath.delete(0, END)
    ui.entRawPath.insert(0, ui.raw_folder)
    

def writeFirstFrames(ui):

    ui.raw_folder = ui.entRawPath.get()
    ui.prefix = ui.entPrefix.get()
    if ui.raw_folder is not '' and ui.prefix is not '':
        ui.filelist = get_filelist(ui)
        ui.boxMetaOut.insert(END, 'Writing first frames...\n')
        write_first_frames(ui)
    else:
        showerror(message='Data path and prefix must be filled. ')


def readMeta(ui):

    ui.raw_folder = ui.entRawPath.get()
    ui.prefix = ui.entPrefix.get()
    if ui.raw_folder is not '' and ui.prefix is not '':
        try:
            ui.filelist = get_filelist(ui)
            ui.filegrid = get_filegrid(ui)
        except:
            pass
    try:
        ui.y_shift = float(ui.entRoughY.get())
        ui.x_shift = float(ui.entRoughX.get())
        ui.shiftgrid = get_rough_shiftgrid(ui)
    except:
        showerror(message='Estimated shifts must be numbers and file path must be valid.')
    ui.boxMetaOut.insert(END, '--------------\n')
    ui.boxMetaOut.insert(END, 'Metadata logged:\n')
    ui.boxMetaOut.insert(END, 'Raw folder: {:s}\n'.format(ui.raw_folder))
    ui.boxMetaOut.insert(END, 'Prefix: {:s}\n'.format(ui.prefix))
    ui.boxMetaOut.insert(END, 'Estimated shift: ({:.2f}, {:.2f})\n'.format(ui.y_shift, ui.x_shift))
    ui.boxMetaOut.insert(END, 'File/shift grid established.\n')
    ui.boxMetaOut.insert(END, '--------------\n')