from functools import partial
import copy

from Tkinter import *
from ttk import Notebook
from tkFileDialog import *
from tkMessageBox import showerror, showwarning, showinfo

from metascripts import *
from regiscripts import *


def metatab_ui(ui):

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
    framePrefix = Frame(formMeta)
    labPrefix = Label(framePrefix, text='Prefix:')
    labPrefix.pack(side=LEFT)
    ui.entPrefix = Entry(framePrefix)
    ui.entPrefix.pack(side=LEFT, fill=X, expand=True)

    # writer line
    frameWriter = Frame(formMeta)
    buttFirstFrame = Button(frameWriter, text='Write first projection frame for all files',
                            command=partial(writeFirstFrames, ui))
    buttFirstFrame.pack(side=LEFT, fill=X, expand=True)

    # shift line
    frameShift = Frame(formMeta)
    labRoughY = Label(frameShift, text='Estimated shift Y:')
    labRoughY.pack(side=LEFT)
    ui.entRoughY = Entry(frameShift)
    ui.entRoughY.pack(side=LEFT, fill=X, expand=True)
    labRoughX = Label(frameShift, text='X:')
    labRoughX.pack(side=LEFT)
    ui.entRoughX = Entry(frameShift)
    ui.entRoughX.pack(side=LEFT, fill=X, expand=True)

    # outbox line
    frameOutMeta = Frame(formMeta, height=285)
    frameOutMeta.pack_propagate(False)
    ui.boxMetaOut = Text(frameOutMeta)
    ui.boxMetaOut.insert(END, 'Tomosaic GUI (Beta)\n--------------\n')
    ui.boxMetaOut.pack(side=LEFT, fill=BOTH, expand=True)

    # confirm button line
    buttMetaSave = Button(bottMeta, text='Save all parameters...', command=ui.saveAllAttr)
    buttMetaSave.grid(row=0, column=0, sticky=W+E)
    buttMetaConfirm = Button(bottMeta, text='Confirm', command=partial(readMeta, ui))
    buttMetaConfirm.grid(row=0, column=1, sticky=W+E)

    framePath.pack(fill=X)
    framePrefix.pack(fill=X)
    frameWriter.pack(fill=X)
    frameShift.pack(fill=X)
    frameOutMeta.pack(fill=X)
    formMeta.pack(fill=BOTH, expand=True)
    bottMeta.pack(side=BOTTOM)
    
    
def getRawDirectory(ui):

    ui.raw_folder = askdirectory()
    try:
        ui.entRawPath.delete(0, END)
        ui.entRawPath.insert(0, ui.raw_folder)
    except:
        pass
    

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