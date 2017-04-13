#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Tkinter import *
from ttk import Notebook
from tkFileDialog import *
from tkMessageBox import showerror, showwarning, showinfo

from metascripts import *


class TomosaicUI(Frame):

    def __init__(self, parent):

        Frame.__init__(self, parent, background='white')
        self.parent = parent
        self.initUI()

        self.raw_folder = None
        self.prefix = None
        self.y_shift = None
        self.x_shift = None
        self.filelist = None

    def initUI(self):

        self.parent.title('Tomosaic')

        # ======================================================
        # menubar

        menuFrame = Frame(self.parent)
        menubar = Menu(menuFrame)
        self.parent.config(menu=menubar)

        fileFenu = Menu(menubar)
        fileFenu.add_command(label='Save parameters...')
        fileFenu.add_command(label='Exit', command=self.onExit)
        menubar.add_cascade(label='File', menu=fileFenu)

        # ======================================================
        # tabs

        tabFrame = Frame(root)
        tabs = Notebook(tabFrame)
        tabMeta = Frame(tabs)
        tabRegi = Frame(tabs)
        tabMerg = Frame(tabs)
        tabReco = Frame(tabs)

        tabs.add(tabMeta, text='Metadata')
        tabs.add(tabRegi, text='Registration')
        tabs.add(tabMerg, text='Merging')
        tabs.add(tabReco, text='Reconstruction')

        # ======================================================
        # metadata tab

        rowPrefix = 2
        rowFirstFrame = 3
        rowShift = 4

        formMeta = Frame(tabMeta)
        bottMeta = Frame(tabMeta)

        # path line
        framePath = Frame(formMeta)
        labRawPath = Label(framePath, text='Data path:')
        labRawPath.pack(side=LEFT)
        self.entRawPath = Entry(framePath)
        self.entRawPath.pack(side=LEFT, fill=X, expand=True)
        buttRawBrowse = Button(framePath, text='Browse...', command=self.getRawDirectory)
        buttRawBrowse.pack(side=LEFT)

        # prefix line
        labPrefix = Label(formMeta, text='Prefix:')
        labPrefix.grid(row=rowPrefix, column=0, sticky=W)
        self.entPrefix = Entry(formMeta)
        self.entPrefix.grid(row=rowPrefix, column=1, columnspan=3, sticky=W+E)

        # writer line
        buttFirstFrame = Button(formMeta, text='Write first projection frame for all files',
                                command=self.writeFirstFrames)
        buttFirstFrame.grid(row=rowFirstFrame, columnspan=4)

        # shift line
        labRoughY = Label(formMeta, text='Estimated shift Y:')
        labRoughY.grid(row=rowShift, column=0, sticky=W)
        self.entRoughY = Entry(formMeta)
        self.entRoughY.grid(row=rowShift, column=1)
        labRoughX = Label(formMeta, text='X:')
        labRoughX.grid(row=rowShift, column=2, sticky=W)
        self.entRoughX = Entry(formMeta)
        self.entRoughX.grid(row=rowShift, column=3)

        # confirm button line
        buttMetaSave = Button(bottMeta, text='Save all parameters...')
        buttMetaSave.grid(row=0, column=0, sticky=W+E)
        buttMetaConfirm = Button(bottMeta, text='Confirm', command=self.readMeta)
        buttMetaConfirm.grid(row=0, column=1, sticky=W+E)

        framePath.grid(row=0, column=0, columnspan=4, sticky=W+E)
        formMeta.pack()
        bottMeta.pack(side=BOTTOM)

        # ======================================================

        tabFrame.pack()
        tabs.pack()

    def getRawDirectory(self):

        self.raw_folder = askdirectory()
        self.entRawPath.insert(0, raw_folder)

    def writeFirstFrames(self):

        self.readMeta()
        if self.raw_folder is None or self.prefix is None:
            showerror(message='Data path and prefix must be filled. ')
        else:
            write_first_frames()

    def readMeta(self):

        self.raw_folder = self.entRawPath.get()
        self.prefix = self.entPrefix.get()
        print raw_folder
        print raw_folder is None
        print raw_folder == ''
        if self.raw_folder is not None and prefix is not None:
            self.filelist = get_filelist()

    def onExit(self):

        self.quit()

if __name__ == '__main__':

    root = Tk()
    # root.geometry("250x150+300+300")
    app = TomosaicUI(root)
    root.mainloop()
