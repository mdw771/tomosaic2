#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial

from Tkinter import *
from ttk import Notebook
from tkFileDialog import *
from tkMessageBox import showerror, showwarning, showinfo

from metascripts import *
from regiscripts import *


class TomosaicUI(Frame):

    def __init__(self, parent):

        Frame.__init__(self, parent, background='white')
        self.parent = parent

        self.raw_folder = None
        self.prefix = None
        self.y_shift = None
        self.x_shift = None
        self.filelist = None
        self.filegrid = None
        self.shiftgrid = None
        self.shift_path = None
        self.mpi_ncore = 1

        self.initUI()

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

        # ======================================================d
        # tabs

        tabFrame = Frame(root)
        tabs = Notebook(tabFrame)
        tabMeta = Frame(tabs)
        tabRegi = Frame(tabs)
        tabMerg = Frame(tabs)
        tabCent = Frame(tabs)
        tabReco = Frame(tabs)

        tabs.add(tabMeta, text='Metadata')
        tabs.add(tabRegi, text='Registration')
        tabs.add(tabMerg, text='Merging')
        tabs.add(tabCent, text='Center optimization')
        tabs.add(tabReco, text='Reconstruction')

        # ======================================================
        # metadata tab

        rowPrefix = 2
        rowFirstFrame = 3
        rowShift = 4
        rowOutbox = 5

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

        # outbox line
        self.boxMetaOut = Text(formMeta)
        self.boxMetaOut.insert(END, 'Tomosaic GUI (Beta)\n--------------\n')
        self.boxMetaOut.grid(row=rowOutbox, column=0, rowspan=4, columnspan=4, sticky=N+S+W+E)

        # confirm button line
        buttMetaSave = Button(bottMeta, text='Save all parameters...')
        buttMetaSave.grid(row=0, column=0, sticky=W+E)
        buttMetaConfirm = Button(bottMeta, text='Confirm', command=self.readMeta)
        buttMetaConfirm.grid(row=0, column=1, sticky=W+E)

        framePath.grid(row=0, column=0, columnspan=4, sticky=W+E)
        formMeta.pack()
        bottMeta.pack(side=BOTTOM)

        # ======================================================
        # registration tab

        formRegi = Frame(tabRegi)
        bottRegi = Frame(tabRegi)

        # read shifts line

        frameShiftPath = Frame(formRegi)
        labRegiUp = Label(frameShiftPath, text='Read existing shift datafile...')
        labRegiUp.pack()
        labShiftPath = Label(frameShiftPath, text='File location:')
        labShiftPath.pack(side=LEFT)
        self.entShiftPath = Entry(frameShiftPath)
        self.entShiftPath.pack(side=LEFT, fill=X, expand=True)
        buttShiftPath = Button(frameShiftPath, text='Browse...', command=self.getShiftFilePath)
        buttShiftPath.pack(side=LEFT)
        buttShiftPath = Button(frameShiftPath, text='Use default', command=self.getDefaultShiftPath)
        buttShiftPath.pack(side=LEFT)
        buttShiftRead = Button(frameShiftPath, text='Read', command=self.readShifts)
        buttShiftRead.pack(side=LEFT)

        # find shifts line

        frameFindShift = Frame(formRegi)
        labRegiDown = Label(frameFindShift, text='Or compute shifts...')
        labRegiDown.pack()

        # MPI choice line

        frameRegiMPI = Frame(frameFindShift)
        labRegiMPI = Label(frameRegiMPI, text='Use MPI:')
        labRegiMPI.pack(side=LEFT)
        self.ifmpi = BooleanVar()
        radMPIY = Radiobutton(frameRegiMPI, variable=self.ifmpi, text='Yes', value=True)
        radMPIY.pack(side=LEFT)
        radMPIN = Radiobutton(frameRegiMPI, variable=self.ifmpi, text='No', value=False)
        radMPIN.pack(side=LEFT, padx=10)
        labRegiNCore = Label(frameRegiMPI, text='Number of processes to initiate:')
        labRegiNCore.pack(side=LEFT)
        self.entRegiNCore = Entry(frameRegiMPI)
        self.entRegiNCore.pack(side=LEFT, fill=X, expand=True)

        # out box line

        frameRegiOut = Frame(frameFindShift)
        self.boxRegiOut = Text(frameRegiOut)
        self.boxRegiOut.pack()

        # button line

        buttLaunch = Button(bottRegi, text='Launch', command=self.launchRegistration)
        buttLaunch.pack(side=LEFT)
        buttRegiResave = Button(bottRegi, text='Resave shifts...')
        buttRegiResave.pack(side=LEFT)


        frameRegiMPI.pack(fill=X, expand=True)
        frameRegiOut.pack(fill=X, expand=True)
        frameShiftPath.pack(fill=X, expand=True)
        frameFindShift.pack(fill=X, expand=True)
        formRegi.pack()
        bottRegi.pack(side=BOTTOM)

        # ======================================================

        tabFrame.pack()
        tabs.pack()

    def getRawDirectory(self):

        self.raw_folder = askdirectory()
        self.entRawPath.insert(0, self.raw_folder)

    def getDirectory(self, var):

        var = askdirectory()

    def getShiftFilePath(self):

        self.shift_path = askopenfilename()
        self.entShiftPath.insert(0, self.shift_path)

    def getFilePath(self, var):

        var = askopenfilename()

    def getDefaultShiftPath(self):

        try:
            self.entShiftPath.insert(0, os.path.join(self.raw_folder, 'shifts.txt'))
        except:
            showerror(message='Raw folder is not specified.')

    def writeFirstFrames(self):

        self.raw_folder = self.entRawPath.get()
        self.prefix = self.entPrefix.get()
        if self.raw_folder is not '' and self.prefix is not '':
            self.filelist = get_filelist(self)
            self.boxMetaOut.insert(END, 'Writing first frames...\n')
            write_first_frames(self)
        else:
            showerror(message='Data path and prefix must be filled. ')

    def readMeta(self):

        self.raw_folder = self.entRawPath.get()
        self.prefix = self.entPrefix.get()
        if self.raw_folder is not '' and self.prefix is not '':
            self.filelist = get_filelist(self)
            self.filegrid = get_filegrid(self)
        try:
            self.y_shift = float(self.entRoughY.get())
            self.x_shift = float(self.entRoughX.get())
            self.shiftgrid = get_rough_shiftgrid(self)
        except:
            showerror(message='Estimated shifts must be numbers.')
        self.boxMetaOut.insert(END, '--------------\n')
        self.boxMetaOut.insert(END, 'Metadata logged:\n')
        self.boxMetaOut.insert(END, 'Raw folder: {:s}\n'.format(self.raw_folder))
        self.boxMetaOut.insert(END, 'Prefix: {:s}\n'.format(self.prefix))
        self.boxMetaOut.insert(END, 'Estimated shift: ({:.2f}, {:.2f})\n'.format(self.y_shift, self.x_shift))
        self.boxMetaOut.insert(END, 'File/shift grid established.\n')
        self.boxMetaOut.insert(END, '--------------\n')

    def launchRegistration(self):

        self.mpi_ncore = int(self.entRegiNCore.get())
        find_shifts_mpi(self)

    def readShifts(self):

        self.shiftgrid = read_shifts(self)

    def beginRegistration(self):

        pass

    def onExit(self):

        self.quit()

if __name__ == '__main__':

    root = Tk()
    # root.geometry("250x150+300+300")
    app = TomosaicUI(root)
    root.mainloop()
