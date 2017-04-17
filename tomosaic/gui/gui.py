#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
import copy

from Tkinter import *
from ttk import Notebook
from tkFileDialog import *
from tkMessageBox import showerror, showwarning, showinfo

from meta_ui import *
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
        self.relative_shift = None
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
        fileFenu.add_command(label='Save parameters...', command=self.saveAllAttr)
        fileFenu.add_command(label='Load parameters...', command=self.loadAllAttr)
        fileFenu.add_command(label='Exit', command=self.onExit)
        menubar.add_cascade(label='File', menu=fileFenu)

        # ======================================================d
        # tabs

        tabFrame = Frame(root)
        tabs = Notebook(tabFrame)
        self.tabMeta = Frame(tabs)
        tabRegi = Frame(tabs)
        tabMerg = Frame(tabs)
        tabCent = Frame(tabs)
        tabReco = Frame(tabs)

        tabs.add(self.tabMeta, text='Metadata')
        tabs.add(tabRegi, text='Registration')
        tabs.add(tabMerg, text='Merging')
        tabs.add(tabCent, text='Center optimization')
        tabs.add(tabReco, text='Reconstruction')

        metatab_ui(self)

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
        self.boxRegiOut.insert(END, 'Registration\n--------------\n')
        self.boxRegiOut.pack()

        # button line

        buttLaunch = Button(bottRegi, text='Launch', command=self.launchRegistration)
        buttLaunch.pack(side=LEFT)
        buttRegiResave = Button(bottRegi, text='Resave shifts...', command=self.saveShifts)
        buttRegiResave.pack(side=LEFT)


        frameRegiMPI.pack(fill=X, expand=True)
        frameRegiOut.pack(fill=X, expand=True)
        frameShiftPath.pack(fill=X, expand=True)
        frameFindShift.pack(fill=X, expand=True)
        formRegi.pack()
        bottRegi.pack(side=BOTTOM)

        # ======================================================
        # merging tab

        formMerg = Frame(tabMerg)
        bottMerg = Frame(tabMerg)



        tabFrame.pack()
        tabs.pack()

    def saveAllAttr(self):

        dict = copy.copy(self.__dict__)
        for key in dict.keys():
            if key[:3] in ['box', 'ent', 'tab']:
                del dict[key]
            elif isinstance(dict[key], Entry) or isinstance(dict[key], Text):
                del dict[key]
            elif key[0] == '_':
                del dict[key]
            elif key in ['children', 'widgetName', 'master', 'parent', 'tk']:
                del dict[key]
            elif isinstance(dict[key], BooleanVar):
                dict[key] = dict[key].get()
        path = asksaveasfilename()
        if path is not None:
            np.save(path, dict)

    def loadAllAttr(self):

        path = askopenfilename()
        dict = np.load(path)
        dict = dict.item()
        write_pars(self, dict)
        readMeta(self)



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


    def launchRegistration(self):

        self.boxRegiOut.insert(END, 'Initiating registration...\n')
        self.mpi_ncore = int(self.entRegiNCore.get())
        self.boxRegiOut.insert(END, 'Refer to initial terminal window for intermediate output.\n')
        self.shiftgrid, self.relative_shift = find_shifts_mpi(self)
        self.boxRegiOut.insert(END, 'Done.\n')

    def readShifts(self):

        fname = self.entShiftPath.get()
        self.shiftgrid = read_shifts(self, fname)

    def saveShifts(self):

        self._savepath = asksaveasfilename()
        resave_shifts(self)

    def onExit(self):

        self.quit()

if __name__ == '__main__':

    root = Tk()
    # root.geometry("250x150+300+300")
    app = TomosaicUI(root)
    root.mainloop()
