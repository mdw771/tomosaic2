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
from regi_ui import *
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
        self.tabRegi = Frame(tabs)
        self.tabMerg = Frame(tabs)
        self.tabCent = Frame(tabs)
        self.tabReco = Frame(tabs)

        tabs.add(self.tabMeta, text='Metadata')
        tabs.add(self.tabRegi, text='Registration')
        tabs.add(self.tabMerg, text='Merging')
        tabs.add(self.tabCent, text='Center optimization')
        tabs.add(self.tabReco, text='Reconstruction')

        metatab_ui(self)

        regitab_ui(self)

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

    def getFilePath(self, var):

        var = askopenfilename()

    def onExit(self):

        self.quit()

if __name__ == '__main__':

    root = Tk()
    # root.geometry("250x150+300+300")
    app = TomosaicUI(root)
    root.mainloop()
