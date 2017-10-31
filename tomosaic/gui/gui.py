#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
import copy

from six.moves import tkinter
from six.moves import tkinter_ttk as ttk
from six.moves import tkinter_filedialog as tkFileDialog
from six.moves import tkinter_messagebox as tkMessageBox

from meta_ui import *
from regi_ui import *
from merg_ui import *
from phas_ui import *
from cent_ui import *
from reco_ui import *
from metascripts import *

from tkinter import *
from ttk import Notebook
from tkFileDialog import *
from tkMessageBox import showerror, showwarning, showinfo



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
        self.ifmpi = BooleanVar()
        self.ifmpi.set(True)
        self.mpi_ncore = 5
        self.merge_src = None
        self.merge_dest_folder = None
        self.merge_dest_fname = None
        self.merge_meth1 = None
        self.merge_meth2 = None
        self.merge_opts1 = {}
        self.merge_opts2 = {}
        self.merge_mpi_ncore = 1
        self.phas_src_folder = None
        self.phas_src_fname = None
        self.phas_dest_folder = None
        self.phas_dest_fname = None
        self.phas_meth = None
        self.phas_opts = {}
        self.phas_mpi_ncore = 1
        self.cent_type = None
        self.cent_src = None
        self.cent_dest = None
        self.cent_start = None
        self.cent_end = None
        self.cent_step = None
        self.cent_slice = None
        self.cent_algo = None
        self.cent_ds = 1
        self.cent_mpi_ncore = 1
        self.cent_mode = '180'
        self.reco_type = None
        self.reco_src = None
        self.reco_dest = None
        self.reco_cent = None
        self.reco_start = None
        self.reco_end = None
        self.reco_step = None
        self.reco_algo = None
        self.reco_mode = '180'
        self.reco_ds = 1
        self.reco_algo_opts = {}
        self.reco_pr = None
        self.reco_pr_opts = {}
        self.reco_chunk = 10
        self.reco_mpi_ncore = 1
        self.reco_blend = None
        self.reco_blend_opts = {}

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
        self.tabPhas = Frame(tabs)
        self.tabCent = Frame(tabs)
        self.tabReco = Frame(tabs)

        tabs.add(self.tabMeta, text='Metadata')
        tabs.add(self.tabRegi, text='Registration')
        tabs.add(self.tabMerg, text='Merging')
        tabs.add(self.tabPhas, text='Phase')
        tabs.add(self.tabCent, text='Center optimization')
        tabs.add(self.tabReco, text='Reconstruction')

        metatab_ui(self)
        regitab_ui(self)
        mergtab_ui(self)
        phastab_ui(self)
        centtab_ui(self)
        recotab_ui(self)

        tabFrame.pack()
        tabs.pack()

    def saveAllAttr(self):

        dict = copy.copy(self.__dict__)
        for key in dict.keys():
            if key[:3] in ['box', 'ent', 'tab', 'opt', 'lst', 'lab'] or key[:5] in ['frame']:
                del dict[key]
            elif isinstance(dict[key], Entry) or isinstance(dict[key], Text):
                del dict[key]
            elif key[0] == '_':
                del dict[key]
            elif key in ['children', 'widgetName', 'master', 'parent', 'tk']:
                del dict[key]
            elif isinstance(dict[key], BooleanVar) or isinstance(dict[key], StringVar):
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
