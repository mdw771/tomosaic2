#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Tkinter import *
from ttk import Notebook
from pars import *

# from tomosaic.merge import *
# from tomosaic.misc import *
# from tomosaic.register import *
# from tomosaic.util import *


class TomosaicUI(Frame):

    def __init__(self, parent):

        Frame.__init__(self, parent, background='white')
        self.parent = parent
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

        # ======================================================
        # tab

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

        formMeta = Frame(tabMeta)
        bottMeta = Frame(tabMeta)

        # path line
        framePath = Frame(formMeta)
        labRawPath = Label(framePath, text='Data path:').pack(side=LEFT)
        self.entRawPath = Entry(framePath).pack(side=LEFT, fill=X, expand=True)
        buttRawBrowse = Button(framePath, text='Browse...').pack(side=LEFT)
        labDiv = Label(framePath, text='--------------------')

        # prefix line
        labPrefix = Label(formMeta, text='Prefix:').grid(row=1, column=0, sticky=W)
        self.entPrefix = Entry(formMeta).grid(row=1, column=1, columnspan=3, sticky=W+E)

        # shift line
        labRoughY = Label(formMeta, text='Estimated shift Y:')
        labRoughY.grid(row=2, column=0, sticky=W)
        self.entRoughY = Entry(formMeta)
        self.entRoughY.grid(row=2, column=1)
        labRoughX = Label(formMeta, text='X:')
        labRoughX.grid(row=2, column=2, sticky=W)
        self.entRoughX = Entry(formMeta)
        self.entRoughX.grid(row=2, column=3)

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

    def readMeta(self):

        prefix = self.entPrefix.get()

    def onExit(self):

        self.quit()

if __name__ == '__main__':

    root = Tk()
    # root.geometry("250x150+300+300")
    app = TomosaicUI(root)
    root.mainloop()
