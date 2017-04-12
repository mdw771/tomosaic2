#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Tkinter import *

class TomosaicUI(Frame):

    def __init__(self, parent):

        Frame.__init__(self, parent, background='white')
        self.parent = parent
        self.initUI()

    def initUI(self):

        self.parent.title('Tomosaic')

        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)

        # menubar
        fileFenu = Menu(menubar)
        fileFenu.add_command(label='Save parameters...')
        fileFenu.add_command(label='Exit', command=self.onExit)
        menubar.add_cascade(label='File', menu=fileFenu)



    def onExit(self):

        self.quit()

if __name__ == '__main__':

    root = Tk()
    root.geometry("250x150+300+300")
    app = TomosaicUI(root)
    root.mainloop()
