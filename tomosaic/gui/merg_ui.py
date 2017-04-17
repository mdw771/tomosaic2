from functools import partial
import copy

from Tkinter import *
from ttk import Notebook
from tkFileDialog import *
from tkMessageBox import showerror, showwarning, showinfo

from metascripts import *
from regiscripts import *
from mergscripts import *


def mergtab_ui(ui):

    formMerg = Frame(ui.tabMerg)
    bottMerg = Frame(ui.tabMerg)