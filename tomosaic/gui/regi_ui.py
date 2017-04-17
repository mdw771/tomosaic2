from functools import partial
import copy

from Tkinter import *
from ttk import Notebook
from tkFileDialog import *
from tkMessageBox import showerror, showwarning, showinfo

from metascripts import *
from regiscripts import *


def regitab_ui(ui):
    
    formRegi = Frame(ui.tabRegi)
    bottRegi = Frame(ui.tabRegi)
    
    # read shifts line
    
    frameShiftPath = Frame(formRegi)
    labRegiUp = Label(frameShiftPath, text='Read existing shift datafile...')
    labRegiUp.pack()
    labShiftPath = Label(frameShiftPath, text='File location:')
    labShiftPath.pack(side=LEFT)
    ui.entShiftPath = Entry(frameShiftPath)
    ui.entShiftPath.pack(side=LEFT, fill=X, expand=True)
    buttShiftPath = Button(frameShiftPath, text='Browse...', command=partial(getShiftFilePath, ui))
    buttShiftPath.pack(side=LEFT)
    buttShiftPath = Button(frameShiftPath, text='Use default', command=partial(getDefaultShiftPath, ui))
    buttShiftPath.pack(side=LEFT)
    buttShiftRead = Button(frameShiftPath, text='Read', command=partial(readShifts, ui))
    buttShiftRead.pack(side=LEFT)
    
    # find shifts line
    
    frameFindShift = Frame(formRegi)
    labRegiDown = Label(frameFindShift, text='Or compute shifts...')
    labRegiDown.pack()
    
    # MPI choice line
    
    frameRegiMPI = Frame(frameFindShift)
    labRegiMPI = Label(frameRegiMPI, text='Use MPI:')
    labRegiMPI.pack(side=LEFT)
    ui.ifmpi = BooleanVar()
    radMPIY = Radiobutton(frameRegiMPI, variable=ui.ifmpi, text='Yes', value=True)
    radMPIY.pack(side=LEFT)
    radMPIN = Radiobutton(frameRegiMPI, variable=ui.ifmpi, text='No', value=False)
    radMPIN.pack(side=LEFT, padx=10)
    labRegiNCore = Label(frameRegiMPI, text='Number of processes to initiate:')
    labRegiNCore.pack(side=LEFT)
    ui.entRegiNCore = Entry(frameRegiMPI)
    ui.entRegiNCore.pack(side=LEFT, fill=X, expand=True)
    
    # out box line
    
    frameRegiOut = Frame(frameFindShift)
    ui.boxRegiOut = Text(frameRegiOut)
    ui.boxRegiOut.insert(END, 'Registration\n--------------\n')
    ui.boxRegiOut.pack()
    
    # button line
    
    buttLaunch = Button(bottRegi, text='Launch', command=partial(launchRegistration, ui))
    buttLaunch.pack(side=LEFT)
    buttRegiResave = Button(bottRegi, text='Resave shifts...', command=partial(saveShifts, ui))
    buttRegiResave.pack(side=LEFT)
    
    frameRegiMPI.pack(fill=X, expand=True)
    frameRegiOut.pack(fill=X, expand=True)
    frameShiftPath.pack(fill=X, expand=True)
    frameFindShift.pack(fill=X, expand=True)
    formRegi.pack()
    bottRegi.pack(side=BOTTOM)


def getShiftFilePath(ui):

    ui.shift_path = askopenfilename()
    ui.entShiftPath.insert(0, ui.shift_path)
    
    
def getDefaultShiftPath(ui):

    try:
        ui.entShiftPath.insert(0, os.path.join(ui.raw_folder, 'shifts.txt'))
    except:
        showerror(message='Raw folder is not specified.')


def launchRegistration(ui):

    ui.boxRegiOut.insert(END, 'Initiating registration...\n')
    ui.mpi_ncore = int(ui.entRegiNCore.get())
    ui.boxRegiOut.insert(END, 'Refer to initial terminal window for intermediate output.\n')
    ui.shiftgrid, ui.relative_shift = find_shifts_mpi(ui)
    ui.boxRegiOut.insert(END, 'Done.\n')


def readShifts(ui):

    fname = ui.entShiftPath.get()
    ui.shiftgrid = read_shifts(ui, fname)


def saveShifts(ui):

    ui._savepath = asksaveasfilename()
    resave_shifts(ui)