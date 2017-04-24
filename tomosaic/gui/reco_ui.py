from functools import partial
import copy

from Tkinter import *
from ttk import Notebook
from tkFileDialog import *
from tkMessageBox import showerror, showwarning, showinfo

from metascripts import *
from recoscripts import *
from tomosaic.merge.merge import _get_algorithm_kwargs
from tomosaic.util.phase import _get_pr_kwargs


def recotab_ui(ui):

    formReco = Frame(ui.tabReco)
    bottReco = Frame(ui.tabReco)

    # type line

    frameRecoType = Frame(formReco)
    labRecoType = Label(frameRecoType, text='Operation type:')
    labRecoType.pack(side=LEFT)
    ui.varRecoType = StringVar()
    ui.varRecoType.set('dis')
    radRecoDis = Radiobutton(frameRecoType, variable=ui.varRecoType, text='Discrete files', value='dis')
    radRecoDis.pack(side=LEFT)
    radRecoSin = Radiobutton(frameRecoType, variable=ui.varRecoType, text='Single HDF5', value='sin')
    radRecoSin.pack(side=LEFT)

    # source line

    frameRecoSrc = Frame(formReco)
    labRecoSrc = Label(frameRecoSrc, text='Source folder or file:')
    labRecoSrc.pack(side=LEFT)
    ui.entRecoSrc = Entry(frameRecoSrc)
    ui.entRecoSrc.pack(side=LEFT, fill=X, expand=True)
    buttRecoSrc = Button(frameRecoSrc, text='Browse...', command=partial(getRecoSrc, ui))
    buttRecoSrc.pack(side=LEFT)
    buttRecoRaw = Button(frameRecoSrc, text='Same as raw folder', command=partial(getRawFolder, ui))
    buttRecoRaw.pack(side=LEFT)

    # destination line

    frameRecoDest = Frame(formReco)
    labRecoDest = Label(frameRecoDest, text='Destination directory:')
    labRecoDest.pack(side=LEFT)
    ui.entRecoDest = Entry(frameRecoDest)
    ui.entRecoDest.pack(side=LEFT, fill=X, expand=True)
    buttRecoDestBrowse = Button(frameRecoDest, text='Browse...', command=partial(getRecoDest, ui))
    buttRecoDestBrowse.pack(side=LEFT)
    buttRecoDestDefault = Button(frameRecoDest, text='Use default', command=partial(getRecoDestDefault, ui))
    buttRecoDestDefault.pack(side=LEFT)

    # center line

    frameRecoCent = Frame(formReco)
    labRecoCent = Label(frameRecoCent, text='Center vector location:')
    labRecoCent.pack(side=LEFT)
    ui.entRecoCent = Entry(frameRecoCent)
    ui.entRecoCent.pack(side=LEFT, fill=X, expand=True)
    buttRecoCentBrowse = Button(frameRecoCent, text='Browse...', command=partial(getRecoCent, ui))
    buttRecoCentBrowse.pack(side=LEFT)
    buttRecoCentDefault = Button(frameRecoCent, text='Use default', command=partial(getRecoCentDefault, ui))
    buttRecoCentDefault.pack(side=LEFT)

    # range line

    frameRecoRange = Frame(formReco)
    labelRecoRange = Label(frameRecoRange, text='Sino Range:')
    labelRecoRange.pack(side=LEFT)
    ui.entRecoStart = Entry(frameRecoRange)
    ui.entRecoStart.pack(side=LEFT)
    labelRecoHyphen = Label(frameRecoRange, text='-')
    labelRecoHyphen.pack(side=LEFT)
    ui.entRecoEnd = Entry(frameRecoRange)
    ui.entRecoEnd.pack(side=LEFT)
    labelRecoStep = Label(frameRecoRange, text='Step:')
    labelRecoStep.pack(side=LEFT)
    ui.entRecoStep = Entry(frameRecoRange)
    ui.entRecoStep.insert(0, '1')
    ui.entRecoStep.pack(side=LEFT)

    # algo line

    frameRecoAlgo = Frame(formReco)
    labRecoAlgo = Label(frameRecoAlgo, text='Algorithm:')
    labRecoAlgo.pack(side=LEFT)
    ui.varRecoAlgo = StringVar()
    ui.varRecoAlgo.set('gridrec')
    algo_opts = ('gridrec',)
    optRecoAlgo = OptionMenu(frameRecoAlgo, ui.varRecoAlgo, *algo_opts, command=partial(updateAlgoOpts, ui))
    optRecoAlgo.pack(side=LEFT, fill=X, expand=True)
    labRecoMode = Label(frameRecoAlgo, text='Mode:')
    labRecoMode.pack(side=LEFT)
    ui.varRecoMode = StringVar()
    ui.varRecoMode.set('180')
    mode_opts = ('180', '360')
    optRecoMode = OptionMenu(frameRecoAlgo, ui.varRecoMode, *mode_opts)
    optRecoMode.pack(side=LEFT)
    labRecoDs = Label(frameRecoAlgo, text=' Downsampling:')
    labRecoDs.pack(side=LEFT)
    ui.entRecoDs = Entry(frameRecoAlgo)
    ui.entRecoDs.insert(0, '1')
    ui.entRecoDs.pack(side=LEFT)

    # blending line

    frameRecoBlendMethod = Frame(formReco)
    labBlendMeth = Label(frameRecoBlendMethod, text='Blending method (dis. type only):')
    labBlendMeth.pack(side=LEFT)
    ui.varBlendMeth = StringVar()
    ui.varBlendMeth.set('max')
    lsBlendMeth = ('overlay', 'max', 'min', 'alpha', 'pyramid', 'poisson')
    ui.optBlendMeth = OptionMenu(frameRecoBlendMethod, ui.varBlendMeth, command=partial(updateBlendOpt, ui), *lsBlendMeth)
    ui.optBlendMeth.pack(side=LEFT, fill=X, expand=True)

    # blending opts line

    ui.frameBlendOpts = Frame(formReco)
    labBlendOpt = Label(ui.frameBlendOpts, text='Blending options: ')
    labBlendOpt.pack(side=LEFT)
    ui.frameBlendOptsInp = Frame(ui.frameBlendOpts)
    labBlendOptDefault = Label(ui.frameBlendOptsInp, text='Select a method.')
    labBlendOptDefault.pack(side=LEFT)
    ui.frameBlendOptsInp.pack(side=LEFT)

    # algo opts line

    frameRecoAlgoOpts = Frame(formReco)
    labRecoAlgoOpts = Label(frameRecoAlgoOpts, text='Algorithm options: ')
    labRecoAlgoOpts.pack(side=LEFT)
    ui.frameAlgoOptsField = Frame(frameRecoAlgoOpts)
    labAlgoOptsDefault = Label(ui.frameAlgoOptsField, text='Select an algorithm.')
    labAlgoOptsDefault.pack(side=LEFT)
    updateAlgoOpts(ui, ui.varRecoAlgo.get())
    ui.frameAlgoOptsField.pack(side=LEFT)

    # pr line

    frameRecoPr = Frame(formReco)
    labPr = Label(frameRecoPr, text='Phase retrieval')
    labPr.pack(side=LEFT)
    pr_opts = ('None', 'paganin')
    ui.varRecoPr = StringVar()
    ui.varRecoPr.set('None')
    optRecoPr = OptionMenu(frameRecoPr, ui.varRecoPr, command=partial(updatePrOpts, ui), *pr_opts)
    optRecoPr.pack(side=LEFT)
    labUnit = Label(frameRecoPr, text='Dimensions: length (cm), energy (keV)')
    labUnit.pack(side=LEFT)

    # pr options

    ui.frameRecoPrOpts = Frame(formReco)
    labPrOpts = Label(ui.frameRecoPrOpts, text='Phase retrieval options will be shown here if a method is selected.')
    labPrOpts.pack(side=LEFT)

    # chunk line

    frameRecoChunk = Frame(formReco)
    labChunk = Label(frameRecoChunk, text='Chunk size (set small if MPI used for sin type):')
    labChunk.pack(side=LEFT)
    ui.entRecoChunk = Entry(frameRecoChunk)
    ui.entRecoChunk.insert(0, '10')
    ui.entRecoChunk.pack(side=LEFT, fill=X, expand=True)

    # mpi line

    frameRecoMPI = Frame(formReco)
    labRecoMPI = Label(frameRecoMPI, text='Use MPI:')
    labRecoMPI.pack(side=LEFT)
    radMPIY = Radiobutton(frameRecoMPI, variable=ui.ifmpi, text='Yes', value=True)
    radMPIY.pack(side=LEFT)
    radMPIN = Radiobutton(frameRecoMPI, variable=ui.ifmpi, text='No', value=False)
    radMPIN.pack(side=LEFT, padx=10)
    labRecoNCore = Label(frameRecoMPI, text='Number of processes to initiate:')
    labRecoNCore.pack(side=LEFT)
    ui.entRecoNCore = Entry(frameRecoMPI)
    ui.entRecoNCore.insert(0, '5')
    ui.entRecoNCore.pack(side=LEFT, fill=X, expand=True)

    # out box line

    frameRecoOut = Frame(formReco, height=100)
    frameRecoOut.pack_propagate(False)
    ui.boxRecoOut = Text(frameRecoOut)
    ui.boxRecoOut.insert(END, 'Reconstruction\n')
    ui.boxRecoOut.insert(END, 'Refer to initial terminal window for intermediate output.\n--------------\n')
    ui.boxRecoOut.pack(side=LEFT)

    # button line

    buttLaunch = Button(bottReco, text='Launch', command=partial(launchRecon, ui))
    buttLaunch.pack(side=LEFT)
    buttRecoConfirm = Button(bottReco, text='Confirm parameters', command=partial(readRecoPars, ui))
    buttRecoConfirm.pack(side=LEFT)

    frameRecoType.pack(fill=X)
    frameRecoSrc.pack(fill=X)
    frameRecoDest.pack(fill=X)
    frameRecoCent.pack(fill=X)
    frameRecoRange.pack(fill=X)
    frameRecoBlendMethod.pack(fill=X)
    ui.frameBlendOpts.pack(fill=X)
    frameRecoAlgo.pack(fill=X)
    frameRecoAlgoOpts.pack(fill=X)
    frameRecoPr.pack(fill=X)
    ui.frameRecoPrOpts.pack(fill=X)
    frameRecoChunk.pack(fill=X)
    frameRecoMPI.pack(fill=X)
    frameRecoOut.pack(fill=X)

    formReco.pack(fill=X, expand=NO)
    bottReco.pack(side=BOTTOM)


def getRecoSrc(ui):

    if ui.varRecoType.get() == 'dis':
        src = askdirectory()
    elif ui.varRecoType.get() == 'sin':
        src = askopenfilename()
    ui.entRecoSrc.insert(0, src)


def getRawFolder(ui):

    try:
        ui.entRecoSrc.insert(0, ui.raw_folder)
    except:
        showerror(message='Raw folder must be specified in metadata tab.')


def getRecoDest(ui):

    src = askdirectory()
    ui.entRecoDest.insert(0, src)


def getRecoDestDefault(ui):

    ui.entRecoDest.insert(0, os.path.join(ui.raw_folder, 'recon'))


def getRecoCent(ui):

    src = askopenfilename()
    ui.entRecoCent.insert(0, src)


def getRecoCentDefault(ui):

    ui.entRecoCent.insert(0, os.path.join(ui.raw_folder, 'center_vec.txt'))


def updateAlgoOpts(ui, meth):

    for w in ui.frameAlgoOptsField.winfo_children():
        w.destroy()
    if meth == 'gridrec':
        ui.labAlgoOptsDefault = Label(ui.frameAlgoOptsField, text='Selected algorithm has no options available.')
        ui.labAlgoOptsDefault.pack(side=LEFT)


def updatePrOpts(ui, meth):

    for w in ui.frameRecoPrOpts.winfo_children():
        w.destroy()
    default_opts = _get_pr_kwargs()
    if meth == 'paganin':
        width = 10
        ui.lab1 = Label(ui.frameRecoPrOpts, text='Px size:')
        ui.lab1.grid(row=0, column=0)
        ui.ent1 = Entry(ui.frameRecoPrOpts)
        ui.ent1.insert(0, default_opts['pixel'])
        ui.ent1.grid(row=0, column=1)
        ui.lab2 = Label(ui.frameRecoPrOpts, text='Dist:')
        ui.lab2.grid(row=0, column=2)
        ui.ent2 = Entry(ui.frameRecoPrOpts)
        ui.ent2.insert(0, default_opts['distance'])
        ui.ent2.grid(row=0, column=3)
        ui.lab3 = Label(ui.frameRecoPrOpts, text='E:')
        ui.lab3.grid(row=0, column=4)
        ui.ent3 = Entry(ui.frameRecoPrOpts)
        ui.ent3.insert(0, default_opts['energy'])
        ui.ent3.grid(row=0, column=5)
        ui.lab4 = Label(ui.frameRecoPrOpts, text='Alpha:')
        ui.lab4.grid(row=0, column=6)
        ui.ent4 = Entry(ui.frameRecoPrOpts)
        ui.ent4.insert(0, default_opts['alpha_paganin'])
        ui.ent4.grid(row=0, column=7)
        ui.ent1['width'] = width
        ui.ent2['width'] = width
        ui.ent3['width'] = width
        ui.ent4['width'] = width

    else:
        ui.labNone = Label(ui.frameRecoPrOpts, text='No options available.')
        ui.labNone.pack(side=LEFT)


def updateBlendOpt(ui, meth):

    field = ui.frameBlendOptsInp

    ui.lstAlpha = [None, None]
    ui.lstDepth = [None, None]
    ui.lstBlur = [None, None]
    ui.lstOrder = [None, None]

    for w in field.winfo_children():
        w.destroy()

    default_opts = _get_algorithm_kwargs()

    if meth in ('max', 'min', 'poisson'):
        lab0 = Label(field, text='No options available for the selected method.')
        lab0.pack(side=LEFT)
    elif meth == 'overlay':
        lab0 = Label(field, text='Image on the top (1 or 2): ')
        lab0.pack(side=LEFT)
        ui.entOrder = Entry(field)
        ui.entOrder.insert(0, '2')
        ui.entOrder.pack(side=LEFT, fill=X)
    elif meth == 'alpha':
        lab0 = Label(field, text='Alpha: ')
        lab0.pack(side=LEFT)
        ui.entAlpha = Entry(field)
        ui.entAlpha.insert(0, default_opts['alpha'])
        ui.entAlpha.pack(side=LEFT, fill=X)
    elif meth == 'pyramid':
        lab0 = Label(field, text='Depth: ')
        lab0.pack(side=LEFT)
        ui.entDepth = Entry(field)
        ui.entDepth.insert(0, default_opts['depth'])
        ui.entDepth.pack(side=LEFT)
        lab1 = Label(field, text=' Blur: ')
        lab1.pack(side=LEFT)
        ui.entBlur = Entry(field)
        ui.entBlur.insert(0, default_opts['blur'])
        ui.entBlur.pack(side=LEFT)
    else:
        lab0 = Label(field, text='No options available for the selected method.')
        lab0.pack(side=LEFT)


def launchRecon(ui):

    readRecoPars(ui)
    recon_mpi(ui)
    ui.boxRecoOut.insert(END, 'Done.\n')


def buildBlendOpts(ui, meth, dict):

    if meth == 'overlay':
        dict['order'] = 1 if ui.entOrder.get() == 2 else 2
    elif meth == 'alpha':
        dict['alpha'] = float(ui.entAlpha.get())
    elif meth == 'pyramid':
        dict['depth'] = int(ui.entDepth.get())
        dict['blur'] = float(ui.entBlur.get())


def buildPrOpts(ui, meth, dict):

    if meth == 'paganin':
        dict['pixel_size'] = ui.ent1.get()
        dict['dist'] = ui.ent2.get()
        dict['energy'] = ui.ent3.get()
        dict['alpha'] = ui.ent4.get()


def readRecoPars(ui):

    ui.reco_type = ui.varRecoType.get()
    ui.reco_src = ui.entRecoSrc.get()
    ui.reco_dest = ui.entRecoDest.get()
    ui.reco_blend = ui.varBlendMeth.get()
    ui.reco_cent = ui.entRecoCent.get()
    ui.reco_start = int(ui.entRecoStart.get())
    ui.reco_end = int(ui.entRecoEnd.get())
    ui.reco_step = int(ui.entRecoStep.get())
    ui.reco_algo = ui.varRecoAlgo.get()
    ui.reco_mode = ui.varRecoMode.get()
    ui.reco_ds = ui.entRecoDs.get()
    ui.reco_pr = ui.varRecoPr.get()
    ui.reco_chunk = ui.entRecoChunk.get()
    if ui.reco_pr == 'None':
        ui.reco_pr = None
    ui.reco_mpi_ncore = ui.entRecoNCore.get()
    buildPrOpts(ui, ui.reco_pr, ui.reco_pr_opts)
    buildBlendOpts(ui, ui.reco_blend, ui.reco_blend_opts)
    ui.boxRecoOut.insert(END, 'Parameters read.\n')
