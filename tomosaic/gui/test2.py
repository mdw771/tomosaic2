from Tkinter import *
from ttk import *

root = Tk()
note = Notebook(root)

tab2 = Frame(note)
tab3 = Frame(note)

note.add(tab2, text = "Tab Two")
note.add(tab3, text = "Tab Three")
note.pack()
root.mainloop()
exit()