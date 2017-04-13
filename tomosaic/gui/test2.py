from Tkinter import *

root = Tk()

Label(root, text="First").grid(row=0)
Label(root, text="Second").grid(row=1)

e1 = Entry(root)
e2 = Entry(root)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)


mainloop()