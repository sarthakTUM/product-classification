import io
import base64
import tkinter as tk
from urllib.request import urlopen
from PIL import Image, ImageTk
import urllib
from io import BytesIO
import similar_images_AE as sae


root = tk.Tk()
global panel
global check
global orgpanel
check = 1
panel = []

root.geometry('450x450+500+300')
root.title('Product Recommender System')
root.columnconfigure(0, pad=3)
root.columnconfigure(1, pad=3)
root.columnconfigure(2, pad=3)
root.columnconfigure(3, pad=3)

root.rowconfigure(0, pad=3)
root.rowconfigure(1, pad=3)
root.rowconfigure(2, pad=3)
root.rowconfigure(3, pad=3)
root.rowconfigure(4, pad=3)
# scrollbar = tk.Scrollbar(root)
# scrollbar.pack(side='right', fill='y')


asinTxt = tk.StringVar()
mlabel = tk.Label(root, text='Enter Asin').grid(column=10, row=2)
asinEntry = tk.Entry(root, textvariable=asinTxt).grid(column=10, row=4)


# panel = Panel(root, yscrollcommand=scrollbar.set)
# panel.pack(side="top", padx=10, pady=10)
def recommendSearch():
    global check
    global panel
    global orgpanel

    # del panel[:]
    results = sae.main(asinTxt.get())

    orgimg = ImageTk.PhotoImage(Image.open(results))
    if check == 1:
        orgpanel = tk.Label(root, image=orgimg)
        orgpanel.grid(row=8, column=0, columnspan=50, rowspan=20, padx=5, pady=5)
        orgpanel.image = orgimg
    else:
        orgpanel.configure(image=orgimg)
        orgpanel.image = orgimg


    check = check + 1


searchBtn = tk.Button(root, text="Search", command=recommendSearch).grid(column=10, row=6)

root.mainloop()
