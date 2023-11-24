import customtkinter as ctk
from customtkinter import E, S, W
from tkinter import *
from PIL import Image

root = ctk.CTk()

root.title("Image Viewer")

root.geometry("666x700")

my_image = ctk.CTkImage(dark_image=Image.open("Sample.jpg"),
                        size=(666, 666))

label = ctk.CTkLabel(root, image=my_image, text="")  # display image with a CTkLabel

label.grid(row=1, column=0, columnspan=3)

button_load = ctk.CTkButton(root, text="Load")

button_exit = ctk.CTkButton(root, text="Exit",
					command=root.quit)

button_run = ctk.CTkButton(root, text="Run")

button_load.grid(row=5, column=0)
button_exit.grid(row=5, column=1)
button_run.grid(row=5, column=2)

root.mainloop()
