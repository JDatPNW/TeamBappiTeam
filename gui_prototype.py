import customtkinter as ctk
from customtkinter import E, S, W
from tkinter import *
from PIL import Image, ImageTk

ctk.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

root = ctk.CTk()

root.title("Image Viewer")

root.geometry("666x700")

def load_file():
    global my_image, label, PIL_Image

    filetypes = (
        ('text files', '*.png'),
        ('text files', '*.jpg'),
        ('All files', '*.*')
    )

    filename = ctk.filedialog.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    PIL_Image = Image.open(filename)
    my_image = ctk.CTkImage(dark_image=PIL_Image, size=(666, 666))
    label.configure(image=my_image, text="")  # display image with a CTkLabel

def run_model():
    global my_image, label, PIL_Image
    image = my_image._dark_image
    grey_image = image.convert('LA')
    grey_image_ctk = ctk.CTkImage(dark_image=grey_image, size=(666, 666))
    label.configure(image=grey_image_ctk, text="")  # display image with a CTkLabel


label = ctk.CTkLabel(root, width=666, height=666, text="Select Image")  # display image with a CTkLabel

label.grid(row=1, column=0, columnspan=3)

button_load = ctk.CTkButton(root, command=load_file, text="Load", fg_color="#2B719E", hover_color="#1B4864")

button_exit = ctk.CTkButton(root, text="Exit", fg_color="#9E2B2B", hover_color="#641B1B",
					command=root.quit)

button_run = ctk.CTkButton(root, text="Run", command=run_model, fg_color="#239E8B", hover_color="#10473E")

button_load.grid(row=5, column=0, pady=3)
button_exit.grid(row=5, column=1)
button_run.grid(row=5, column=2)

root.mainloop()
