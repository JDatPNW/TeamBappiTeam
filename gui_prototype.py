"""
   ____ _   _ ___ 
  / ___| | | |_ _|
 | |  _| | | || | 
 | |_| | |_| || | 
  \____|\___/|___|
                  
This script uses the customtkinter module to create a simple image viewer GUI.
It allows users to load an image, convert it to grayscale, and view the results.
The GUI includes buttons for loading an image, running a model, and exiting the application.
"""

# The code is importing the necessary modules for the program to work:
import customtkinter as ctk  # Importing customtkinter module as ctk
from tkinter import *  # Importing everything from the tkinter module
from PIL import Image  # Importing the Image module from the PIL library
"""
The code is setting the default color theme for the customtkinter module to "dark-blue". It then
creates a custom tkinter instance called `root` and sets the title of the main window to "Image
Viewer". Finally, it sets the size of the main window to 666x700 pixels.
"""
ctk.set_default_color_theme("dark-blue")  # Set the default color theme for customtkinter: "dark-blue"
root = ctk.CTk()  # Creating a custom tkinter instance
root.title("Image Viewer")  # Setting the title of the main window to "Image Viewer"
root.geometry("666x700")  # Setting the size of the main window to 666x700 pixels

def load_file():
    """
    The function `load_file()` opens a file dialog to select an image file, opens the selected image
    file using PIL, creates a CTkImage object from the PIL image, and configures a label to display the
    loaded image.
    """
    global my_image, label, PIL_Image  # Declaring global variables
    # Defining a tuple of file types for the file dialog
    filetypes = (
        ('text files', '*.png'),
        ('text files', '*.jpg'),
        ('All files', '*.*')
    )
    # Displaying the file dialog to select a file
    filename = ctk.filedialog.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes
    )
    PIL_Image = Image.open(filename) # Opening the selected image file using PIL
    my_image = ctk.CTkImage(dark_image=PIL_Image, size=(666, 666)) # Creating a CTkImage object from the PIL image
    label.configure(image=my_image, text="") # Configuring the label to display the loaded image

def save_file():
    global my_image, label, PIL_Image  # Declaring global variables
    filename = ctk.filedialog.asksaveasfile(mode='w', defaultextension=".jpeg")
    if not filename:
        return
    my_image._dark_image.convert('RGB').save(filename)

def run_model(): # TODO: implement model once it exists
    """
    The function `run_model()` converts the loaded image to grayscale,
    creates a CTkImage object from the grayscale image, and updates the
    label to display the grayscale version.
    """
    global my_image, label, PIL_Image  # Declaring global variables
    image = my_image._dark_image  # Accessing the dark image from the CTkImage object
    grey_image = image.convert('LA')  # Converting the image to grayscale
    grey_image_ctk = ctk.CTkImage(dark_image=grey_image, size=(666, 666))  # Creating a CTkImage object from the grayscale image
    my_image = grey_image_ctk
    label.configure(image=my_image, text="")  # Configuring the label to display the grayscale image

"""
This code is creating a GUI (Graphical User Interface) for an image viewer application using the
customtkinter module.
"""
label = ctk.CTkLabel(root, width=666, height=666, text="Select Image")  # Creating a label widget for displaying images
label.grid(row=1, column=0, columnspan=4)  # Placing the label in the grid layout

# Creating buttons for loading, exiting, and running the model
button_load = ctk.CTkButton(root, command=load_file, text="Load", fg_color="#2B719E", hover_color="#1B4864") # Load, Blue
button_save = ctk.CTkButton(root, command=save_file, text="Save", fg_color="#2B719E", hover_color="#1B4864") # Load, Blue
button_exit = ctk.CTkButton(root, text="Exit", fg_color="#9E2B2B", hover_color="#641B1B", command=root.quit) # Exit, Red
button_run = ctk.CTkButton(root, text="Run", command=run_model, fg_color="#239E8B", hover_color="#10473E") # Run, Green

# Placing the buttons in the grid layout
button_exit.grid(row=5, column=0, pady=3) # Left
button_load.grid(row=5, column=1, pady=3) # Center Left
button_save.grid(row=5, column=2, pady=3) # Center Right
button_run.grid(row=5, column=3, pady=3) # Right

root.mainloop()  # Running the main event loop to display the GUI
