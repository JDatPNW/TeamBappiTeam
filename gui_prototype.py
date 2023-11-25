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
root.geometry("666x733")  # Setting the size of the main window to 666x700 pixels

def load_file():
    """
    The function `load_file()` opens a file dialog to select an image file, opens the selected image
    file using PIL, creates a CTkImage object from the PIL image, and configures a label to display the
    loaded image.
    """
    global my_image, label, status, PIL_Image  # Declaring global variables
    # Defining a tuple of file types for the file dialog
    filetypes = (
        ('image files', '*.png'),
        ('image files', '*.jpg'),
        ('image files', '*.jpeg'),
    )

    status.configure(text="Select Image from Dialog Window") # Status Message

    # Displaying the file dialog to select the image file
    filename = ctk.filedialog.askopenfilename(
        title='Open an image',
        initialdir='/',
        filetypes=filetypes
    )
    if not filename: # If no file was chosen, exit the function without opening anything
        status.configure(text="No Image selected - Try again!") # Status Message
        return

    status.configure(text="Image Loading...") # Status Message

    PIL_Image = Image.open(filename) # Opening the selected image file using PIL
    my_image = ctk.CTkImage(dark_image=PIL_Image, size=(666, 666)) # Creating a CTkImage object from the PIL image
    label.configure(image=my_image, text="") # Configuring the label to display the loaded image
    status.configure(text="Image Loaded! You can now run the model!") # Status Message

def save_file():
    """
    The function "save_file" saves an image as a JPEG file.
    """
    global my_image, label, status, PIL_Image  # Declaring global variables

    try: # check if image was already loaded
        my_image
    except NameError: # if not abort
        status.configure(text="No Image loaded yet! - Load one before saving!") # Status Message    
        return

    # Defining a tuple of file types for the file dialog
    filetypes = (
        ('image files', '*.jpeg'),
        ('image files', '*.jpg')
    )

    status.configure(text="Save Image via Dialog Window") # Status Message

    # Displaying the file dialog to save the image
    filename = ctk.filedialog.asksaveasfile(
        mode='w',
        defaultextension=".jpeg",
        title='Save the image',
        initialdir='/',
        filetypes=filetypes
)

    if not filename: # If no file was chosen, exit the function without opening anything
        status.configure(text="Failed Saving Image - Make sure filename location are correct and try again!") # Status Message
        return
    
    status.configure(text="Image Saving...") # Status Message

    my_image._dark_image.convert('RGB').save(filename)
    status.configure(text="Image Saved!") # Status Message


def run_model(): # TODO: implement model once it exists
    """
    The function `run_model()` converts the loaded image to grayscale,
    creates a CTkImage object from the grayscale image, and updates the
    label to display the grayscale version.
    """
    global my_image, label, status, PIL_Image  # Declaring global variables

    try: # check if image was already loaded
        my_image
    except NameError: # if not abort
        status.configure(text="No Image loaded yet! - Load one before running!")     # Status Message
        return

    status.configure(text="Image converting...") # Status Message   
    image = my_image._dark_image  # Accessing the dark image from the CTkImage object
    grey_image = image.convert('LA')  # Converting the image to grayscale
    grey_image_ctk = ctk.CTkImage(dark_image=grey_image, size=(666, 666))  # Creating a CTkImage object from the grayscale image
    my_image = grey_image_ctk
    label.configure(image=my_image, text="")  # Configuring the label to display the grayscale image
    status.configure(text="Image converted!") # Status Message  


"""
This code is creating a GUI (Graphical User Interface) for an image viewer application using the
customtkinter module.
"""
label = ctk.CTkLabel(root, width=666, height=666, text="Select Image")  # Creating a label widget for displaying images
label.grid(row=1, column=0, columnspan=4)  # Placing the label in the grid layout

status = ctk.CTkLabel(root, text="Select Image")  # Creating a label widget for displaying status messages
status.grid(row=2, column=0, columnspan=4)  # Placing the status message label in the grid layout

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
