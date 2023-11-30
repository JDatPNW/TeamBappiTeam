"""
   ____ _   _ ___ 
  / ___| | | |_ _|
 | |  _| | | || | 
 | |_| | |_| || | 
  \____|\___/|___|
                  
This script uses the customtkinter module to create a simple image viewer GUI.
It allows users to load an image, convert it to the style that the tf model stores, and view the results.
The GUI includes buttons for loading an image, running a model, and exiting the application.
"""

# The code is importing the necessary modules for the program to work:
import customtkinter as ctk
from tkinter import *
from PIL import Image
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from keras.models import load_model
import numpy as np

# Needed to Load model (which uses this custom layer)
# Can be ignored in terms of this being a GUI
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(64, 3, padding="same")
        self.batch = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('relu')
        self.add = tf.keras.layers.Add()
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch(x)
        x = self.activation(x)
        x = self.add([inputs, x])
        return x    

"""
The code is setting the default color theme for the customtkinter module to "dark-blue". It then
creates a custom tkinter instance called `root` and sets the title of the main window to "Image
Viewer". Finally, it sets the size of the main window
"""
ctk.set_default_color_theme("dark-blue")  # Set the default color theme for customtkinter: "dark-blue"
root = ctk.CTk()  # Creating a custom tkinter instance
root.title("Image Viewer")  # Setting the title of the main window to "Image Viewer"
root.geometry("666x733")  # Setting the size of the main window

model = load_model('./model.h5', custom_objects={"ResBlock": ResBlock}, compile=False)

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
    my_image = ctk.CTkImage(dark_image=PIL_Image, size=(416, 666)) # Creating a CTkImage object from the PIL image
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
    The function `run_model()` takes the input image
    and converts it using the tf model that was loaded 
    """
    global my_image, label, status, PIL_Image, model  # Declaring global variables

    try: # check if image was already loaded
        my_image
    except NameError: # if not abort
        status.configure(text="No Image loaded yet! - Load one before running!")     # Status Message
        return

    status.configure(text="Image converting...") # Status Message   
    image = my_image._dark_image  # Accessing the dark image from the CTkImage object
   
    input = image.resize([80,128]) # Resizes image to the size required by the model
    input_array = np.array([input]) # Concerting it to an np.array, as this is needed for predicting
    predicted_image = model.predict((input_array - 127.5) / 127.5, verbose=0) # Get predicted image, based on input (normalized to -1 to 1 due to how model was trained), from model
    predicted_image_reshape = predicted_image.reshape(128,80,3) # reshape to correct size to get rid of batch dimension
    predicted_image_pil = Image.fromarray((predicted_image_reshape * 127.5 + 127.5).astype(np.uint8)) # Convert to PIL image (also go from normalized float to 255 int)
    predicted_image_ctk = ctk.CTkImage(dark_image=predicted_image_pil, size=(416, 666))  # Creating a CTkImage object from the predicted image
    my_image = predicted_image_ctk
    label.configure(image=my_image, text="")  # Configuring the label to display the predicted image
    status.configure(text="Image converted!") # Status Message  


"""
This code is creating a GUI (Graphical User Interface) for an image viewer application using the
customtkinter module.
"""
label = ctk.CTkLabel(root, width=416, height=666, text="Select Image")  # Creating a label widget for displaying images
label.grid(row=1, column=0, columnspan=4)  # Placing the label in the grid layout

status = ctk.CTkLabel(root, text="Select Image", width=666)  # Creating a label widget for displaying status messages
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
