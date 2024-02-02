import cv2  # Importing the OpenCV library for computer vision
from vidgear.gears import CamGear  # Importing CamGear for video capturing
import time  # Importing the time module for measuring execution time
import argparse  # Importing argparse for parsing command-line arguments
import requests  # Importing requests for making HTTP requests
import re  # Importing re for regular expressions
import os # Importing os to access the current directory
import numpy as np

# Parsing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--videolink", required=False, type=str, help="File containing YouTube Video URLs", default='./yt_input.txt' )
parser.add_argument("--destination", required=False, type=str, help="Target path to save images", default='../data')
parser.add_argument("--quality", required=False, type=str, help="Select either best or worst or any of the common YouTube qualities e.g. 720p", default="worst")
parser.add_argument("--frameskip", required=False, type=int, help="Only captures every n-th frame", default=10)
parser.add_argument("--outputsize", required=False, type=float, help="Percentage (0.0-1.0) of original size to which the images should be rescaled", default=1)
parser.add_argument("--showframe", required=False, type=bool, help="Should Frame be displayed?", default=False)
parser.add_argument("--useai", required=False, type=bool, help="Should AI model be used to help sort images?", default=False)
parser.add_argument("--aithreshold", required=False, type=float, help="How convinced should the model be the image is good? Lower is more generous, higher is more strict.", default=0.75)
parser.add_argument("--aisaveboth", required=False, type=bool, help="Should bad images be saved too (into a seperate folder)?", default=True)
args = parser.parse_args()

# Extracting values from command-line arguments
file_path = args.videolink
path = args.destination
quality = args.quality
frameskip = args.frameskip
outputsize = args.outputsize
show_frame = args.showframe
use_ai = args.useai
ai_threshold = args.aithreshold
ai_save_both = args.aisaveboth

if use_ai:
    import tensorflow as tf
    model = tf.keras.models.load_model("./ai_helper.h5")

#Create dir if it does not already exist
if not os.path.exists(path):
    os.makedirs(path)
    if use_ai and ai_save_both:
        os.makedirs(path + "/bad/")

# Reading video links from the specified file
with open(file_path, 'r') as file:
    linklist = [line.strip() for line in file.readlines()]

# Processing each video link
for link in linklist:

    if link.startswith("https://www.youtube.com/playlist?list="):
        # Handling YouTube playlist links
        page_text = requests.get(link).text

        # Extracting video links from the playlist
        parser = re.compile(r"watch\?v=\S+?list=")
        playlist = set(re.findall(parser, page_text))
        playlist = map(
            (lambda x: "https://www.youtube.com/" + x.replace("\\u0026list=", "")), playlist
        )
    else:
        playlist = [link]

    # Processing each video in the playlist
    for video in playlist:

        source = video

        time_start = time.time()

        options = {"STREAM_RESOLUTION": quality}

        # Starting the video stream
        stream = CamGear(
            source=source,
            stream_mode=True,
            time_delay=0.5,
            logging=True,
            **options
        ).start()

        currentframe = 0
        while True:
            # Reading a frame from the video stream
            frame = stream.read()

            if frame is None:
                break
            if show_frame:
                cv2.imshow("Output Frame", frame)

            if currentframe % frameskip == 0:
                # Creating and saving the image with metadata
                name = path + "/" + stream.ytv_metadata["title"] + "_frame" + str(currentframe) + ".jpg"
                print("Creating..." + name)
                if outputsize != 1:
                    frame = cv2.resize(frame, (0, 0), fx=outputsize, fy=outputsize)
                if use_ai: # entered if using AI helper
                    frame_resized = cv2.resize(frame, (213, 120)) # resize to ai model input size
                    frame_resized_gs = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY) # make greyscale
                    frame_resized_gs_expanded = np.expand_dims(frame_resized_gs, axis=-1) # add dim to end
                    input_frame = np.expand_dims(frame_resized_gs_expanded, axis=0) #  # add dim to beginning
                    prediction = model.predict(input_frame) # index 0 should be bad, index 1 should be good
                    print(f"----AI helper: Frame is {round(100*prediction[0][0], 2)}% bad and {round(100*prediction[0][1], 2)}% good.")
                    if prediction[0][1] >= ai_threshold: # check if images is good
                        cv2.imwrite(name, frame)
                        print("--------Saved image as good!")
                    elif ai_save_both: # if also saving bad ones
                        name = path + "/bad/" + stream.ytv_metadata["title"] + "_frame" + str(currentframe) + ".jpg"
                        cv2.imwrite(name, frame)
                        print("--------Saved image as bad!")
                else:
                    cv2.imwrite(name, frame)
                    print("----Saved image!")

            currentframe += 1

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        # Closing OpenCV windows and stopping the video stream
        cv2.destroyAllWindows()
        stream.stop()

        time_end = time.time()

        time_taken = round(time_end - time_start, 3)
        print("=======================================================")
        print("-------------------------------------------------------")
        print(f"## The time taken to create dataset: {time_taken} seconds ##")
        print("-------------------------------------------------------")
        print("=======================================================")
