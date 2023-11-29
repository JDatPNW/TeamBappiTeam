import os
from PIL import Image, ImageOps
import numpy as np
import argparse

# Set up command-line argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--inputpath", required=False, type=str, help="Folder containing image data", default='../data/')
parser.add_argument("--destination", required=False, type=str, help="Target path to save modified images", default='../mod_data/')
parser.add_argument("--threshold", required=False, type=str, help="Threshold for black and white conversion", default=20)
parser.add_argument("--cropx", required=False, type=str, help="Target width for cropping", default=64)
parser.add_argument("--cropy", required=False, type=str, help="Target height for cropping", default=96)
parser.add_argument("--height_modifier", required=False, type=str, help="Should Frame be displayed?", default=1.5)
args = parser.parse_args()

# Extracting values from command-line arguments
input_path = args.inputpath  # Path to the folder containing original image data
output_path = args.destination  # Target path to save modified images
threshold = args.threshold  # Threshold for black and white conversion
cropx = args.cropx  # Target width for cropping
cropy = args.cropy  # Target height for cropping
height_modifier = args.height_modifier  # Modifier for adjusting the new height during resizing

# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Iterate through files in the input path
for i, file in enumerate(os.listdir(input_path)):
    # Check if the file is a JPG or PNG image
    if file.endswith(".jpg") or file.endswith(".png"):
        # Construct the full path to the image
        location = os.path.join(input_path, file)

        # Open the image using the PIL library
        image = Image.open(location)

        # Convert the image to a NumPy array for processing
        for_white = np.array(image)

        # Create a binary mask for pixels below a certain threshold - to filter out black borders
        mask = for_white < (255 - threshold)
        mask = mask.any(2)
        mask0, mask1 = mask.any(0), mask.any(1)

        # Apply the mask to the original image to get a cropped image
        cropped_image = for_white[np.ix_(mask1, mask0)]
        del image, for_white, mask, mask0, mask1

        # Convert the cropped NumPy array back to a PIL image
        for_black = Image.fromarray(cropped_image)

        # Find non-zero elements in the black and white image - to filter out black borders
        y_nonzero, x_nonzero, _ = np.nonzero(np.array(for_black) > threshold)

        # Crop the image based on the non-zero elements
        cropped_image_again = for_black.crop(
            (np.min(x_nonzero), np.min(y_nonzero), np.max(x_nonzero), np.max(y_nonzero))
        )
        del cropped_image, for_black, x_nonzero, y_nonzero

        # Resize the cropped image to a specified height while maintaining the aspect ratio
        width, height = cropped_image_again.size
        new_height = int(cropy * height_modifier)
        new_width = (new_height * width) // height
        resized_image = cropped_image_again.resize((new_width, new_height), Image.LANCZOS)
        del cropped_image_again

        # Convert the resized image back to a NumPy array
        for_rescale = np.array(resized_image)
        y, x, _ = for_rescale.shape

        # Crop the resized image to the target dimensions
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        output_image = for_rescale[starty: starty + cropy, startx: startx + cropx]
        del for_rescale

        # Convert the output NumPy array back to a PIL image
        for_save = Image.fromarray(output_image)
        del output_image

        # Check if the final image has the correct dimensions
        s_width, s_height = for_save.size
        if s_width == cropx and s_height == cropy:
            # Save the image and its mirrored version
            print(f"Saving Image {i} ({file}) as {output_path}image{i}.jpg")
            for_save.save(f"{output_path}image{i}.jpg")

            # Image gets inverted here to double dataset
            mirror_image = ImageOps.mirror(for_save)
            print(f"Saving Image {i} ({file}) mirrored as {output_path}image{i}_mirror.jpg")
            mirror_image.save(f"{output_path}image{i}_mirror.jpg")
            del for_save, mirror_image
        else:
            # Print a message if the final image has incorrect dimensions
            print(f"WRONG SIZE! ({s_width},{s_height})")
    else:
        # Continue to the next iteration if the file is not a JPG or PNG image
        continue
