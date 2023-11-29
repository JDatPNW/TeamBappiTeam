# Image Processing Script
## Overview
This script processes a collection of images by applying various transformations, including cropping, resizing, and mirroring. The script is designed to be run from the command line, allowing users to customize parameters such as input and output paths, thresholds, and target dimensions.

## Requirements
- Python 3.x
- PIL (Python Imaging Library)
- NumPy

## Usage
Install the required dependencies:

```bash
pip install pillow numpy
```
Run the script from the command line with optional arguments:

```bash
python data_mod.py.py --inputpath [path/to/input/folder] --destination [path/to/output/folder] --threshold [threshold_value] --cropx [target_width] --cropy [target_height] --height_modifier [height_modifier_value]
```
## Command-line Arguments
- --inputpath: Path to the folder containing the original image data. Default is ../data/.
- --destination: Target path to save modified images. Default is ../mod_data/.
- --threshold: Threshold for black and white conversion. Default is 20.
- --cropx: Target width for cropping. Default is 64.
- --cropy: Target height for cropping. Default is 96.
- --height_modifier: Modifier for adjusting the new height during resizing. Default is 1.5.

NOTE: The height modifier is tricky and can cause issues if not used correctly, play with this if too much data is of the wrong dimensions and is not saved

## Example
``` bash
python data_mod.py.py --inputpath ../sample_images/ --destination ../output_images/ --threshold 30 --cropx 80 --cropy 120 --height_modifier 2.0
```
#### In this example:

- Input images are located in the '../sample_images/' folder.
- Processed images will be saved in the '../output_images/' folder.
- The threshold for black and white conversion is set to 30.
- The target width for cropping is 80.
- The target height for cropping is 120.
- The height modifier during resizing is set to 2.0.


## Output
The script processes each image in the input folder according to the specified parameters and saves the results in the output folder. The modified images are saved with file names in the format image{i}.jpg and image{i}_mirror.jpg.

