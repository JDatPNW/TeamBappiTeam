# Data Acquisition Modules

## Webtoon combiner (webtoon_combiner.py)

This Python script combines seperate images obtained with [Webtoon-Downloader](https://github.com/Zehina/Webtoon-Downloader).<br>
We created a test dataset using [ComicPanelSegmentation](https://github.com/reidenong/ComicPanelSegmentation) on combine images.

### Prerequisites

- Python 3.x
- pillow
- numpy
- tqdm

### Usage

```bash
python webtoon_combiner.py --filepath <file_path> --outputpath <int:output_path> --start <int:start_episode> --end <end_episode> --all <True/False> --combine <int:Unit to combine imgs with> --remove <True/False>
```

- --filepath: Directory where imgs are saved by using Webtoon-Downloader.
- --outputpath: Directory where to save the combine imgs.
- --start: Episode to start combine.
- --end: Episode to end combine.
- --all: Whether to combine all.
- --combine: Unit to combine imgs with.
- --remove: Whether to remove the original imgs.


## YouTube Video Dataset Creator (yt_out.py)

This Python script captures frames from YouTube videos and creates a dataset of images. It supports both single video links and entire YouTube playlists.

### Prerequisites

- Python 3.x
- OpenCV
- VidGear
- Requests

Install the required libraries using:

```bash
pip install opencv-python vidgear requests
```

### Usage
Clone the repository or download the script. (see README in root folder of repository)

Run the script with the following command-line arguments:

```bash
python script_name.py --videolink <file_path> --destination <output_path> --quality <video_quality> --frameskip <frame_skip> --outputsize <resize_percentage> --showframe <True_or_False>
```

None of the arguments are necessary and will revert to the defaults if not specifically passed to the script!
- --videolink: Path to a file containing YouTube video URLs (default: './yt_input.txt').
- --destination: Target path to save images (default: './yt_out').
- --quality: Video quality, choose from 'best', 'worst', or a specific resolution (default: 'worst').
- --frameskip: Capture every n-th frame (default: 10).
- --outputsize: Resize percentage of the original frame (default: 1).
- --showframe: Display each frame during processing (default: False).

The script will process the specified video links, capture frames, and save them to the destination folder.

### Examples

#### Running the script
```bash
python script_name.py --videolink ./my_video_list.txt --destination ./output_folder --quality best --frameskip 5 --outputsize 0.8 --showframe True
```

#### Input File example

yt_input.txt
```
https://www.link.to.youtube.video.com
https://www.link.to.youtube.playlist.com
https://www.link.to.youtube.playlist.com
https://www.link.to.youtube.video.com
```

### Acknowledgments
- [OpenCV](https://opencv.org/)
- [VidGear](https://abhitronix.github.io/vidgear/)
- [Requests](https://docs.python-requests.org/en/latest/)
- [youtube-frame-capture](https://github.com/qaixerabbas/youtube-frame-capture)
- [YTPlaylistParser.py](https://gist.github.com/Axeltherabbit/5b147d508faf1b5cd735a52bd916b1e4)

Feel free to customize the script according to your needs!

