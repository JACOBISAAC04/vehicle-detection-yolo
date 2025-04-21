# Vehicle-Detection-YOLO

This repository contains a project that uses the YOLOv8 model for detecting vehicles (cars, trucks, etc.) in images and videos. The project includes pre-trained models (linked externally), scripts for real-time video detection, and output examples.

## Project Structure

```
vehicle-detection-yolo/
├── data/
│   ├── videos/
│   └── models/
├── src/
│   ├── video_detection.py
├── output/
│   ├── images/
│   ├── videos/
│   └── gifs/
├── notebooks/
│   ├── exploration.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── check_model_classes.py
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JACOBISAAC04/vehicle-detection-yolo.git
   cd vehicle-detection-yolo
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## YOLOv8 Model File

This project uses the YOLOv8x model from Ultralytics.

🔗 [Download yolov8x.pt from the official source](https://github.com/ultralytics/assets/releases/latest)

After downloading, place the file in:

```
data/models/yolov8x.pt
```

## Usage

### Video Detection

Run the following command to detect vehicles in a video and generate a GIF:
```bash
python src/video_detection.py --video data/videos/input_video.mp4 --output_video output/videos/output_video.mp4 --output_gif output/gifs/output_video.gif
```

## Example Output

Here is an example output of the video detection saved as a GIF:

![Example Output](output/gifs/output_video.gif)

## Project Components

### data/
Contains input videos, and the external model file.

### src/
Includes Python scripts for video vehicle detection using YOLOv8.

### output/
Stores the processed outputs — video results, and animated GIFs.

### notebooks/
Exploratory notebooks used for testing and development.
