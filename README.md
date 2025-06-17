# OJO - We give eyes to the blind

### About:
This is a simple CV model designed to take camera input,
detect objects visible to the camera and generate audio descriptions of the scenery.
This script's application involves helping blind people "see" their surroundings by listening to it.

### Dependencies:
To get started, install the following Python packages:

- opencv-python (for image processing)
- torch (PyTorch, required for YOLO)
- numpy (for numerical operations)
- ultralytics (for YOLOv8 model)
- pyttsx3 (text-to-speech module)

### Installation:
Run the following command to install all dependencies:
```
pip install opencv-python torch numpy ultralytics pyttsx3
```
Ensure you have Python 3.8+ installed before running the script.

### Running the script:
clone this repo (or copy paste the python file `CameraFunction.py`)
and then run 
```sh
python run CameraFunction.py
```
from the folder where the file is located

### Note about the `HEADLESS` flag:
I am using a `HEADLESS` flag (line 6 in `CameraFunction.py`) to conditionally enable
headless mode (no display), and it's disabled for testing purpose. If you want to run this on a
Raspberry PI or any device without a monitor,
just set the boolean to `True` and you will hear just the audio.

### Note about exiting the script:
Due to the blocking behavior of pyttsx3's text-to-speech engine, stopping the script cleanly can be tricky.
To exit:
  Spam `Ctrl + C` in the terminal window where the script is running.
If you find a cleaner alternative to quit the script, feel free to contribute!
### Credits:
Author - Roshan M R (rossmaxx)

Co-Authors 
- Aswin Ganga - (v1.0 - For AIthon 2.0 AI hackathon)
- Team RAYS - Mini project
    - Adithya K V 
    - Sneha K
    - Yuktha Prakash K
