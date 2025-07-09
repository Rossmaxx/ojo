import cv2
import numpy as np 
import pyttsx3

from os import kill, getpid
from signal import signal, SIGINT
from sys import exit

from ultralytics import YOLO

# to compile in headless mode (Global flag)
HEADLESS = False


def detect_objects(yolo_model, frame):
    results = yolo_model(frame, conf=0.6)
    detections = results[0].boxes.data.cpu().numpy()  # Convert tensor to NumPy
    class_names = yolo_model.names  # Class names from the model
    return detections, class_names


def detections_to_text(detections, class_names, frame_width, frame_height):
    speech_text = ""
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, _, class_id = detection[:6]
        class_name = class_names[int(class_id)]

        position = get_relative_position(x1, y1, x2, y2, frame_width, frame_height)
        # generate template text for narration
        if i == 0:
            speech_text += f"There is a {class_name} at {position} "
        else:
            speech_text += f" and a {class_name} at {position} "

    # say the template text
    if speech_text:
        return speech_text
    
    return ""


def get_relative_position(x1, y1, x2, y2, frame_width, frame_height):
    """Determine relative position (left, center, right and top, middle, bottom)"""
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Determine horizontal position
    if center_x < frame_width * 0.33:
        horizontal_position = "left"
    elif center_x > frame_width * 0.66:
        horizontal_position = "right"
    else:
        horizontal_position = "center"

    # Determine vertical position
    if center_y < frame_height * 0.33:
        vertical_position = "top"
    elif center_y > frame_height * 0.66:
        vertical_position = "bottom"
    else:
        vertical_position = "middle"

    return f"{horizontal_position} {vertical_position}"


def draw_boxes(image, detections, class_names):
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        class_name = class_names[int(class_id)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name}: {confidence:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

def compute_depth_map(left_gray, right_gray):
    # Create StereoBM matcher
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    disparity = stereo.compute(left_gray, right_gray)
    disp_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    return disp_normalized.astype(np.uint8)


def signal_handler(sig, frame):
    print("Interrupt received. Cleaning up...")
    tts_engine.stop()
    cam_left.release()
    cam_right.release()
    cv2.destroyAllWindows()
    exit(0)


def speak_out(text, tts_engine):
    if not text.strip():
        return
    tts_engine.say(text)
    tts_engine.runAndWait()


if __name__ == "__main__":
    # to allow graceful exit
    signal(SIGINT, signal_handler)

    # text-to-speech
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 180)  # Adjust rate as needed

    speak_out("Please wait, YOLO is loading for initialisation", tts_engine)
    yolo_model = YOLO('yolov8n.pt')

    # camera working
    cam_left = cv2.VideoCapture(1)
    cam_right = cv2.VideoCapture(2)
    if not cam_left.isOpened() or not cam_right.isOpened():
        error_msg = "Error, video device failed to open"
        print(error_msg)
        speak_out(error_msg, tts_engine)
        exit(1)
    
    while True:
        ret_left, frame_left = cam_left.read()
        ret_right, frame_right = cam_right.read()
        if not ret_left or not ret_right or frame_left is None or frame_right is None:
            print("Warning: Failed to read from camera.")
            continue

        # Convert to grayscale for depth estimation
        gray_l = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        depth_map = compute_depth_map(gray_l, gray_r)
        
        detections, class_names = detect_objects(yolo_model, frame_left)

        if not HEADLESS:
            # Draw bounding boxes directly on 'frame'
            draw_boxes(frame_left, detections, class_names)
            cv2.imshow('Left Camera (Detection)', frame_left)
            cv2.imshow('Right Camera', frame_right)
            cv2.imshow('Depth Map', depth_map)

        frame_height, frame_width = frame_left.shape[:2]  # Extract frame dimensions
        speech_text = detections_to_text(detections, class_names, frame_width, frame_height)
        
        speak_out(speech_text, tts_engine)

        # fixme - not working due to lack of threading
        if cv2.waitKey(1) & 0xFF == ord('q'):
            signal_handler(SIGINT, None)
