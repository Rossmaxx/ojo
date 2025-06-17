import cv2 
import pyttsx3

from ultralytics import YOLO

# to compile in headless mode (Global flag)
HEADLESS = False

def detect_objects(yolo_model, image_tensor):
    results = yolo_model(image_tensor, conf=0.6)
    detections = results[0].boxes.data.cpu().numpy()  # Convert tensor to NumPy
    class_names = yolo_model.names  # Class names from the model
    return detections, class_names


def detections_to_tts(detections, class_names, frame_width, frame_height):
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
        tts_engine.say(speech_text)
        tts_engine.runAndWait()

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

def open_camera(yolo_model):
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        print("Error, video device failed to open")
        return
    
    while True:
        ret, frame = vid.read()

        if not HEADLESS:
            cv2.imshow('frame', frame.copy())  # Show before processing
        
        detections, class_names = detect_objects(yolo_model, frame)

        if not HEADLESS:
            # Draw bounding boxes directly on 'frame'
            draw_boxes(frame, detections, class_names)

        frame_height, frame_width = frame.shape[:2]  # Extract frame dimensions
        detections_to_tts(detections, class_names, frame_width, frame_height)

        if not HEADLESS:
            cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # text-to-speech
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 180)  # Adjust rate as needed

    tts_engine.say("Please wait, YOLO is loading for initialisation")
    tts_engine.runAndWait()
    
    # yolo initialisation
    yolo_model = YOLO('yolov8n.pt')

    open_camera(yolo_model)
