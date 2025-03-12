import cv2
import pyttsx3
import tkinter as tk
from tkinter import Label, Listbox
from PIL import Image, ImageTk

from ultralytics import YOLO


def detect_objects(yolo_model, image_tensor):
    results = yolo_model(image_tensor, conf=0.6)
    detections = results[0].boxes.data.cpu().numpy()  # Convert tensor to NumPy
    class_names = yolo_model.names  # Class names from the model
    return detections, class_names


def batch_and_process_descriptions(detections, class_names, frame_width, frame_height):
    description_batch = []
    speech_text = ""
    detected_objects.clear()  # Clear previous detections for frontend listbox

    for i, detection in enumerate(detections):
        x1, y1, x2, y2, _, class_id = detection[:6]
        class_name = class_names[int(class_id)]
        description = f"Label: {class_name}, Position: ({(x2+x1)/2:.2f}, {(y2+y1)/2:.2f})"
        description_batch.append(description)

        # Append detected object to frontend listbox
        detected_objects.append(class_name)

        position = get_relative_position(x1, y1, x2, y2, frame_width, frame_height)
        if i == 0:
            speech_text += f"There is a {class_name} at {position} "
        else:
            speech_text += f"and a {class_name} at {position} "

    tts_engine.say(speech_text)
    tts_engine.runAndWait()

    update_listbox()  # Update the frontend listbox with detected objects
    return " ".join(description_batch)


def get_relative_position(x1, y1, x2, y2, frame_width, frame_height):
    """Determine relative position (left, center, right and top, middle, bottom)"""
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    horizontal_position = "left" if center_x < frame_width * 0.33 else "right" if center_x > frame_width * 0.66 else "center"
    vertical_position = "top" if center_y < frame_height * 0.33 else "bottom" if center_y > frame_height * 0.66 else "middle"

    return f"{horizontal_position} {vertical_position}"


def draw_boxes(image, detections, class_names):
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        class_name = class_names[int(class_id)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name}: {confidence:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def process_frame(frame, detections, class_names, previous_labels, frame_width, frame_height):
    current_labels = {}
    
    draw_boxes(frame, detections, class_names)

    for detection in detections:
        class_id = detection[5]
        class_name = class_names[int(class_id)]
        current_labels[class_name] = class_name

    batch_and_process_descriptions(detections, class_names, frame_width, frame_height)

    previous_labels.clear()
    previous_labels.update(current_labels)


def update_frame():
    ret, frame = vid.read()
    if ret:
        img_pil, class_names = detect_objects(yolo_model, frame)
        process_frame(frame, img_pil, class_names, previous_labels, frame.shape[1], frame.shape[0])
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(img)

        label.config(image=img_tk)
        label.image = img_tk

    window.after(10, update_frame)  # Refresh frame every 10ms


def update_listbox():
    """Update the frontend Listbox with detected objects."""
    object_listbox.delete(0, tk.END)  # Clear old items
    for obj in detected_objects:
        object_listbox.insert(tk.END, obj)  # Add new detections


def openCameraFunction(yolo_model, previous_labels):
    global vid  

    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        print("Error, video device failed to open")
        return

    update_frame()
    window.mainloop()

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 180)

    tts_engine.say("Please wait, the model is starting")

    # YOLO model initialization
    yolo_model = YOLO('yolov8m.pt')

    previous_labels = {}
    detected_objects = []  # Store detected objects for frontend display

    # Tkinter GUI setup
    window = tk.Tk()
    window.title("Object Detection")

    label = Label(window)
    label.pack()

    # Listbox for displaying detected objects
    object_listbox = Listbox(window, height=10, width=50)
    object_listbox.pack()

    openCameraFunction(yolo_model, previous_labels)
