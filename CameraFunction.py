import cv2
import pyttsx3
import tkinter as tk
from tkinter import Label, Listbox, Button
from PIL import Image, ImageTk
from ultralytics import YOLO

# Initialize YOLO model
yolo_model = YOLO('yolov8m.pt')

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 180)

# Global variables
vid = None
previous_labels = {}
detected_objects = []
camera_open = False  # Track camera state

# Tkinter GUI setup
window = tk.Tk()
window.title("Object Detection")
window.geometry("900x600")  # Set predefined size to prevent UI lag

# Placeholder black image to maintain UI size
placeholder_img = ImageTk.PhotoImage(Image.new("RGB", (640, 480), "black"))

label = Label(window, image=placeholder_img)
label.pack()

# Listbox for displaying detected objects
object_listbox = Listbox(window, height=10, width=50)
object_listbox.pack()

# Function to detect objects
def detect_objects(yolo_model, image_tensor):
    results = yolo_model(image_tensor, conf=0.6)
    detections = results[0].boxes.data.cpu().numpy()  # Convert tensor to NumPy
    class_names = yolo_model.names  # Class names from the model
    return detections, class_names

# Function to process detections and announce them
def batch_and_process_descriptions(detections, class_names, frame_width, frame_height):
    detected_objects.clear()
    speech_text = ""

    for i, detection in enumerate(detections):
        x1, y1, x2, y2, _, class_id = detection[:6]
        class_name = class_names[int(class_id)]
        detected_objects.append(class_name)

        position = get_relative_position(x1, y1, x2, y2, frame_width, frame_height)
        speech_text += f"There is a {class_name} at {position}. " if i == 0 else f"And a {class_name} at {position}. "

    update_listbox()
    if speech_text:
        tts_engine.say(speech_text)
        tts_engine.runAndWait()

# Function to determine relative position
def get_relative_position(x1, y1, x2, y2, frame_width, frame_height):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    horizontal = "left" if center_x < frame_width * 0.33 else "right" if center_x > frame_width * 0.66 else "center"
    vertical = "top" if center_y < frame_height * 0.33 else "bottom" if center_y > frame_height * 0.66 else "middle"

    return f"{horizontal} {vertical}"

# Function to draw bounding boxes
def draw_boxes(image, detections, class_names):
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        class_name = class_names[int(class_id)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name}: {confidence:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Function to process each frame
def process_frame(frame):
    detections, class_names = detect_objects(yolo_model, frame)
    draw_boxes(frame, detections, class_names)
    batch_and_process_descriptions(detections, class_names, frame.shape[1], frame.shape[0])

# Function to update the camera feed
def update_frame():
    if camera_open:
        ret, frame = vid.read()
        if ret:
            process_frame(frame)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img)
            label.config(image=img_tk)
            label.image = img_tk
        window.after(10, update_frame)

# Function to update Listbox
def update_listbox():
    object_listbox.delete(0, tk.END)
    for obj in detected_objects:
        object_listbox.insert(tk.END, obj)

# Function to toggle camera on/off
def toggle_camera():
    global vid, camera_open
    if camera_open:
        camera_open = False
        toggle_button.config(text="Open Camera")
        vid.release()
        cv2.destroyAllWindows()
        label.config(image=placeholder_img)
    else:
        vid = cv2.VideoCapture(0)
        if not vid.isOpened():
            print("Error: Unable to open camera")
            return
        camera_open = True
        toggle_button.config(text="Close Camera")
        update_frame()

# Toggle button to open/close camera
toggle_button = Button(window, text="Open Camera", command=toggle_camera, height=2, width=15)
toggle_button.pack()

window.mainloop()
