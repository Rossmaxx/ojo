import cv2
import pyttsx3
import tkinter as tk
from tkinter import Label, Listbox, Button
from PIL import Image, ImageTk
from ultralytics import YOLO


def detect_objects(yolo_model, image_tensor):
    results = yolo_model(image_tensor, conf=0.6)
    detections = results[0].boxes.data.cpu().numpy()
    class_names = yolo_model.names
    return detections, class_names

def batch_and_process_descriptions(detections, class_names, frame_width, frame_height, tts_engine, detected_objects):
    detected_objects.clear()
    speech_text = ""
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, _, class_id = detection[:6]
        class_name = class_names[int(class_id)]
        detected_objects.append(class_name)
        position = get_relative_position(x1, y1, x2, y2, frame_width, frame_height)
        speech_text += f"There is a {class_name} at {position}. " if i == 0 else f"And a {class_name} at {position}. "
    if speech_text:
        tts_engine.say(speech_text)
        tts_engine.runAndWait()

def get_relative_position(x1, y1, x2, y2, frame_width, frame_height):
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    horizontal = "left" if center_x < frame_width * 0.33 else "right" if center_x > frame_width * 0.66 else "center"
    vertical = "top" if center_y < frame_height * 0.33 else "bottom" if center_y > frame_height * 0.66 else "middle"
    return f"{horizontal} {vertical}"

def draw_boxes(image, detections, class_names):
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        class_name = class_names[int(class_id)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name}: {confidence:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def process_frame(frame, yolo_model, tts_engine, detected_objects):
    detections, class_names = detect_objects(yolo_model, frame)
    draw_boxes(frame, detections, class_names)
    batch_and_process_descriptions(detections, class_names, frame.shape[1], frame.shape[0], tts_engine, detected_objects)

def update_frame(label, vid, camera_open, yolo_model, tts_engine, detected_objects):
    if camera_open[0]:
        ret, frame = vid.read()
        if ret:
            process_frame(frame, yolo_model, tts_engine, detected_objects)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img)
            label.config(image=img_tk)
            label.image = img_tk
        label.after(10, lambda: update_frame(label, vid, camera_open, yolo_model, tts_engine, detected_objects))

def toggle_camera(toggle_button, label, vid, camera_open, yolo_model, tts_engine, detected_objects, placeholder_img):
    if camera_open[0]:
        camera_open[0] = False
        toggle_button.config(text="Open Camera")
        vid.release()
        cv2.destroyAllWindows()
        label.config(image=placeholder_img)
    else:
        vid.open(0)
        if not vid.isOpened():
            print("Error: Unable to open camera")
            return
        camera_open[0] = True
        toggle_button.config(text="Close Camera")
        update_frame(label, vid, camera_open, yolo_model, tts_engine, detected_objects)

if __name__ == "__main__":
    yolo_model = YOLO('yolov8m.pt')
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 180)
    
    window = tk.Tk()
    window.title("Object Detection")
    window.geometry("900x900")
    
    placeholder_img = ImageTk.PhotoImage(Image.new("RGB", (640, 480), "black"))
    label = Label(window, image=placeholder_img)
    label.pack()
    
    object_listbox = Listbox(window, height=10, width=50)
    object_listbox.pack()
    
    vid = cv2.VideoCapture()
    camera_open = [False]
    detected_objects = []
    
    toggle_button = Button(window, text="Open Camera", command=lambda: toggle_camera(toggle_button, label, vid, camera_open, yolo_model, tts_engine, detected_objects, placeholder_img), height=2, width=15)
    toggle_button.pack()
    
    window.mainloop()
