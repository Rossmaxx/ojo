import cv2 
import torch
import matplotlib.pyplot as plt
import time


# to supress a warning running on every cycle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def detect_objects(yolo_model, image_tensor):
    result = yolo_model(image_tensor)

    detections = result.pandas().xyxy[0]
    return detections


def batch_descriptions(detections):
    description_batch = []
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        class_name = row['name']

        description = f"Label: {class_name}\nPosition: Center at ({(x2+x1)/2:.2f}, {(y2+y1)/2:.2f}), Size: ({x2-x1}, {y2-y1})\n"
        description_batch.append(description)

    return " ".join(description_batch)


def process_batch(image, detections, previous_labels, language_ch):
    current_labels = {}
    draw_boxes(image, detections)

    for _, row in detections.iterrows():
        class_name = row['name']
        current_labels[class_name] = class_name
    
    # Check if the labels have changed
    if current_labels != previous_labels:
        batch_descriptions(detections)

    # Update the previous labels record
    previous_labels.clear()
    previous_labels.update(current_labels)
    

def draw_boxes(image, detections):
    # Drawing bounding boxes on the image
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        class_name = row['name']
        confidence = row['confidence']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display image
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def openCameraFunction(yolo_model, previous_label):
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        print("Error, video device failed to open")
        return
    
    while True:
        ret, frame = vid.read()
        
        # Step 1: Add "please wait, processing" to the frame
        processing_frame = frame.copy()
        cv2.putText(processing_frame, "Please wait, processing...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', processing_frame)
        cv2.waitKey(1)
        time.sleep(0.5)

        # pre process the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = detect_objects(yolo_model, frame)
        process_batch(frame, img_pil, previous_label)

        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # yolo initialisation
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    yolo_model.eval()
    
    
    # Initialize the previous_labels dictionary
    previous_labels = {}
    openCameraFunction(yolo_model, previous_labels)
