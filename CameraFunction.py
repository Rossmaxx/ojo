import cv2 

from ultralytics import YOLO


def detect_objects(yolo_model, image_tensor):
    results = yolo_model(image_tensor, conf=0.6)
    detections = results[0].boxes.data.cpu().numpy()  # Convert tensor to NumPy
    class_names = yolo_model.names  # Class names from the model
    return detections, class_names


def batch_descriptions(detections, class_names):
    description_batch = []
    for detection in detections:
        x1, y1, x2, y2, _, class_id = detection[:6]
        class_name = class_names[int(class_id)]
        description = f"Label: {class_name}\nPosition: Center at ({(x2+x1)/2:.2f}, {(y2+y1)/2:.2f}), Size: ({x2-x1}, {y2-y1})\n"
        description_batch.append(description)
    return " ".join(description_batch)


def process_batch(frame, detections, class_names, previous_labels):
    current_labels = {}
    
    # Draw bounding boxes directly on 'frame'
    draw_boxes(frame, detections, class_names)

    for detection in detections:
        _, _, _, _, _, class_id = detection[:6]
        class_name = class_names[int(class_id)]
        current_labels[class_name] = class_name

    # Check if the labels have changed
    if current_labels != previous_labels:
        print(batch_descriptions(detections, class_names))

    # Update the previous labels record
    previous_labels.clear()
    previous_labels.update(current_labels)


def draw_boxes(image, detections, class_names):
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        class_name = class_names[int(class_id)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name}: {confidence:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def openCameraFunction(yolo_model, previous_label):
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        print("Error, video device failed to open")
        return
    
    while True:
        ret, frame = vid.read()
        
        img_pil, classname = detect_objects(yolo_model, frame)
        process_batch(frame, img_pil, classname , previous_label)

        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # yolo initialisation
    yolo_model = YOLO('yolov8m.pt')
    
    # Initialize the previous_labels dictionary
    previous_labels = {}
    openCameraFunction(yolo_model, previous_labels)
