import cv2 
import torch
import matplotlib.pyplot as plt
import requests
import pyttsx3
import os
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

def translate_speak(description, choice):
    languages = {
        'EN' : 'English',
        'DE' : 'German',
        'FR' : 'French',
        'ES' : 'Spanish',
        'PO' : 'Portugese'
    }

    if choice not in languages:
        choice = 'EN'

    data = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": "Translate this to " + languages[choice] + " : "+ description}
        ]
    }

    response = requests.post(url, json=data, headers=headers)
    # Handle response...
    print("API called, labels changed")

    try:
        response.raise_for_status()  # Raises an HTTPError for bad responses

        result = response.json()
        # Extract the message content
        content_list = result.get('content', [])
        if content_list and isinstance(content_list, list) and 'text' in content_list[0]:
            assistant_message = content_list[0]['text']

            # output
            print("Claude's response:", assistant_message)
            speak(assistant_message)
            # write to txt
            output = 'records.txt'
            with open(output, 'a') as outfile:
                outfile.write(assistant_message)  

        else:
            print("Claude's response structure might have changed or the response is incomplete.")
            print("Full response:", result)

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

def speak(description):
    speech_engine = pyttsx3.init()
    speech_engine.setProperty('volume', 0.9)
    speech_engine.say(description)
    speech_engine.runAndWait()

def process_batch(image, detections, previous_labels, language_ch):
    current_labels = {}
    draw_boxes(image, detections)

    for _, row in detections.iterrows():
        class_name = row['name']
        current_labels[class_name] = class_name
    
    # Check if the labels have changed
    if current_labels != previous_labels:
        description_text = batch_descriptions(detections)
        
        translate_speak(description_text, language_ch)

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


def openCameraFunction(yolo_model, previous_label, language_ch):
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
        process_batch(frame, img_pil, previous_label, language_ch)

        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # claude integrations
    # api key
    # claude_api_key =

    # API endpoint, add

    # Headers
    headers = {
        "Content-Type": "application/json",
        # missing param
        "anthropic-version": "2023-06-01"
    }

    # Message data
    data = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 1000,
        "messages": [
            {"role": "user",
             "content": "Imagine you are a blind person's assistant and the frame data will be given soon. Describe the scene per frame."}
        ]
    }

    # Make the API call
    response = requests.post(url, json=data, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        assistant_message = result['content'][0]['text']
        print("Claude's response:", assistant_message)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

    # clear records json file for new entry
    if os.path.exists('records.txt'):
        os.remove('records.txt')

    # yolo initialisation
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    yolo_model.eval()
    
    
    # Initialize the previous_labels dictionary
    previous_labels = {}
    print("EN - English, DE - German, FR - French, ES - Spanish, PO - Portugese")
    choice = input("Enter language : ")
    openCameraFunction(yolo_model, previous_labels, choice)
