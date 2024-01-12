import torch
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

def detect_image(weights_path, image_path, img_size=640, confidence_threshold=0.25):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path=weights_path,force_reload=True)
    model.eval()

    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image
    img = model.img_size(img) if img_size is None else model.img_size(img_size)

    # Perform inference
    results = model(img)

    # Process the results
    detected_objects = []
    for det in results.xyxy[0]:
        conf = det[4].item()
        label = int(det[5].item())
        if conf > confidence_threshold:
            # Get bounding box coordinates
            xmin, ymin, xmax, ymax = map(int, det[:4])

            # Add detected object information to the list
            detected_objects.append({
                'label': model.names[label],
                'confidence': conf,
                'bounding_box': [xmin, ymin, xmax, ymax]
            })

    return detected_objects

if __name__ == '__main__':
    # Example usage
    weights_path = 'best.pt'
    image_path = 'k.jpg'

    # Perform image detection
    results = detect_image(weights_path, image_path)

    # Print the detected objects
    for obj in results:
        print(f"Label: {obj['label']}, Confidence: {obj['confidence']:.2f}, Bounding Box: {obj['bounding_box']}")
