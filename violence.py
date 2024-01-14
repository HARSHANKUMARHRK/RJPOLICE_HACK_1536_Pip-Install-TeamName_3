import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
MoBiLSTM_model = load_model("violence_classification.h5")

# Assuming IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH, and CLASSES_LIST are defined somewhere in your code
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["Non-Violence", "Violence"]  # Replace with your actual class names

def preprocess_frame(frame):
    # Resize the frame to the desired resolution
    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to the range [0, 1]
    return normalized_frame

def predict_video(video_file_path):
    video_reader = cv2.VideoCapture(video_file_path)
    frames_list = []

    for frame_counter in range(SEQUENCE_LENGTH):
        success, frame = video_reader.read()

        if not success:
            break

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)
        frames_list.append(preprocessed_frame)

    # Assuming MoBiLSTM_model is a global variable
    predicted_labels_probabilities = MoBiLSTM_model.predict(np.expand_dims(frames_list, axis=0))[0]
    predicted_label = np.argmax(predicted_labels_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label]

    print(f'Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

    video_reader.release()

# Example usage
predict_video("uploaded_video.mp4")
