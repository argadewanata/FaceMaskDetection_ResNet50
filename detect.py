import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import mediapipe as mp
import time

# Load the trained model
model_path = 'MaskDetectionModel.keras'
model = load_model(model_path)

class_labels = ['with_mask', 'without_mask']

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Flag to toggle drawing face landmarks
draw_face_landmarks = False

# Define the preprocessing function


def preprocess_frame(frame, target_size=(224, 224)):
    # Resize the frame to the 224x224
    resized_frame = cv2.resize(frame, target_size)
    img_array = img_to_array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_frame = preprocess_input(img_array)
    return preprocessed_frame


def detect_and_predict_mask(frame, model, face_detection, draw_landmarks=True):
    # Convert the frame from BGR (default by OpenCV) to RGB (required for Mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Perform face detection using Mediapipe
    results = face_detection.process(rgb_frame)
    # Get the detected faces. If no faces are detected, set detections to an empty list.
    detections = results.detections if results.detections else []
    # Loop through all detected faces
    for detection in detections:
        # Get the bounding box coordinates for the detected face
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape  # Get the image height and width
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin *
                                               ih), int(bboxC.width * iw), int(bboxC.height * ih)
        # Crop the face from the frame using the bounding box coordinates
        cropped_face = frame[y:y+h, x:x+w]
        # Preprocess the cropped face for the mask detection model
        preprocessed_face = preprocess_frame(cropped_face)
        # Predict if the person is wearing a mask or not
        prediction = model.predict(preprocessed_face)
        # Get the predicted class (0 or 1)
        predicted_class = np.argmax(prediction, axis=1)[0]
        # Get the label corresponding to the predicted class
        label = class_labels[predicted_class]
        # Draw a bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        # Put the prediction label above the bounding box
        cv2.putText(frame, f"{label}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # If draw_landmarks flag is True, draw face landmarks on the frame
        if draw_landmarks:
            mp_drawing.draw_detection(frame, detection)
    # Return the frame with bounding boxes and predictions drawn on it and whether any faces were detected
    return frame, len(detections) > 0


def main():
    global draw_face_landmarks
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            start_time = time.time()
            frame, faces_detected = detect_and_predict_mask(
                frame, model, face_detection, draw_landmarks=draw_face_landmarks)
            end_time = time.time()

            # Calculate and display FPS
            fps = 1 / (end_time - start_time)
            if not faces_detected:
                # Adding a short delay when no faces are detected
                time.sleep(0.03)
                fps = 30  # Setting a constant FPS value when no faces are detected

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Mask Detection', frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('l'):
                draw_face_landmarks = not draw_face_landmarks
                print(f"Draw face landmarks toggled to {draw_face_landmarks}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
