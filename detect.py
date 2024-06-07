import cv2
import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image

# Define model path
model_path = 'MaskDetectionModel.keras'

# Load the pretrained model using tf.keras
model = tf.keras.models.load_model(model_path)

# Load the cascade to detect human faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns None
    scale_factor = 1.05
    min_neighbors = 6
    faces = face_cascade.detectMultiScale(img, scale_factor, min_neighbors, minSize=(100, 100),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    cropped_face = None
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

# Doing some face recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    face = face_extractor(frame)
    if face is not None:
        # Resize the image to match the pretrained model input
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image array

        # Use the model to predict
        pred = model.predict(img_array)
        mask_prob = pred[0][0]  # Assuming the first class is 'mask'

        # Check the threshold
        if mask_prob > 0.5:
            name = 'Mask Found'
        else:
            name = 'No Mask Found'

        cv2.putText(frame, name, (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face found", (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
