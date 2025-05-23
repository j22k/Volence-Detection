import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
import time # To calculate FPS

# --- Configuration ---
MODEL_PATH = "MoBiLSTM_model.h5"  # <--- IMPORTANT: SET THIS PATH
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16  # Number of frames to feed into the model
CLASSES_LIST = ["NonViolence", "Violence"] # Make sure this order matches your training

# --- Optional: Configure GPU Memory Growth (helps prevent OOM errors) ---
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found GPUs: {gpus}, memory growth enabled.")
    else:
        print("No GPU found by TensorFlow. Running on CPU.")
except Exception as e:
    print(f"Error configuring GPU: {e}")

# --- Load the trained model ---
print(f"Loading model from: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
    # model.summary() # Optional
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Initialize Webcam ---
# Try different camera indices if 0 doesn't work (e.g., 1, 2)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully. Press 'q' to quit.")

# --- Variables for processing ---
frames_queue = deque(maxlen=SEQUENCE_LENGTH)
predicted_class_name = "Processing..."
prediction_confidence = 0.0
prediction_text_color = (0, 255, 0) # Default Green for NonViolence

# For FPS calculation
prev_time = 0
fps_text = ""

while True:
    # Read a frame from the webcam
    success, frame = cap.read()
    if not success:
        print("Error: Failed to grab frame from webcam.")
        break

    # Make a copy for display to avoid drawing on the frame used for processing
    display_frame = frame.copy()

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    normalized_frame = resized_frame / 255.0
    frames_queue.append(normalized_frame)

    # Check if we have enough frames in the queue to make a prediction
    if len(frames_queue) == SEQUENCE_LENGTH:
        # Prepare the input for the model
        input_frames = np.expand_dims(np.array(list(frames_queue)), axis=0) # (1, seq_len, H, W, C)

        # Make prediction
        try:
            predictions = model.predict(input_frames, verbose=0) # verbose=0 for no log spam
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = CLASSES_LIST[predicted_class_index]
            prediction_confidence = predictions[0][predicted_class_index]

            if predicted_class_name == "Violence":
                prediction_text_color = (0, 0, 255) # Red for Violence
            else:
                prediction_text_color = (0, 255, 0) # Green for NonViolence

        except Exception as e:
            print(f"Error during prediction: {e}")
            predicted_class_name = "Error"
            prediction_text_color = (0,165,255) # Orange for error

    # Calculate FPS
    current_time = time.time()
    if prev_time != 0 : # Avoid division by zero on first frame
        fps = 1 / (current_time - prev_time)
        fps_text = f"FPS: {fps:.1f}"
    prev_time = current_time

    # Display FPS on the frame
    cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the prediction on the frame
    text_to_display = f"{predicted_class_name} ({prediction_confidence*100:.1f}%)"
    cv2.putText(display_frame, text_to_display, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, prediction_text_color, 2)
    
    # Display the resulting frame
    cv2.imshow('Real-time Violence Detection', display_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
print("Application closed.")