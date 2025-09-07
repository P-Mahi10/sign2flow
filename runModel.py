import os
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# ---------------- CONFIG -------------------
MODEL_PATH = "sign_model.h5"
DATASET_PATH = "Dataset_processed"
TIMESTEPS = 60
NUM_LANDMARKS = 48  # 6 arms + 21 left + 21 right
NUM_COORDS = 2
# -------------------------------------------

#Load label names from subfolder structure
def get_labels_from_folder(main_folder):
    labels = sorted([
        folder for folder in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, folder))
    ])
    return labels

# Get label names and encoder
label_names = get_labels_from_folder(DATASET_PATH)
le = LabelEncoder()
le.fit(label_names)

#Load model
model = load_model(MODEL_PATH)

# MediaPipe setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

#Sequence buffer
sequence = deque(maxlen=TIMESTEPS)

# Start webcam
cap = cv2.VideoCapture(0)
print("Live prediction started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    # Initialize all-zero coords for 48 landmarks
    coords = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

    # Fill coordinates if detected
    try:
        # Arms (shoulders → elbows → wrists)
        if results.pose_landmarks:
            pose = results.pose_landmarks.landmark
            coords[0] = [pose[11].x, pose[11].y]  # left shoulder
            coords[1] = [pose[13].x, pose[13].y]  # left elbow
            coords[2] = [pose[15].x, pose[15].y]  # left wrist
            coords[3] = [pose[12].x, pose[12].y]  # right shoulder
            coords[4] = [pose[14].x, pose[14].y]  # right elbow
            coords[5] = [pose[16].x, pose[16].y]  # right wrist

        # Left hand
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                coords[6 + i] = [lm.x, lm.y]

        # Right hand
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                coords[27 + i] = [lm.x, lm.y]

    except Exception as e:
        pass  # Any missing data stays zero

    # Append current frame to sequence
    sequence.append(coords)

    # Predict
    if len(sequence) == TIMESTEPS:
        seq_array = np.array(sequence)  # (60, 48, 2)
        seq_array = seq_array.reshape(1, TIMESTEPS, NUM_LANDMARKS * NUM_COORDS)  # (1, 60, 96)

        y_pred = model.predict(seq_array, verbose=0)[0]
        predicted_index = np.argmax(y_pred)
        predicted_label = le.inverse_transform([predicted_index])[0]
        confidence = y_pred[predicted_index]

        # Display prediction
        display_text = f"{predicted_label} ({confidence:.2f})"
        cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

    # Draw landmarks
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
