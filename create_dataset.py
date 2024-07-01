# Import necessary libraries
import os  # Library for interacting with the operating system
import pickle  # Library for saving and loading Python objects
import cv2  # OpenCV library for handling real-time computer vision
import mediapipe as mp  # MediaPipe library for detecting hands

# Setup MediaPipe for hand landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the directory where the image data is stored
DATA_DIR = './data'

# Initialize lists to store data and labels
data = []
labels = []

# Loop through each directory in the data folder
for dir_ in os.listdir(DATA_DIR):
    # Loop through each image in the directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # Initialize lists to store adjusted coordinates of hand landmarks
        data_aux = []
        x_ = []
        y_ = []

        # Read the image and convert it to RGB
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe to detect hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # For each set of hand landmarks detected
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract and store each landmark's x and y coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Adjust and store the normalized coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x - min(x_)
                    y = hand_landmarks.landmark[i].y - min(y_)
                    data_aux.append(x)
                    data_aux.append(y)

            # Add the processed data and corresponding label to the lists
            data.append(data_aux)
            labels.append(dir_)

# Save the data and labels to a file using pickle
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
