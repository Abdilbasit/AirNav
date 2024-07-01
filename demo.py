import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from pynput.keyboard import Key, Controller
import time

# Load the trained model from a file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)  # Ensure camera index is correct

# Setup MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands with options
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary to map model predictions to character labels
labels_dict = {0: 'A', 1: 'B', 2: 'L', 3: 'Fuck You'}

# Dictionary to define responses based on inputs
responses = {
    'A': 'How are you doing?',
    'B': 'How is your day?',
    #'L': 'Let me know',
    'Fuck You': 'No, fuck you!'
}

keyboard = Controller()
last_mute_time = time.time() - 10  # Initialize mute time

# Loop to continuously read frames from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        data_aux = []
        x_ = [landmark.x for landmark in hand_landmarks.landmark]
        y_ = [landmark.y for landmark in hand_landmarks.landmark]

        min_x = min(x_)
        min_y = min(y_)

        for x, y in zip(x_, y_):
            data_aux.append(x - min_x)
            data_aux.append(y - min_y)

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            response_message = responses.get(predicted_character,)  #"No response defined"  )

            # Handle mute logic specifically for 'L'
            current_time = time.time()
            if predicted_character == 'L' and (current_time - last_mute_time > 2):
                keyboard.press(Key.media_volume_mute)
                keyboard.release(Key.media_volume_mute)
                print("System muted")
                last_mute_time = current_time

            # Display the character and response immediately
            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, response_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the video frame with annotations
    cv2.imshow('frame', frame)

    # Wait for the 's' key to be pressed to save the user input
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        print(f"User input: {predicted_character}")
    elif key & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

def print_pyramid(word):
    length = len(word)
    for i in range(length):
        # Calculate the number of spaces needed for padding
        spaces = ' ' * (length - i - 1)
        # Get the substring to print for this level of the pyramid
        substring = word[:i + 1]
        # Print the line with padding and substring centered
        print(spaces + ' '.join(substring) + spaces)

# Call the function with the word "START"
print_pyramid("START")
