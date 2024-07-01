import pickle
import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller as MouseController, Button
from collections import deque
import time

# Load the trained model from a file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Setup MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)

# Initialize mouse controller
mouse = MouseController()

# Define screen width and height for mouse control scaling
screen_width, screen_height = 1920, 1080

# Smooth mouse and scroll movement by averaging the last N positions
smooth_factor = 5
index_positions = deque(maxlen=smooth_factor)
scroll_positions = deque(maxlen=smooth_factor)

# Clicking mechanism settings
click_distance_threshold = 20  # Adjust based on your system
drag_threshold = 30            # Distance for starting drag
last_click_time = 0
click_cooldown = 0.3
is_dragging = False

# Initialize variables for gesture tracking
is_scrolling_gesture_active = False
gesture_start_threshold = 40  # Adjusted for better gesture recognition
gesture_end_threshold = 50
prev_y = 0
scroll_accumulator = 0

# Main loop to process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    active_gesture = "None"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Get index and thumb finger tips
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_x, index_y = int(index_tip.x * W), int(index_tip.y * H)
            thumb_x, thumb_y = int(thumb_tip.x * W), int(thumb_tip.y * H)

            # Calculate distance for click and drag detection
            index_thumb_distance = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

            # Click detection
            if not is_dragging and index_thumb_distance < click_distance_threshold and time.time() - last_click_time > click_cooldown:
                mouse.click(Button.left, 1)
                last_click_time = time.time()
                active_gesture = "Click"

            # Dragging gesture
            elif index_thumb_distance < drag_threshold and not is_dragging:
                mouse.press(Button.left)
                is_dragging = True
                active_gesture = "Dragging"
            elif index_thumb_distance > drag_threshold and is_dragging:
                mouse.release(Button.left)
                is_dragging = False

            # Scroll gesture
            if index_thumb_distance > gesture_start_threshold and not is_dragging:
                is_scrolling_gesture_active = True
                scroll_positions.append(index_y)
                active_gesture = "Scrolling"
            elif index_thumb_distance < gesture_end_threshold:
                is_scrolling_gesture_active = False
                scroll_positions.clear()

            if is_scrolling_gesture_active and len(scroll_positions) > 1:
                current_y = scroll_positions[-1]
                if prev_y != 0:
                    scroll_diff = current_y - prev_y
                    scroll_accumulator += scroll_diff

                    if abs(scroll_accumulator) > 15:
                        scroll_steps = int(scroll_accumulator / 15)
                        mouse.scroll(0, -scroll_steps)
                        scroll_accumulator = 0

                prev_y = current_y

            # General mouse movement
            index_positions.append((index_x, index_y))
            if index_positions:
                avg_index_x = int(sum(pos[0] for pos in index_positions) / len(index_positions))
                avg_index_y = int(sum(pos[1] for pos in index_positions) / len(index_positions))
                mouse.position = (avg_index_x * screen_width // W, avg_index_y * screen_height // H)

    # Show active gesture on screen
    cv2.putText(frame, f"Active Gesture: {active_gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
