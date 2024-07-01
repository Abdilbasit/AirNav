import cv2
import mediapipe as mp
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Listener, Key
import numpy as np
import time

# Initialize MediaPipe Hands and mouse controller
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
mouse = MouseController()


# Initialize MediaPipe Hands with options
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

mp_drawing_styles = mp.solutions.drawing_styles

# Video capture
cap = cv2.VideoCapture(0)

# Screen and cursor settings
screen_width, screen_height = 1920, 1080
sensitivity = 3  # Control sensitivity
smoothing_factor = 0.2  # Controls how smooth the mouse movements are
dead_zone_threshold = 0.01  # Dead zone for finger movement

# Last recorded position and initialization of stability check
last_position = (screen_width / 2, screen_height / 2)
last_stable_position = (0, 0)

# Dragging settings
dragging = False
thumb_index_distance_threshold = 0.05  # Adjust based on testing

# Dwell Click Settings
dwell_time = 0.5  # Time the cursor must be still to trigger a click
last_move_time = time.time()
cursor_position_stable = (0, 0)

# Click and drag detection parameters
last_z = None
click_threshold = 0.03
click_stability = 0.1
last_click_time = time.time()

# Scroll control
scroll_enabled = False
last_scroll_position = None
scroll_activation_threshold = 0.04  # Distance fingers must move together to start scroll

def on_press(key):
    global scroll_enabled, last_scroll_position
    try:
        if key.char == 's':  # Toggle scroll on 's' key press
            scroll_enabled = not scroll_enabled
            scrolling = False  # Reset scrolling status when toggling
            last_scroll_position = None  # Reset position when enabling/disabling scroll
            print("Scrolling toggled:", "Enabled" if scroll_enabled else "Disabled")
    except AttributeError:
        pass

# Start listening to keyboard input
listener = Listener(on_press=on_press)
listener.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()

            # Calculate cursor position based on the index finger
            cursor_x = int((index_tip.x - 0.5) * sensitivity * screen_width + screen_width / 2)
            cursor_y = int((index_tip.y - 0.5) * sensitivity * screen_height + screen_height / 2)
            cursor_z = index_tip.z  # Depth value of the index finger tip
            # Calculate movement difference
            movement_diff = np.sqrt((cursor_x - last_stable_position[0]) ** 2 + (cursor_y - last_stable_position[1]) ** 2)

            # Update only if movement is beyond the dead zone
            if movement_diff > dead_zone_threshold:
                smoothed_x = last_position[0] + (cursor_x - last_position[0]) * smoothing_factor
                smoothed_y = last_position[1] + (cursor_y - last_position[1]) * smoothing_factor
                last_stable_position = (cursor_x, cursor_y)
            else:
                smoothed_x, smoothed_y = last_position

            # Set mouse position
            mouse.position = (int(smoothed_x), int(smoothed_y))
            last_position = (smoothed_x, smoothed_y)


            # Distance between index finger and thumb for dragging
            thumb_index_distance = np.sqrt((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2)

            # Handle dragging
            if thumb_index_distance < thumb_index_distance_threshold and not dragging:
                mouse.press(Button.left)
                dragging = True
                print("Dragging started")
            elif thumb_index_distance > thumb_index_distance_threshold and dragging:
                mouse.release(Button.left)
                dragging = False
                print("Dragging ended")

            # Dwell click detection
            if not dragging and cursor_position_stable == (int(smoothed_x), int(smoothed_y)):
                if time.time() - last_move_time > dwell_time:
                    mouse.click(Button.left, 1)
                    print("Dwell click triggered")
                    last_move_time = time.time()
            else:
                cursor_position_stable = (int(smoothed_x), int(smoothed_y))
                last_move_time = time.time()

            # Forward movement click detection
            if last_z is not None and (last_z - cursor_z > click_threshold) and (time.time() - last_click_time > click_stability):
                mouse.click(Button.left, 1)
                print("Click triggered by forward movement")
                last_click_time = time.time()
            
            last_z = cursor_z  # Update the last z-coordinate

            index_y = index_tip.y
            middle_y = middle_tip.y
            avg_y = (index_y + middle_y) / 2

            # Handle scrolling only when enabled
            if scroll_enabled:
                if last_scroll_position is not None and scrolling:
                    scroll_change = last_scroll_position - avg_y
                    mouse.scroll(0, int(scroll_change * 100))
                # Determine if the fingers have moved sufficiently together to start scrolling
                if np.abs(index_y - middle_y) < scroll_activation_threshold:
                    scrolling = True
                else:
                    scrolling = False
                last_scroll_position = avg_y

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
listener.stop()  # Stop listening to keyboard input
