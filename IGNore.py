import cv2
import mediapipe as mp
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Listener
import numpy as np
import time
import pickle

# Load the trained model from a file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands and controllers
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
mouse = MouseController()
keyboard = KeyboardController()

# Video capture
cap = cv2.VideoCapture(0)

# Screen and cursor settings
screen_width, screen_height = 1920, 1080
sensitivity = 1.5
smoothing_factor = 0.2
dead_zone_threshold = 0.02

# Last recorded position and initialization of stability check
last_position = (screen_width / 2, screen_height / 2)
last_stable_position = (0, 0)

# Variables for drawing the box
drawing = False
box_start = (0, 0)
box_end = (0, 0)
box_defined = False

# Dragging settings
dragging = False
thumb_index_distance_threshold = 0.04

# Dwell Click Settings
dwell_time = 0.6
last_move_time = time.time()
cursor_position_stable = (0, 0)

# Click detection parameters
last_z = None
click_threshold = 0.025
click_stability = 0.1
last_click_time = time.time()
click_interval = 1.0

# Scroll control
scroll_enabled = False
last_scroll_position = None
scrolling = False
scroll_activation_threshold = 0.03

# Debounce settings for click
debounce_time = 0.3
last_click_event_time = time.time()

# Typing mode
typing_mode = False
labels_dict = {0: 'A', 1: 'B', 2: 'L', 3: 'F'}
last_typed_character = None
typing_interval = 1.0
last_typing_time = time.time()

def draw_box(event, x, y, flags, param):
    global box_start, box_end, drawing, box_defined

    if not box_defined:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            box_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            box_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            box_end = (x, y)
            box_defined = True

def process_hand_landmarks(hand_landmarks, frame):
    global last_position, last_stable_position, dragging, last_click_time, cursor_position_stable, last_move_time, typing_mode, last_z, last_typing_time, last_typed_character
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

    cursor_x = int(index_tip.x * frame.shape[1])
    cursor_y = int(index_tip.y * frame.shape[0])

    if box_x_min < cursor_x < box_x_max and box_y_min < cursor_y < box_y_max:
        if typing_mode:
            handle_typing_mode(hand_landmarks, frame)
        else:
            handle_mouse_mode(index_tip, thumb_tip, middle_tip, cursor_x, cursor_y)

def handle_mouse_mode(index_tip, thumb_tip, middle_tip, cursor_x, cursor_y):
    global last_position, last_stable_position, dragging, last_click_event_time, cursor_position_stable, last_move_time, last_z, last_click_time

    # Calculate the position within the box
    relative_x = (cursor_x - box_x_min) / (box_x_max - box_x_min)
    relative_y = (cursor_y - box_y_min) / (box_y_max - box_y_min)

    # Map the position to the screen dimensions
    screen_x = int(relative_x * screen_width)
    screen_y = int(relative_y * screen_height)

    # Apply smoothing
    smoothed_x = last_position[0] + (screen_x - last_position[0]) * smoothing_factor
    smoothed_y = last_position[1] + (screen_y - last_position[1]) * smoothing_factor

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
            if time.time() - last_click_event_time > debounce_time:  # Debouncing
                mouse.click(Button.left, 1)
                last_click_event_time = time.time()
                print("Dwell click triggered")
            last_move_time = time.time()
    else:
        cursor_position_stable = (int(smoothed_x), int(smoothed_y))
        last_move_time = time.time()

    # Forward movement click detection
    if last_z is not None and (last_z - index_tip.z > click_threshold) and (time.time() - last_click_time > click_stability):
        if time.time() - last_click_event_time > debounce_time:  # Debouncing
            mouse.click(Button.left, 1)
            last_click_event_time = time.time()
            print("Click triggered by forward movement")
        last_click_time = time.time()

    last_z = index_tip.z  # Update the last z-coordinate

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

def handle_typing_mode(hand_landmarks, frame):
    global last_typing_time, last_typed_character
    x_ = [landmark.x for landmark in hand_landmarks.landmark]
    y_ = [landmark.y for landmark in hand_landmarks.landmark]

    min_x = min(x_)
    min_y = min(y_)

    data_aux = []
    for x, y in zip(x_, y_):
        data_aux.append(x - min_x)
        data_aux.append(y - min_y)

    if len(data_aux) == 42:
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        current_time = time.time()
        if predicted_character != last_typed_character or (current_time - last_typing_time > typing_interval):
            last_typed_character = predicted_character
            last_typing_time = current_time

            # Display the character
            x1, y1 = int(min(x_) * frame.shape[1]) - 10, int(min(y_) * frame.shape[0]) - 10
            x2, y2 = int(max(x_) * frame.shape[1]) + 10, int(max(y_) * frame.shape[0]) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            keyboard.press(predicted_character)
            keyboard.release(predicted_character)

def toggle_typing_mode():
    global typing_mode
    typing_mode = not typing_mode
    print("Typing mode toggled:", typing_mode)

def on_press(key):
    global typing_mode
    try:
        if key.char == 'k':  # Toggle typing mode on 'k' key press
            toggle_typing_mode()
    except AttributeError:
        pass

cv2.namedWindow("Draw Box")
cv2.setMouseCallback("Draw Box", draw_box)

# Start listening to keyboard input
listener = Listener(on_press=on_press)
listener.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if not box_defined:
        if drawing:
            cv2.rectangle(frame, box_start, box_end, (0, 255, 0), 2)
        cv2.imshow("Draw Box", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        box_x_min, box_y_min = min(box_start[0], box_end[0]), min(box_start[1], box_end[1])
        box_x_max, box_y_max = max(box_start[0], box_end[0]), max(box_start[1], box_end[1])

        results = hands.process(frame_rgb)

        # Draw the defined box
        cv2.rectangle(frame, (box_x_min, box_y_min), (box_x_max, box_y_max), (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                process_hand_landmarks(hand_landmarks, frame)

    cv2.imshow("Draw Box", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
listener.stop()  # Stop listening to keyboard input
