# Import necessary libraries
import os  # Library for interacting with the operating system
import cv2  # OpenCV library for handling real-time computer vision

# Define the directory where data will be stored
DATA_DIR = './data'
# Check if the directory exists, if not, create it
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes and the size of the dataset per class
number_of_classes = 3
dataset_size = 100

# Start the camera
cap = cv2.VideoCapture(0)  # 0 means the default camera

# Loop over each class
for j in range(number_of_classes):
    # Create a directory for each class inside the data directory
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    # Notify the user about the data collection for the current class
    print('Collecting data for class {}'.format(j))

    # Wait for the user to be ready before collecting data
    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        # Display a prompt on the frame
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Display the frame
        # Wait for the user to press 'q' to quit the prompt loop
        if cv2.waitKey(25) == ord('q'):
            break

    # Initialize a counter to keep track of the number of images collected
    counter = 0
    # Collect the specified number of images for the class
    while counter < dataset_size:
        ret, frame = cap.read()  # Read a frame from the camera
        cv2.imshow('frame', frame)  # Display the frame
        cv2.waitKey(25)  # Wait for a brief moment
        # Save the frame as an image file in the respective class folder
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1  # Increment the counter

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
