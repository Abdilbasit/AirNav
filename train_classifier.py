# Import necessary libraries
import pickle  # Library for loading and saving Python objects
import numpy as np  # Library for numerical operations
from sklearn.ensemble import RandomForestClassifier  # Import the RandomForest algorithm
from sklearn.model_selection import train_test_split  # Tool to split data into training and test sets
from sklearn.metrics import accuracy_score  # Tool to evaluate the model

# Load the data and labels from the preprocessed file
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])  # Convert data list to a NumPy array for machine learning processing
labels = np.asarray(data_dict['labels'])  # Convert labels list to a NumPy array

# Split the data into training and test sets
# 80% of the data is used for training, 20% for testing
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier()

# Train the model using the training data
model.fit(x_train, y_train)

# Predict the labels for the test data
y_predict = model.predict(x_test)

# Calculate the accuracy of the model on the test data
score = accuracy_score(y_predict, y_test)

# Print the accuracy in percentage
print(f'{score * 100}% of samples were classified correctly !')

# Save the trained model to a file using pickle
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
