from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os
import cv2
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Accuracy


def load_images_and_labels(folder_path, img_size=(128, 128)):

  images = []
  labels = []
  class_names = []

  for subfolder in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolder)

    if not os.path.isdir(subfolder_path):
      continue

    for filename in os.listdir(subfolder_path):
      image_path = os.path.join(subfolder_path, filename)
      img = cv2.imread(image_path)

      if img is not None:
        # Resize image
        img = cv2.resize(img, img_size)
        images.append(img)
        labels.append(subfolder)
        class_names.append(subfolder)  # Store unencoded class names

  X = np.array(images)
  y = np.array(labels)

  le = LabelEncoder()
  y = le.fit_transform(y)

  return X, y, class_names

# Example usage
folder_path = "Frame_Dataset"
img_size = (128, 128)  # Adjust image size as needed
X, y, class_names = load_images_and_labels(folder_path, img_size)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
  MaxPooling2D((2, 2)),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
  Flatten(),
  Dense(64, activation='relu'),
  Dense(1, activation='sigmoid')
])

# Model compilation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[Accuracy()])

# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Save the model (optional)
model.save('my_cnn_model.h5')


import cv2
import os
from tensorflow.keras.models import load_model

# Define paths
model_path = "my_cnn_model.h5"  # Path to your trained CNN model
video_path = "Video_Dataset/Normal/shop_lifter_n_2.mp4"  # Path to the video file

# Load the CNN model
model = load_model(model_path)

# Function to preprocess an image for the model
def preprocess_image(img):
  # Resize the image to match the model's input size
  img = cv2.resize(img, (model.input_shape[1], model.input_shape[2]))

  # Add an extra dimension for batch processing (even though it's a single image)
  img = img.reshape((1,) + img.shape)
  return img

# Read the video

cap = cv2.VideoCapture(video_path)
font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a suitable font

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Check if frame is read correctly
  if not ret:
    print("Can't receive frame (stream end?). Exiting...")
    break

  # Preprocess the frame
  preprocessed_frame = preprocess_image(frame)

  # Make prediction using the model
  prediction = model.predict(preprocessed_frame)[0]  # Assuming model outputs a single value

  # Process the prediction result (adjust based on your model's output)
  text = "Positive Class" if prediction > 0.5 else "Negative Class"  # Adjust threshold and text

  # Put text on the frame (adjust position and font size)
  cv2.putText(frame, text, (10, 30), font, 1, (0, 255, 0), 2)

  # Display the frame with prediction
  cv2.imshow('Frame', frame)
  if cv2.waitKey(1) == ord('q'):
    break

# Release resources
cap.release()
cv2.destroyAllWindows()