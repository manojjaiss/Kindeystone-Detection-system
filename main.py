import cv2
import numpy as np
from skimage.filters import gabor
from skimage.segmentation import chan_vese
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image1.jpg', 0)  # Load in grayscale

# Preprocessing: Remove speckle noise using median filter
median_filtered = cv2.medianBlur(image, 5)

# Apply Gabor filter
real, imag = gabor(median_filtered, frequency=0.6)

# Histogram equalization
equalized = cv2.equalizeHist((real * 255).astype(np.uint8))

# Level Set Segmentation to detect kidney region
cv_kidney = chan_vese(equalized / 255.0, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_num_iter=200, dt=0.5, init_level_set="checkerboard")[0]
cv_kidney = (cv_kidney > 0).astype(np.uint8) * 255

# Level Set Segmentation to detect stone region
cv_stone = chan_vese(cv_kidney / 255.0, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_num_iter=200, dt=0.5, init_level_set="checkerboard")[0]
cv_stone = (cv_stone > 0).astype(np.uint8) * 255

# Prepare data for ANN
X = cv_stone.flatten().reshape(-1, 1)
y = np.array([1 if pixel > 0 else 0 for pixel in X]).reshape(-1, 1)

# One-hot encode the labels
y_categorical = to_categorical(y)

# Define the ANN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y_categorical, epochs=10, batch_size=32)

# Predict
predictions = model.predict(X)

# Reshape predictions to image format
prediction_image = predictions[:, 1].reshape(cv_stone.shape)

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Kidney Segmented Image')
plt.imshow(cv_kidney, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Stone Segmented Image')
plt.imshow(prediction_image, cmap='gray')

plt.show()