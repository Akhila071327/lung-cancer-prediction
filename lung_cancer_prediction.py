import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set dataset path
DATASET_PATH = r"C:\Users\akhil\Downloads\project\archive\The IQ-OTHNCCD lung cancer dataset"

# Define categories
categories = ["Normal cases", "Malignant cases", "Bengin cases"]
data = []
labels = []

# Load and preprocess images
for category in categories:
    path = os.path.join(DATASET_PATH, category)
    label = categories.index(category)
    
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img = cv2.resize(img, (128, 128))  # Resize to 128x128 pixels
        data.append(img)
        labels.append(label)

# Convert lists to NumPy arrays
data = np.array(data) / 255.0  # Normalize the images
labels = np.array(labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Reshape for CNN input
X_train = X_train.reshape(-1, 128, 128, 1)
X_test = X_test.reshape(-1, 128, 128, 1)

# ✅ CNN Model
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: Normal, Malignant, Benign
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train CNN model
cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# ✅ Random Forest Model
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_flat, y_train)
rf_predictions = rf_model.predict(X_test_flat)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# ✅ Support Vector Machine (SVM) Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_flat, y_train)
svm_predictions = svm_model.predict(X_test_flat)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")

# ✅ Predict function (using majority voting)
def predict_lung_cancer(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128)) / 255.0
    img = img.reshape(1, 128, 128, 1)
    img_flat = img.reshape(1, -1)

    # Get predictions from each model
    cnn_prediction = np.argmax(cnn_model.predict(img))
    rf_prediction = rf_model.predict(img_flat)[0]
    svm_prediction = svm_model.predict(img_flat)[0]

    # Majority voting
    predictions = [cnn_prediction, rf_prediction, svm_prediction]
    final_prediction = max(set(predictions), key=predictions.count)

    classes = ["Normal cases", "Malignant cases", "Benign cases"]
    return classes[final_prediction]

# ✅ Test the model
test_image = r"C:\Users\akhil\Downloads\project\archive\Test cases\000310_01_01_115.png"
result = predict_lung_cancer(test_image)
print(f"Predicted Lung Cancer Type: {result}")
