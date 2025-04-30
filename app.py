import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ✅ Setup Flask app
app = Flask(__name__)

# ✅ Load and preprocess dataset
DATASET_PATH = r"C:\Users\akhil\Downloads\project\archive\The IQ-OTHNCCD lung cancer dataset"
categories = ["Normal cases", "Malignant cases", "Bengin cases"]
data = []
labels = []

# Load images
for category in categories:
    path = os.path.join(DATASET_PATH, category)
    label = categories.index(category)
    
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        data.append(img)
        labels.append(label)

data = np.array(data) / 255.0
labels = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train_cnn = X_train.reshape(-1, 128, 128, 1)
X_test_cnn = X_test.reshape(-1, 128, 128, 1)

# ✅ CNN model
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
    Dense(3, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train, epochs=10, validation_data=(X_test_cnn, y_test))

# ✅ Random Forest and SVM models
X_train_flat = X_train_cnn.reshape(len(X_train_cnn), -1)
X_test_flat = X_test_cnn.reshape(len(X_test_cnn), -1)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_flat, y_train)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_flat, y_train)

# ✅ Prediction logic
def predict_lung_cancer_from_image(img_array):
    img = cv2.resize(img_array, (128, 128)) / 255.0
    img = img.reshape(1, 128, 128, 1)
    img_flat = img.reshape(1, -1)

    cnn_prediction = np.argmax(cnn_model.predict(img))
    rf_prediction = rf_model.predict(img_flat)[0]
    svm_prediction = svm_model.predict(img_flat)[0]

    predictions = [cnn_prediction, rf_prediction, svm_prediction]
    final_prediction = max(set(predictions), key=predictions.count)

    classes = ["Normal cases", "Malignant cases", "Benign cases"]
    return classes[final_prediction]

# ✅ Flask Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    result = predict_lung_cancer_from_image(img)
    return render_template('index.html', result=result)

# ✅ Run server
if __name__ == '__main__':
    app.run(debug=True)
