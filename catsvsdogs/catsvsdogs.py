import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ✅ Your updated dataset folder name
DATA_DIR = "catsvsdogs"
IMG_SIZE = 64

# Load and preprocess the images
def load_data(data_dir):
    X, y = [], []
    for label_name in ("cat", "dog"):
        folder = os.path.join(data_dir, label_name)
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img.flatten())
            y.append(0 if label_name == "cat" else 1)
    return np.array(X)/255.0, np.array(y)

# Load the dataset
print("[INFO] Loading dataset...")
X, y = load_data(DATA_DIR)
print(f"[INFO] Loaded {len(X)} images.")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
print("[INFO] Training SVM classifier...")
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))
print("Accuracy:", accuracy_score(y_test, y_pred))
