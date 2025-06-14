import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split  # Used for splitting data into training and testing sets

# ==========================
# Configuration
# ==========================

NAME           = "you"  # Name/label of the person being trained
DATA_DIR       = "data"  # Base directory containing all user folders
TRAINER_FILE   = "trainer.yml"  # File to save the trained model
LABEL_MAP_FILE = "label_map.json"  # JSON file to save label-to-name mapping
TEST_RATIO     = 0.2  # Portion of images to be used for testing (20%)

# ==========================
# Load face images from folder
# ==========================
def load_images():
    faces = []
    labels = []
    person_dir = os.path.join(DATA_DIR, NAME)

    # Check if person directory exists
    if not os.path.isdir(person_dir):
        print(f"❌ Folder not found: {person_dir}")
        return [], []

    # Load and preprocess images
    for img_name in sorted(os.listdir(person_dir)):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
        if img is None:
            continue
        faces.append(img)
        labels.append(0)  # All images belong to label 0 (this person)

    return faces, labels

# ==========================
# Train model and evaluate on test set
# ==========================
def train_and_test_model():
    faces, labels = load_images()
    if not faces:
        print("❌ No images found for training/testing.")
        return

    # Split the dataset into training and testing sets
    train_faces, test_faces, train_labels, test_labels = train_test_split(
        faces, labels, test_size=TEST_RATIO, random_state=42
    )

    # Create and train the LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_faces, np.array(train_labels))

    # Save the trained model to a file
    recognizer.write(TRAINER_FILE)

    # Save the label-to-name mapping to a JSON file
    with open(LABEL_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump({0: NAME}, f, ensure_ascii=False, indent=2)

    print(f"✅ Model trained and saved to {TRAINER_FILE}")
    print(f"📝 Label map saved to {LABEL_MAP_FILE}")
    print(f"🔍 Testing on {len(test_faces)} images...")

    # ==========================
    # Evaluate model on test set
    # ==========================
    correct = 0
    total = len(test_faces)
    confidences = []

    for i, face in enumerate(test_faces):
        predicted_label, confidence = recognizer.predict(face)
        confidences.append(confidence)
        match = predicted_label == test_labels[i]
        if match:
            correct += 1
        print(f"[TEST {i:02}] Prediction: {predicted_label}, Confidence: {confidence:.2f}, Match: {match}")

    # Calculate and display accuracy and average confidence
    accuracy = correct / total * 100
    avg_conf = sum(confidences) / total

    print(f"\n📊 Test accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"📉 Average confidence: {avg_conf:.2f} (lower is better)")

# ==========================
# Entry point
# ==========================
if __name__ == "__main__":
    train_and_test_model()