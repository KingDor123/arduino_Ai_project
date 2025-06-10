import os
import json
import cv2
import numpy as np

# ======= הגדרות במקום config.py =======
NAME           = "you"
DATA_DIR       = "data"
TRAINER_FILE   = "trainer.yml"
LABEL_MAP_FILE = "label_map.json"
# ========================================

def train_model():
    faces = []
    labels = []
    person_dir = os.path.join(DATA_DIR, NAME)

    # בדיקה שתיקיית האימון קיימת
    if not os.path.isdir(person_dir):
        print(f"❌ לא נמצאה תיקייה: {person_dir}")
        return

    for img_name in sorted(os.listdir(person_dir)):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        faces.append(img)
        labels.append(0)  # כל התמונות בלייבל 0

    if not faces:
        print("❌ לא נמצאו תמונות לאימון. תרוץ קודם את collect.py")
        return

    # מאמן LBPH
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    # שומר את המודל
    recognizer.write(TRAINER_FILE)
    # שומר את המיפוי למיפוי לייבל → שם
    with open(LABEL_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump({0: NAME}, f, ensure_ascii=False, indent=2)

    print(f"✅ אימון הושלם: נשמרו {TRAINER_FILE} ו-{LABEL_MAP_FILE}")

if __name__ == "__main__":
    train_model()