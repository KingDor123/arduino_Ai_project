import os
import cv2

# ========== הגדרות ==========
NAME = "you"
DATA_DIR = "data"
SAVE_DIR = os.path.join(DATA_DIR, NAME)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE = (200, 200)
MAX_IMAGES = 100
MIN_FACE_SIZE = 100  # מינימום רוחב פנים לשמירה
# =============================

def create_folder_if_needed():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

def collect_images():
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    count = 0

    if not cap.isOpened():
        print("❌ לא ניתן לגשת למצלמה.")
        return

    print("📸 לחץ [Space] לשמירה, [ESC] ליציאה. נשמרות רק פנים גדולות מספיק.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ שגיאה בקריאת פריים מהמצלמה.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        largest_face = None
        max_area = 0

        for (x, y, w, h) in faces:
            area = w * h
            if area > max_area and w > MIN_FACE_SIZE:
                largest_face = (x, y, w, h)
                max_area = area
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Collecting Images", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # Space
            if largest_face:
                x, y, w, h = largest_face
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.equalizeHist(face_img)
                face_img = cv2.resize(face_img, FACE_SIZE)
                filename = os.path.join(SAVE_DIR, f"{NAME}_{count:03}.jpg")
                cv2.imwrite(filename, face_img)
                print(f"✅ נשמרה תמונה: {filename}")
                count += 1
            else:
                print("⚠️ אין פנים בגודל מתאים. התקרב או הגבר תאורה.")

        if count >= MAX_IMAGES:
            print(f"✅ הושלמו {MAX_IMAGES} תמונות. יציאה.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_folder_if_needed()
    collect_images()