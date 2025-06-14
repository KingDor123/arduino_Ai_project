import os
import cv2

# ==========================
# Configuration
# ==========================

NAME = "you"  # Name label for the person (used in filenames)
DATA_DIR = "data"  # Base directory to store all data
SAVE_DIR = os.path.join(DATA_DIR, NAME)  # Directory for this person's images

# Path to the Haar Cascade classifier for face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

FACE_SIZE = (200, 200)  # Target size for each face image
MAX_IMAGES = 100  # Number of face images to collect
MIN_FACE_SIZE = 100  # Ignore faces smaller than this width

# ==========================
# Create Folder if Needed
# ==========================
def create_folder_if_needed():
    # Create the directory if it doesn't already exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

# ==========================
# Collect Face Images from Webcam
# ==========================
def collect_images():
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # Open the default camera (device ID 0)
    cap = cv2.VideoCapture(0)
    count = 0  # Counter for saved images

    # Ensure camera is available
    if not cap.isOpened():
        print("Cannot access the camera.")
        return

    print("Press Space to save image, ESC to exit.")

    # Loop to capture frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        largest_face = None
        max_area = 0

        # Find the largest face (assuming it's the most relevant one)
        for (x, y, w, h) in faces:
            area = w * h
            if area > max_area and w > MIN_FACE_SIZE:
                largest_face = (x, y, w, h)
                max_area = area

            # Draw rectangle on all detected faces (for visualization)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the frame with rectangles
        cv2.imshow("Collecting Images", frame)

        # Get key press
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key to exit
            break
        elif key == 32:  # Spacebar to save image
            if largest_face:
                x, y, w, h = largest_face
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.equalizeHist(face_img)  # Improve contrast
                face_img = cv2.resize(face_img, FACE_SIZE)

                # Save the image with a sequential filename
                filename = os.path.join(SAVE_DIR, f"{NAME}_{count:03}.jpg")
                cv2.imwrite(filename, face_img)
                print(f"Saved: {filename}")
                count += 1
            else:
                print("No suitable face found.")

        # Stop collecting after reaching the max limit
        if count >= MAX_IMAGES:
            print(f"Collected {MAX_IMAGES} images. Exiting.")
            break

    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# ==========================
# Entry Point
# ==========================
if __name__ == "__main__":
    create_folder_if_needed()
    collect_images()