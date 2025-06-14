import os, json, cv2, time
import serial  # Module for serial communication with Arduino

# ==========================
# Configuration
# ==========================

TRAINER_FILE = "trainer.yml"  # Path to the trained face recognizer model
LABEL_MAP_FILE = "label_map.json"  # JSON file mapping label IDs to names
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # Haar Cascade for face detection
CAMERA_ID = 0  # Default camera ID

SERIAL_PORT = "/dev/cu.usbserial-210"  # Serial port connected to Arduino
BAUD_RATE = 9600  # Baud rate for serial communication

CONF_THRESH = 50  # Confidence threshold: lower means stricter recognition
FACE_SIZE = (200, 200)  # Standard size to resize detected face regions

# ==========================
# Initialize Serial Connection
# ==========================
def init_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for the serial connection to stabilize
        print(f"✅ Serial open in {SERIAL_PORT}@{BAUD_RATE}")
        return ser
    except Exception as e:
        print(f"❌ Serial error: {e}")
        return None

# ==========================
# Send Command to Arduino
# ==========================
def send_cmd(ser, cmd):
    if ser:
        ser.write(f"{cmd}\n".encode())
        print(f"➡️ Sent command: {cmd}")

# ==========================
# Main Face Recognition and Arduino Control Logic
# ==========================
def recognize_control():
    # Ensure required files exist
    if not os.path.exists(TRAINER_FILE) or not os.path.exists(LABEL_MAP_FILE):
        print("Please run train_model.py first")
        return

    # Load the trained face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    # Load label-to-name mapping
    with open(LABEL_MAP_FILE, encoding="utf-8") as f:
        label_map = json.load(f)

    expected_label_id = 0  # Assuming your face has label 0
    expected_name = label_map.get(str(expected_label_id), "you")

    # Open serial connection
    ser = init_serial()

    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_ID)

    # Load the face detector
    detector = cv2.CascadeClassifier(CASCADE_PATH)

    last_state = None  # To keep track of last recognized state
    print("Face recognition started. Press ESC to exit.")

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        state = "other"

        # Detect faces in the frame
        for (x, y, w, h) in detector.detectMultiScale(gray, 1.1, 5):
            face = gray[y:y+h, x:x+w]
            face = cv2.equalizeHist(face)
            face = cv2.resize(face, FACE_SIZE)

            # Predict the face label and confidence
            label_id, conf = recognizer.predict(face)
            print(f"[DEBUG] label={label_id}, confidence={conf:.2f}")

            # Check if the detected face matches the expected one
            if label_id == expected_label_id and conf < CONF_THRESH:
                state = "you"
                label_text = f"{expected_name} ({int(conf)})"
            else:
                state = "other"
                label_text = f"Unknown ({int(conf)})"

            # Draw rectangle and label
            color = (0, 255, 0) if state == "you" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # If state changed, send corresponding command to Arduino
        if state != last_state:
            if state == "you":
                send_cmd(ser, "OPEN")
                send_cmd(ser, "LED_OFF")
            else:
                send_cmd(ser, "LED_ON")
            last_state = state

        # Show the video with detection overlay
        cv2.imshow("Face Recognition & Control", frame)

        # Exit if ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()

# ==========================
# Entry Point
# ==========================
if __name__ == "__main__":
    recognize_control()