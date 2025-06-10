import os, json, cv2, serial, time

# ========== הגדרות ==========
TRAINER_FILE = "trainer.yml"
LABEL_MAP_FILE= "label_map.json"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CAMERA_ID    = 0

SERIAL_PORT  = "/dev/cu.usbserial-210"
BAUD_RATE    = 9600

CONF_THRESH  = 50  # העלינו לסף סביר — נבדוק מה הערכים שאתה מקבל
FACE_SIZE    = (200, 200)
# =============================

def init_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"✅ Serial פתוח ב-{SERIAL_PORT}@{BAUD_RATE}")
        return ser
    except Exception as e:
        print(f"❌ שגיאת Serial: {e}")
        return None

def send_cmd(ser, cmd):
    if ser:
        ser.write(f"{cmd}\n".encode())
        print(f"➡️ נשלחה פקודה: {cmd}")

def recognize_control():
    if not os.path.exists(TRAINER_FILE) or not os.path.exists(LABEL_MAP_FILE):
        print("❌ הרץ קודם train_model.py")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    with open(LABEL_MAP_FILE, encoding="utf-8") as f:
        label_map = json.load(f)
    
    expected_label_id = 0
    expected_name = label_map.get(str(expected_label_id), "you")

    ser = init_serial()
    cap = cv2.VideoCapture(CAMERA_ID)
    detector = cv2.CascadeClassifier(CASCADE_PATH)

    last_state = None
    print("🚀 זיהוי פנים התחיל. לחץ ESC ליציאה.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        state = "other"

        for (x, y, w, h) in detector.detectMultiScale(gray, 1.1, 5):
            face = gray[y:y+h, x:x+w]
            face = cv2.equalizeHist(face)
            face = cv2.resize(face, FACE_SIZE)

            label_id, conf = recognizer.predict(face)
            print(f"[DEBUG] label={label_id}, confidence={conf:.2f}")

            if label_id == expected_label_id and conf < CONF_THRESH:
                state = "you"
                label_text = f"{expected_name} ({int(conf)})"
            else:
                state = "other"
                label_text = f"Unknown ({int(conf)})"

            color = (0,255,0) if state == "you" else (0,0,255)
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if state != last_state:
            if state == "you":
                send_cmd(ser, "OPEN")
                send_cmd(ser, "LED_OFF")
            else:
                send_cmd(ser, "LED_ON")
            last_state = state

        cv2.imshow("Face Recognition & Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()

if __name__ == "__main__":
    recognize_control()