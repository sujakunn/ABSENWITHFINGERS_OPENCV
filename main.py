import cv2
import csv
from datetime import datetime
from hand_detector import HandDetector
from face_recognition import FaceDetector

cap = cv2.VideoCapture(0)

hand_detector = HandDetector()
face_detector = FaceDetector()

print("SMART ATTENDANCE SYSTEM")
print("Angkat 5 jari untuk absen")
print("Tekan Q untuk keluar")

attendance_done = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    faces = face_detector.detect(frame)
    hand_result = hand_detector.detect(frame)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "syuja", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            fingers = hand_detector.count_fingers(hand_landmarks)

            cv2.putText(frame, f"Finger: {fingers}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            if fingers == 5 and len(faces) > 0 and not attendance_done:
                name = "syuja"
                time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with open("attendance.csv", "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([name, time])

                attendance_done = True
                print("ABSEN BERHASIL")

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
