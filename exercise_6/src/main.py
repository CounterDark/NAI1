import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ====== HAAR ======
eye_cascade = cv2.CascadeClassifier(
    str(BASE_DIR / "haar_cascade_files" / "haarcascade_eye.xml")
)

if eye_cascade.empty():
    raise IOError("Unable to load the eye cascade classifier xml file")

# ====== KAMERA ======
cap = cv2.VideoCapture(0)

# ====== REKLAMA ======
ad = cv2.VideoCapture(str(BASE_DIR / 'ads' / "ad.mp4"))

paused = False
last_ad_frame = None

while True:
    # ====== KAMERA ======
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_rects = eye_cascade.detectMultiScale(gray, 1.3, 5)

    # ====== LOGIKA UWAGI ======
    if len(eye_rects) > 1:
        looking = True
        print("PATRZY", end="\r")
    else:
        looking = False
        print("NIE PATRZY", end="\r")

    paused = not looking

    # ====== REKLAMA ======
    if not paused:
        ret_ad, ad_frame = ad.read()
        if not ret_ad:
            ad.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_ad, ad_frame = ad.read()
        last_ad_frame = ad_frame
    else:
        ad_frame = last_ad_frame

    # ====== INFO NA EKRANIE ======
    if paused and ad_frame is not None:
        cv2.putText(
            ad_frame,
            "REKLAMA ZATRZYMANA",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # ====== OKNA ======
    if ad_frame is not None:
        cv2.imshow("Reklama", ad_frame)

    cv2.imshow("Kamera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
ad.release()
cv2.destroyAllWindows()
