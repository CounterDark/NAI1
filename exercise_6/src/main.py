"""
Project: Smart Ad Player
Authors: Aleksander Kunkowski and Mateusz Anikiej

Description:
This script implements a "Smart Ad Player" that pauses the advertisement when the viewer is not looking at the screen.
It uses Haar Cascade classifiers for eye detection through the webcam.

Features:
- Eye Detection: Uses Haar Cascade to detect eyes in the webcam feed.
- Attention Logic: Pauses the video if no eyes (or fewer than 2) are detected.
- Video Playback: Plays a sample advertisement video.
- Real-time Feedback: Displays "PATRZY" (Looking) or "NIE PATRZY" (Not Looking) in the console and overlays "REKLAMA ZATRZYMANA" on the video when paused.

Usage:
    Run the script directly. Ensure a webcam is connected and the 'ad.mp4' file exists in the 'ads' directory.
    Press 'Esc' to exit.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def main() -> None:
    BASE_DIR = Path(__file__).resolve().parent.parent

    # ====== HAAR ======
    haar_path = BASE_DIR / "haar_cascade_files" / "haarcascade_eye.xml"
    eye_cascade = cv2.CascadeClassifier(str(haar_path))

    if eye_cascade.empty():
        raise IOError(
            f"Unable to load the eye cascade classifier xml file from {haar_path}"
        )

    # ====== KAMERA ======
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # ====== REKLAMA ======
    ad_path = BASE_DIR / "ads" / "ad.mp4"
    ad = cv2.VideoCapture(str(ad_path))
    if not ad.isOpened():
        raise IOError(f"Cannot open video file {ad_path}")

    paused: bool = False
    last_ad_frame: Optional[np.ndarray] = None

    print("Press 'Esc' to exit.")

    while True:
        # ====== KAMERA ======
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye_rects = eye_cascade.detectMultiScale(gray, 1.3, 5)

        # ====== LOGIKA UWAGI ======
        if len(eye_rects) > 1:
            looking = True
            print("PATRZY    ", end="\r")
        else:
            looking = False
            print("NIE PATRZY", end="\r")

        paused = not looking

        # ====== REKLAMA ======
        ad_frame: Optional[np.ndarray] = None

        if not paused:
            ret_ad, ad_frame = ad.read()
            if not ret_ad:
                # Loop video
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
                3,
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


if __name__ == "__main__":
    main()
