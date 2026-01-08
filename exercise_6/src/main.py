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
import time
import numpy as np


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent

    not_looking_limit = 5.0
    not_looking_start = None

    haar_path = base_dir / "haar_cascade_files" / "haarcascade_eye.xml"
    eye_cascade = cv2.CascadeClassifier(str(haar_path))

    if eye_cascade.empty():
        raise IOError(
            f"Unable to load the eye cascade classifier xml file from {haar_path}"
        )

    # Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Ad
    ads_path = base_dir / "ads"
    ads = []
    for f in ads_path.iterdir():
        if f.suffix.lower() in [".mp4", ".avi", ".mov"]:
            print(f"Added ad {f}")
            ads.append(f)
    if len(ads) == 0:
        raise IOError("Ads folder is empty")

    current_ad_index = 0

    ad = cv2.VideoCapture(str(ads[current_ad_index]))
    if not ad.isOpened():
        raise IOError(f"Cannot open video file {ads[current_ad_index]}")

    paused: bool = False
    last_ad_frame: Optional[np.ndarray] = None

    print("Press 'Esc' to exit.")

    ret_ad, last_ad_frame = ad.read()

    while True:
        # read camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye_rects = eye_cascade.detectMultiScale(gray, 1.3, 5)

        # check if both eyes present
        if len(eye_rects) > 1:
            looking = True
            # print("PATRZY    ", end="\r")
        else:
            looking = False
            # print("NIE PATRZY", end="\r")

        current_time = time.time()

        # check countdown start
        not_looking_start = None if looking else current_time if not_looking_start is None else not_looking_start

        #reset video after set limit
        if not_looking_start is not None:
            if current_time - not_looking_start >= not_looking_limit:
                ad.release()
                ad = cv2.VideoCapture(str(ads[current_ad_index]))
                ret_ad, last_ad_frame = ad.read()
                not_looking_start = None

        paused = not looking

        ad_frame: Optional[np.ndarray] = None
        # pause logc
        if not paused:
            ret_ad, ad_frame = ad.read()
            # on ad end
            if not ret_ad:
                ad.release()
                current_ad_index = current_ad_index + 1

                if current_ad_index >= len(ads):
                    current_ad_index = 0

                ad = cv2.VideoCapture(str(ads[current_ad_index]))
                ret_ad, ad_frame = ad.read()
            last_ad_frame = ad_frame
        else:
            ad_frame = last_ad_frame

        # show text on ad
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

        # show ad window
        if ad_frame is not None:
            ad_frame_resized = cv2.resize(ad_frame, (900, 600))
            cv2.imshow("Reklama", ad_frame_resized)
        else:
            print("No ad frame")

        # show camera window
        # cv2.imshow("Kamera", frame)

        # exit on esc
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # cleanup
    cap.release()
    ad.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
