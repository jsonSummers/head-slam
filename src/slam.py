import cv2
import numpy as np
from feature_extraction import extract_orb_features

def run_slam(loader):
    camera_info = loader.sensor_info
    resolution = camera_info.get('resolution', [752, 480])
    rate_hz = camera_info.get('rate_hz', 20)
    print("camera information loaded")
    running = True

    while running:
        frame1, frame2, frame3 = loader.load_3_frames()
        print("3 frames loaded")
        loader.get_next_frame()
        frame4, frame5, frame6 = loader.load_3_frames()

        running = False
'''
        frame = loader.get_next_frame()

        if frame is not None:
            frame_with_features = frame.copy()

            # Convert the frame to grayscale for ORB detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Extract ORB keypoints and descriptors using the feature extraction function
            keypoints, descriptors = extract_orb_features(gray_frame)
'''