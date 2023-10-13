import cv2
import numpy as np


def extract_orb_features(image):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect ORB features and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors


def extract_2frame_features(image1, image2):
    # Detect ORB features and compute descriptors for both images
    keypoints1, descriptors1 = extract_orb_features(image1)
    keypoints2, descriptors2 = extract_orb_features(image2)

    return keypoints1, descriptors1, keypoints2, descriptors2


def extract_and_match_features(image1, image2):
    keypoints1, descriptors1, keypoints2, descriptors2 = extract_2frame_features(image1, image2)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # You can adjust the RANSAC parameters (e.g., 3 for minimum number of points, 0.8 for the maximum reprojection
    # error).
    ransac_threshold = 3.0
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
    good_matches = [matches[i] for i in range(len(matches)) if mask[i] == 1]

    return keypoints1, keypoints2, good_matches
