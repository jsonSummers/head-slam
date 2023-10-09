import cv2
import numpy as np


def extract_orb_features(image):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect ORB features and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors


def extract_and_match_features(image1, image2):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect ORB features and compute descriptors for both images
    keypoints1, descriptors1 = extract_orb_features(image1)
    keypoints2, descriptors2 = extract_orb_features(image2)

    # Create a BFMatcher (Brute-Force Matcher) with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors from image1 to image2
    matches = bf.match(descriptors1, descriptors2)

    ransac_threshold = 100.0  # Adjust this threshold as needed
    filtered_keypoints1, filtered_keypoints2, filtered_matches = apply_ransac(keypoints1, keypoints2, matches,
                                                                              ransac_threshold)

    # Return the keypoints and filtered_matches
    return filtered_keypoints1, filtered_keypoints2, filtered_matches


def apply_ransac(keypoints1, keypoints2, matches, ransac_threshold):
    # Create dictionaries to store associations between keypoints and matches
    keypoint_dict1 = {m.queryIdx: kp for m, kp in zip(matches, keypoints1)}
    keypoint_dict2 = {m.trainIdx: kp for m, kp in zip(matches, keypoints2)}

    # Extract matching points
    src_pts = np.float32([keypoint_dict1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoint_dict2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Use RANSAC to estimate the homography matrix
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)

    # Apply the mask to filter out outliers
    filtered_matches = [m for i, m in enumerate(matches) if mask[i][0] == 1]
    filtered_keypoints1 = [keypoint_dict1[m.queryIdx] for m in filtered_matches]
    filtered_keypoints2 = [keypoint_dict2[m.trainIdx] for m in filtered_matches]

    return filtered_keypoints1, filtered_keypoints2, filtered_matches
