import cv2
import numpy as np


def extract_orb_features(image):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect ORB features and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors


def extract_and_match_features(image1, image2):
    keypoints1, descriptors1, keypoints2, descriptors2 = extract_2frame_features(image1, image2)
    filtered_keypoints1, filtered_keypoints2, filtered_matches = match_2frame_features(keypoints1, descriptors1,
                                                                                       keypoints2, descriptors2)
    return filtered_keypoints1, filtered_keypoints2, filtered_matches


def extract_2frame_features(image1, image2):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect ORB features and compute descriptors for both images
    keypoints1, descriptors1 = extract_orb_features(image1)
    keypoints2, descriptors2 = extract_orb_features(image2)

    return keypoints1, descriptors1, keypoints2, descriptors2


def match_2frame_features(keypoints1, descriptors1, keypoints2, descriptors2):
    # Create a BFMatcher (Brute-Force Matcher) with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors from image1 to image2
    matches = bf.match(descriptors1, descriptors2)

    # Filter matches based on valid keypoints
    valid_matches = []
    for match in matches:
        if match.queryIdx < len(keypoints1) and match.trainIdx < len(keypoints2):
            valid_matches.append(match)

    # Check for invalid keypoints and log them
    invalid_matches = [match for match in matches if match not in valid_matches]
    if invalid_matches:
        print(f"Invalid matches: {len(invalid_matches)}")

    ransac_threshold = 1.0  # Adjust this threshold as needed
    filtered_keypoints1, filtered_keypoints2, filtered_matches = apply_ransac(
        keypoints1, keypoints2, valid_matches, ransac_threshold)

    # Return the keypoints and filtered_matches
    return filtered_keypoints1, filtered_keypoints2, filtered_matches


def apply_ransac(keypoints1, keypoints2, valid_matches, ransac_threshold):
    # Extract matching points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in valid_matches]).reshape(-1, 1, 2)

    # Use RANSAC to estimate the fundamental matrix
    fundamental_matrix, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, ransac_threshold, 0.99)

    # Apply the mask to filter out outliers
    filtered_matches = [m for i, m in enumerate(valid_matches) if mask[i][0] == 1]
    filtered_keypoints1 = [keypoints1[m.queryIdx] for m in filtered_matches]
    filtered_keypoints2 = [keypoints2[m.trainIdx] for m in filtered_matches]

    return filtered_keypoints1, filtered_keypoints2, filtered_matches
