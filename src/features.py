import cv2
import numpy as np


def extract_orb_features(image):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect ORB features and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors


def extract_and_match_features(image1, image2, image3):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect ORB features and compute descriptors for all three frames
    keypoints1, descriptors1 = extract_orb_features(image1)
    keypoints2, descriptors2 = extract_orb_features(image2)
    keypoints3, descriptors3 = extract_orb_features(image3)

    # Create a BFMatcher (Brute-Force Matcher) with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches12 = bf.match(descriptors1, descriptors2)
    matches23 = bf.match(descriptors2, descriptors3)

    ransac_threshold = 5.0  # Adjust this threshold as needed
    keypoints1, keypoints2, matches12 = apply_ransac(keypoints1, keypoints2, matches12, ransac_threshold)
    keypoints2, keypoints3, matches23 = apply_ransac(keypoints2, keypoints3, matches23, ransac_threshold)

    # Return the keypoints and matches
    return keypoints1, keypoints2, keypoints3, matches12, matches23


def apply_ransac(keypoints1, keypoints2, matches, ransac_threshold):
    # Convert keypoints to format needed by findHomography
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Use findHomography with RANSAC to estimate a transformation matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)

    # Apply mask to keep only inliers
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i] == 1]

    # Update keypoints based on inliers
    keypoints1 = [keypoints1[match.queryIdx] for match in inlier_matches]
    keypoints2 = [keypoints2[match.trainIdx] for match in inlier_matches]

    return keypoints1, keypoints2, inlier_matches
