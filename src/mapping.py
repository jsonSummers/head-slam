import numpy as np
import cv2


class MapPoint:
    def __init__(self, point3d):
        self.point3d = point3d
        self.observations = []

    def add_observation(self, frame_id, keypoint_id):
        self.observations.append((frame_id, keypoint_id))

    # def correct_point(self):


class Map:
    def __init__(self):
        self.points = []

    def add_point(self, point3d):
        self.points.append(MapPoint(point3d))

    def add_observation(self, point_id, frame_id, keypoint_id):
        self.points[point_id].add_observation(frame_id, keypoint_id)


def triangulate_points_old(keypoints1, keypoints2, matches, camera_matrix, rotation_matrix1, tvec1,
                           rotation_matrix2, tvec2):
    valid_matches = []  # To store valid matches

    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx

        # Convert valid matches to NumPy arrays
        point1 = np.array(keypoints1[query_idx].pt, dtype=np.float32)
        point2 = np.array(keypoints2[train_idx].pt, dtype=np.float32)

        valid_matches.append((point1, point2))

    if len(valid_matches) < 8:
        # Not enough valid matches for triangulation
        print("No valid matches for triangulation.")
        return None

    # Convert valid matches to NumPy arrays
    points1 = np.array([match[0] for match in valid_matches], dtype=np.float32).T
    points2 = np.array([match[1] for match in valid_matches], dtype=np.float32).T

    # Ensure camera_matrix is a matrix
    camera_matrix = np.array(camera_matrix, dtype=np.float32)

    # Projection matrices
    projection_matrix1 = np.dot(camera_matrix, np.hstack((rotation_matrix1, tvec1)))
    projection_matrix2 = np.dot(camera_matrix, np.hstack((rotation_matrix2, tvec2)))

    # Ensure projection matrices are matrices
    projection_matrix1 = np.array(projection_matrix1, dtype=np.float32)
    projection_matrix2 = np.array(projection_matrix2, dtype=np.float32)

    # Triangulate the 3D points
    points4d = cv2.triangulatePoints(projection_matrix1, projection_matrix2, points1, points2)
    points4d /= points4d[3]  # Normalize by the fourth coordinate

    # Extract the 3D coordinates
    points3d = points4d[:3, :].T

    return points3d


def triangulate_points(keypoints1, keypoints2, matches, camera_matrix, rotation_matrix1, tvec1, rotation_matrix2, tvec2):
    # Convert keypoints to numpy arrays
    keypoints1 = np.array([kp.pt for kp in keypoints1], dtype=np.float32)
    keypoints2 = np.array([kp.pt for kp in keypoints2], dtype=np.float32)

    # Prepare projection matrices for both cameras
    projection_matrix1 = np.dot(camera_matrix, np.hstack((rotation_matrix1, tvec1)))
    projection_matrix2 = np.dot(camera_matrix, np.hstack((rotation_matrix2, tvec2)))

    # Extract the matched points
    matched_points1 = np.array([keypoints1[match.queryIdx] for match in matches])
    matched_points2 = np.array([keypoints2[match.trainIdx] for match in matches])

    # Perform triangulation
    points_4d_homogeneous = cv2.triangulatePoints(projection_matrix1, projection_matrix2,
                                                  matched_points1.T, matched_points2.T)

    # Convert homogeneous coordinates to 3D coordinates
    points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]

    return points_3d.T


def normalize_3d_points(points3d):
    # Check for NaN values in the input
    if np.isnan(points3d).any():
        print("Input contains NaN values.")
        return points3d

    # Check for all zeros (potentially problematic data)
    if np.all(np.isclose(points3d, 0.0)):
        print("Input contains all zero values.")
        return points3d

    # Calculate the mean of 3D points
    mean = np.mean(points3d, axis=0)

    # Calculate the standard deviation of 3D points
    std_dev = np.std(points3d, axis=0)

    # Check for zero standard deviation (potentially problematic data)
    if np.all(np.isclose(std_dev, 0.0)):
        print("Zero standard deviation detected.")
        return points3d

    # Normalize the points
    normalized_points3d = (points3d - mean) / std_dev

    return normalized_points3d