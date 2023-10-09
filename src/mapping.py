import numpy as np
import cv2


class MapPoint:
    def __init__(self, point3d):
        self.point3d = point3d
        self.observations = []

    def add_observation(self, frame_id, keypoint_id):
        self.observations.append((frame_id, keypoint_id))


class Map:
    def __init__(self):
        self.points = []

    def add_point(self, point3d):
        self.points.append(MapPoint(point3d))

    def add_observation(self, point_id, frame_id, keypoint_id):
        self.points[point_id].add_observation(frame_id, keypoint_id)


def triangulate_points(keypoints1, keypoints2, matches, camera_matrix, rotation_matrix1, tvec1,
                       rotation_matrix2, tvec2):
    valid_matches = []  # To store valid matches

    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx

        if query_idx < len(keypoints1) and train_idx < len(keypoints2):
            # Convert matches to homogeneous coordinates
            point1 = keypoints1[query_idx].pt + (1,)
            point2 = keypoints2[train_idx].pt + (1,)

            valid_matches.append((point1, point2))

    if not valid_matches:
        print("No valid matches for triangulation.")
        return None

    # Convert valid matches to NumPy arrays
    points1 = np.array([match[0][:2] for match in valid_matches], dtype=np.float32).T
    points2 = np.array([match[1][:2] for match in valid_matches], dtype=np.float32).T

    projection_matrix1 = np.dot(camera_matrix, np.hstack((rotation_matrix1, tvec1)))
    projection_matrix2 = np.dot(camera_matrix, np.hstack((rotation_matrix2, tvec2)))

    # Ensure data is in float32 format
    projection_matrix1 = np.array(projection_matrix1, dtype=np.float32)
    projection_matrix2 = np.array(projection_matrix2, dtype=np.float32)

    # Triangulate the 3D points
    points4d = cv2.triangulatePoints(projection_matrix1, projection_matrix2, points1, points2)
    points4d /= points4d[3]  # Normalize by the fourth coordinate

    # Extract the 3D coordinates
    points3d = points4d[:3, :].T

    return points3d
