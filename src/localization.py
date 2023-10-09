import cv2
import numpy as np


class Camera:
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        self.rotation_matrix = np.eye(3)  # Initialize with identity matrix
        self.translation_vector = np.zeros((3, 1))  # Initialize with zeros

    def update_pose(self, rotation_matrix, translation_vector):
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector


def estimate_camera_pose(keypoints2, filtered_matches, camera, triangulated_points):
    # Convert keypoints to format needed by solvePnP
    obj_points = []  # 3D world points
    img_points = []  # Corresponding 2D image points
    print('beginning estimate camera')
    print(dir(filtered_matches[0]))
    print(type(filtered_matches[0]))

    for match in filtered_matches:
        # Get the index of the corresponding triangulated point
        query_idx = match.queryIdx
        train_idx = match.trainIdx

        if query_idx < len(triangulated_points):
            obj_points.append(triangulated_points[query_idx])
            img_points.append(keypoints2[train_idx].pt)

    obj_points = np.array(obj_points, dtype=np.float32)
    img_points = np.array(img_points, dtype=np.float32)

    # Use solvePnP to estimate camera pose
    _, rvec, tvec, _ = cv2.solvePnP(obj_points, img_points, camera.camera_matrix, None)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Update the camera's pose
    camera.update_pose(rotation_matrix, tvec)

    return rotation_matrix, tvec
