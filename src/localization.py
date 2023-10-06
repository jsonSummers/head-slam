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


def estimate_camera_pose(keypoints1, keypoints2, matches, camera):
    # Convert keypoints to format needed by solvePnP
    obj_points = []  # 3D world points
    img_points = []  # Corresponding 2D image points

    for match in matches:
        obj_points.append(match.point3d)  # You'll need to populate point3d during triangulation
        img_points.append(keypoints2[match.trainIdx].pt)

    obj_points = np.array(obj_points, dtype=np.float32)
    img_points = np.array(img_points, dtype=np.float32)

    # Use solvePnP to estimate camera pose
    _, rvec, tvec, _ = cv2.solvePnP(obj_points, img_points, camera.camera_matrix, None)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Update the camera's pose
    camera.update_pose(rotation_matrix, tvec)

    return rotation_matrix, tvec
