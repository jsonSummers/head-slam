import cv2
import numpy as np


class Camera:
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        self.rotation_matrix = np.eye(3)  # Initialize with identity matrix
        self.translation_vector = np.zeros((3, 1))  # Initialize with zeros
        self.pose_history = []  # Initialize an empty list to store camera poses

    def update_pose(self, rotation_matrix, translation_vector):
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector

        # Store the current pose in the history
        self.pose_history.append((self.rotation_matrix, self.translation_vector))

    def get_previous_pose(self):
        # Get the immediate previous pose from the history
        if len(self.pose_history) >= 2:
            previous_rotation, previous_translation = self.pose_history[-2]
            return previous_rotation, previous_translation
        else:
            # Return identity rotation and zero translation if there's no previous pose
            return np.eye(3), np.zeros((3, 1))


def estimate_camera_pose(keypoints2, matches, camera, triangulated_points):
    # Convert keypoints to format needed by solvePnP
    obj_points = []  # 3D world points
    img_points = []  # Corresponding 2D image points

    # for match in filtered_matches:
    for n in range(len(matches)):
        # Get the index of the corresponding triangulated point
        obj_points.append(triangulated_points[n])
        img_points.append(keypoints2[n].pt)

    obj_points = np.array(obj_points, dtype=np.float32)
    img_points = np.array(img_points, dtype=np.float32)

    # Use solvePnP to estimate camera pose
    retval, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera.camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)

    if retval:
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Update the camera's pose
        camera.update_pose(rotation_matrix, tvec)

        return rotation_matrix, tvec
    else:
        print("Failed to estimate camera pose.")
        return None, None
