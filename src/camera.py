import cv2
import numpy as np


class Camera:
    def __init__(self, camera_info, camera_matrix):
        self.camera_info = camera_info
        self.camera_matrix = camera_matrix
        self.rotation_matrix = np.eye(3)  # Initialize with identity matrix
        self.translation_vector = np.zeros((3, 1))  # Initialize with zeros
        self.pose_history = []  # Initialize an empty list to store camera poses
        self.resolution = self.camera_info.get('resolution', [752, 480])
        self.rate_hz = self.camera_info.get('rate_hz', 20)

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
