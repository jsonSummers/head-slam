import cv2
import numpy as np
from localization import estimate_camera_pose
from features import extract_and_match_features
from mapping import triangulate_points
from features import apply_ransac


def run_slam(loader, map, camera):
    frame_sequence = []

    # Load the first frame
    first_frame = loader.get_next_frame()

    if first_frame is None:
        return  # No frames to process

    frame_sequence.append(first_frame)

    while True:
        # Load the next frame
        frame = loader.get_next_frame()

        if frame is None:
            break  # End of frames

        # Extract and match features between the current frame and the previous one
        keypoints1, keypoints2, matches = extract_and_match_features(frame_sequence[-1], frame)

        points3d = triangulate_points(keypoints2, matches, camera.camera_matrix,
                                      np.eye(3), np.zeros((3, 1)), np.eye(3), np.zeros((3, 1)))

        # Estimate camera pose
        rotation_matrix, tvec = estimate_camera_pose(keypoints1, keypoints2, matches, camera)

        print("3d  done")
        print(len(points3d))
        # Add the triangulated points to the map
        for point3d, match in zip(points3d, matches):
            match.point3d = point3d  # Populate point3d attribute for the match
            map.add_point(point3d)

        # Update the camera's pose
        camera.update_pose(rotation_matrix, tvec)

        # Append the current frame to the frame sequence
        frame_sequence.append(frame)