import cv2
import numpy as np
from localization import estimate_camera_pose
from features import extract_and_match_features
from mapping import triangulate_points



def run_slam(loader, map, camera):
    frame_sequence = []

    # Load the first frame
    first_frame = loader.get_next_frame()

    if first_frame is None:
        return  # No frames to process

    frame_sequence.append(first_frame)
    running = True
    while running:
        # Load the next frame
        frame = loader.get_next_frame()

        if frame is None:
            break  # End of frames

        # Extract and match features between the current frame and the previous one
        keypoints1, keypoints2, matches = extract_and_match_features(frame_sequence[-1], frame)

        # Call triangulate_points with the initial inputs
        points3d = triangulate_points(keypoints1, keypoints2, matches, camera.camera_matrix,
                                      camera.rotation_matrix, camera.translation_vector,
                                      camera.rotation_matrix, camera.translation_vector)

        rotation_matrix, tvec = estimate_camera_pose(keypoints1, matches, camera, points3d)

        print(rotation_matrix)
        for point in points3d:
            map.add_point(points3d)

        # Update the camera's pose
        camera.update_pose(rotation_matrix, tvec)
        break

        # Append the current frame to the frame sequence
        frame_sequence.append(frame)
