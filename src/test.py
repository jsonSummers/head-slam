import cv2
import numpy as np
from localization import estimate_camera_pose
from features import extract_and_match_features
from mapping import triangulate_points

def test(loader, map, camera):
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

        #print(dir(keypoints1))
        #print(type(keypoints1[0]))
        print(dir(matches[0]))
        print(type(matches[0]))
        print("break\n")
        initial_rotation = np.eye(3)  # Identity rotation matrix
        initial_translation = np.zeros((3, 1))  # Zero translation vector

        # Call triangulate_points with the initial inputs
        points3d = triangulate_points(keypoints1, keypoints2, matches, camera.camera_matrix,
                                      initial_rotation, initial_translation,
                                      initial_rotation, initial_translation)

        rotation_matrix, tvec = estimate_camera_pose(keypoints1, matches, camera, points3d)

        running = False

        for point3d, match in zip(points3d, matches):
            match.point3d = point3d  # Populate point3d attribute for the match
            map.add_point(point3d)

        # Update the camera's pose
        camera.update_pose(rotation_matrix, tvec)

        # Append the current frame to the frame sequence
        frame_sequence.append(frame)

