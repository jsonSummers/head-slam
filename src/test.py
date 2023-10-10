import cv2
import numpy as np
from localization import estimate_camera_pose
from features import extract_and_match_features
from mapping import triangulate_points

def print_matches(keypoints1, keypoints2, matches):
    print("\nkeypoint1s len ", len(keypoints1))
    print("keypoint2s len ", len(keypoints2))
    print("matches len ", len(matches))
    print("\n")

    # print(dir(keypoints1[0]))
    # print("keypoint1 index ", keypoints1[0].index)
    for match in matches:
        print("match query index ", match.queryIdx)
        print("match train index ", match.trainIdx)

    print("\n")
    print(keypoints1[0])

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

        #print_matches(keypoints1, keypoints2, matches)

        # Call triangulate_points with the initial inputs
        points3d = triangulate_points(keypoints1, keypoints2, matches, camera.camera_matrix,
                                      camera.rotation_matrix, camera.translation_vector,
                                      camera.rotation_matrix, camera.translation_vector)

        print(f'Number of filtered matches: {len(matches)}')
        print(f'Number of triangulated points: {len(points3d)}')

        rotation_matrix, tvec = estimate_camera_pose(keypoints1, matches, camera, points3d)

        print(rotation_matrix)

        for point3d, match in zip(points3d, matches):
            match.point3d = point3d  # Populate point3d attribute for the match
            map.add_point(point3d)



        # Update the camera's pose
        camera.update_pose(rotation_matrix, tvec)

        # Append the current frame to the frame sequence
        frame_sequence.append(frame)

