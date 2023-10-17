import cv2
import numpy as np
from localization import estimate_camera_pose
from features import extract_and_match_features
from mapping import triangulate_points
from visualization import initialize_windows, display_image_with_orb_features, display_map
import pygame


def run_slam(loader, map, camera):
    frame_sequence = []

    # Load the first frame
    first_frame = loader.get_next_frame()

    if first_frame is None:
        return  # No frames to process
    frame_sequence.append(first_frame)
    run = 1
    running = True

    resolution = first_frame.shape[1], first_frame.shape[0]
    screen, screen_video, screen_map = initialize_windows(resolution)

    while running:
        print("Frame: ", run)
        # Load the next frame
        frame = loader.get_next_frame()

        if frame is None:
            break  # End of frames

        # Extract and match features between the current frame and the previous one
        keypoints1, keypoints2, matches = extract_and_match_features(frame_sequence[-1], frame)
        prev_rotation_matrix, prev_translation_vector = camera.get_previous_pose()

        # Call triangulate_points with the initial inputs\
        points3d = triangulate_points(keypoints1, keypoints2, matches, camera.camera_matrix,
                                      prev_rotation_matrix, prev_translation_vector,
                                      camera.rotation_matrix, camera.translation_vector)
        print(points3d)
        rotation_matrix, tvec = estimate_camera_pose(keypoints1, matches, camera, points3d)

        # print(rotation_matrix)
        for point in points3d:
            map.add_point(points3d)

        # Update the camera's pose
        camera.update_pose(rotation_matrix, tvec)
        frame_sequence.append(frame)

        display_image_with_orb_features(frame, screen_video, keypoints1, matches)
        #display_map(screen_map, map, camera.camera_matrix, camera.rotation_matrix, camera.translation_vector)


        # Blit the video and map surfaces side by side onto the main Pygame window
        screen.blit(screen_video, (0, 0))
        screen.blit(screen_map, (resolution[0], 0))
        pygame.display.flip()

        run += 1