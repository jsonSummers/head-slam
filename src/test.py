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


def run_test(loader, map, camera):
    frame_sequence = []

    # Load the first frame
    first_frame = loader.get_next_frame()

    if first_frame is None:
        return  # No frames to process

    frame_sequence.append(first_frame)

    # Load the second frame
    second_frame = loader.get_next_frame()

    if second_frame is None:
        return  # No second frame to process

    # Extract and match features between the first and second frame
    keypoints1, keypoints2, matches = extract_and_match_features(frame_sequence[0], second_frame)

    print_matches(keypoints1, keypoints2, matches)

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

    # Display matches between the first and second frame
    display_matches(frame_sequence[0], second_frame, keypoints1, keypoints2, matches)

    # Append the second frame to the frame sequence
    frame_sequence.append(second_frame)

import pygame
import numpy as np
from features import extract_and_match_features
import random

def display_matches(frame1, frame2, keypoints1, keypoints2, matches):
    # Combine the two frames side by side
    height, width, _ = frame1.shape
    combined_frame = np.zeros((height, 2 * width, 3), dtype=np.uint8)
    combined_frame[:, :width] = frame1
    combined_frame[:, width:] = frame2

    # Create a Pygame window
    resolution = (2 * width, height)
    screen = pygame.display.set_mode(resolution)
    pygame.display.set_caption("Matches")

    # Define a list of distinct colors for the lines
    distinct_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    # Draw lines between matched keypoints with distinct colors
    for i, match in enumerate(matches):
        # Cycle through the list of distinct colors
        color = distinct_colors[i % len(distinct_colors)]

        pt1 = (int(keypoints1[match.queryIdx].pt[0]), int(keypoints1[match.queryIdx].pt[1]))
        pt2 = (int(keypoints2[match.trainIdx].pt[0]) + width, int(keypoints2[match.trainIdx].pt[1]))
        cv2.line(combined_frame, pt1, pt2, color, 1)

    # Convert the combined frame to a Pygame surface
    combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
    combined_frame = pygame.surfarray.make_surface(np.rot90(combined_frame))

    screen.blit(combined_frame, (0, 0))
    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

