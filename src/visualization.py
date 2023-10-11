import pygame
import cv2
import numpy as np
from features import extract_orb_features
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from features import extract_orb_features


def initialize_windows(resolution):
    # Initialize Pygame windows for video and map visualization
    pygame.init()
    screen_video = pygame.display.set_mode(resolution)
    pygame.display.set_caption("SLAM Video")

    screen_map = pygame.display.set_mode(resolution)
    pygame.display.set_caption("SLAM Map")

    return screen_video, screen_map



def display_image_with_orb_features(frame, screen_video, keypoints, matches):
    # Visualize the video frame with ORB features and matches
    frame_with_features = frame.copy()

    # Draw ORB keypoints on the frame
    frame_with_features = cv2.drawKeypoints(frame_with_features, keypoints, None, color=(0, 255, 0), flags=0)

    # Draw lines to represent feature matches
    for match in matches:
        pt1 = (int(keypoints[match.queryIdx].pt[0]), int(keypoints[match.queryIdx].pt[1]))
        pt2 = (int(keypoints[match.trainIdx].pt[0]), int(keypoints[match.trainIdx].pt[1]))
        cv2.line(frame_with_features, pt1, pt2, (0, 0, 255), 1)  # Draw a red line for each match

    frame_with_features = cv2.cvtColor(frame_with_features, cv2.COLOR_BGR2RGB)
    frame_with_features = pygame.surfarray.make_surface(np.rot90(frame_with_features))

    screen_video.blit(frame_with_features, (0, 0))
    pygame.display.flip()


zoom = 1.0
rotation_angle = 0


# Function to update the map visualization
def display_map(map_image, screen_map):
    global zoom, rotation_angle  # Access global variables

    # Capture user input events
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                zoom += 0.1  # Increase zoom level
            elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                zoom -= 0.1  # Decrease zoom level
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:
                zoom += 0.1  # Scroll up to zoom in
            elif event.button == 5:
                zoom -= 0.1  # Scroll down to zoom out

    # Update the map with zoom and rotation transformations
    rotated_map = pygame.transform.rotozoom(map_image, rotation_angle, zoom)

    screen_map.blit(rotated_map, (0, 0))
    pygame.display.flip()

























def display_image_with_orb_features__old(loader):
    camera_info = loader.sensor_info
    if camera_info is None:
        return
    resolution = camera_info.get('resolution', [752, 480])
    rate_hz = camera_info.get('rate_hz', 20)

    pygame.init()
    screen = pygame.display.set_mode(resolution)
    pygame.display.set_caption("SLAM Video")

    running = True
    clock = pygame.time.Clock()

    while running:
        frame = loader.get_next_frame()

        if frame is not None:
            frame_with_features = frame.copy()

            # Convert the frame to grayscale for ORB detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Extract ORB keypoints and descriptors using the feature extraction function
            keypoints, descriptors = extract_orb_features(gray_frame)

            # Draw ORB keypoints on the frame
            frame_with_features = cv2.drawKeypoints(frame_with_features, keypoints, None, color=(0, 255, 0), flags=0)

            frame_with_features = cv2.cvtColor(frame_with_features, cv2.COLOR_BGR2RGB)
            frame_with_features = pygame.surfarray.make_surface(np.rot90(frame_with_features))

            screen.blit(frame_with_features, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            clock.tick(rate_hz)

        else:
            running = False

    pygame.quit()
