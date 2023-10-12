import pygame
import cv2
import numpy as np
from features import extract_orb_features
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from features import extract_orb_features


def initialize_windows(resolution):
    # Initialize Pygame window for both video and map visualization side by side
    pygame.init()
    screen_width = resolution[0] * 2  # Double the width to fit video and map side by side
    screen_height = resolution[1]
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("SLAM Video and Map")

    # Create two surfaces for video and map visualization
    screen_video = pygame.Surface((resolution[0], resolution[1]))
    screen_map = pygame.Surface((resolution[0], resolution[1]))

    return screen, screen_video, screen_map



def display_image_with_orb_features(frame, screen_video, keypoints, matches):
    # Visualize the video frame with ORB features
    frame_with_features = frame.copy()

    for match in matches:
        # Get the indices of matched keypoints
        idx1 = match.queryIdx
        idx2 = match.trainIdx

        # Get the corresponding keypoints
        kp1 = keypoints[idx1]
        kp2 = keypoints[idx2]

        # Draw matched keypoints in red
        color = (0, 0, 255)  # Red
        frame_with_features = cv2.drawMarker(frame_with_features, tuple(map(int, kp1.pt)), color,
                                             markerType=cv2.MARKER_DIAMOND)

    frame_with_features = cv2.drawKeypoints(frame_with_features, keypoints, None, color=(0, 255, 0), flags=0)

    frame_with_features = cv2.cvtColor(frame_with_features, cv2.COLOR_BGR2RGB)
    frame_with_features = pygame.surfarray.make_surface(np.rot90(frame_with_features))

    screen_video.blit(frame_with_features, (0, 0))
    pygame.display.flip()


zoom = 1.0
rotation_angle = 0


# Function to update the map visualization
def display_map(screen_map, map, camera_matrix, rotation_matrix, translation_vector):
    # Create a new blank surface to draw the map
    map_surface = pygame.Surface(screen_map.get_size())

    # Clear the map surface with a background color (e.g., white)
    map_surface.fill((255, 255, 255))

    # Draw the map points
    for point in map.points:
        for observation in point.observations:
            frame_id, keypoint_id = observation
            # Calculate the map position where the point should be drawn
            x, y = map_point_to_map_coordinates(point.point3d, camera_matrix, rotation_matrix, translation_vector)
            pygame.draw.circle(map_surface, (0, 0, 255), (x, y), 3)  # Blue circle for map point

    # Blit the map surface onto the map window
    screen_map.blit(map_surface, (0, 0))
    pygame.display.flip()


def map_point_to_map_coordinates(point3d, camera_matrix, rotation_matrix, translation_vector):
    # Project the 3D point to 2D using camera projection
    point3d_homogeneous = np.append(point3d, 1)  # Convert to homogeneous coordinates
    projection_matrix = np.dot(camera_matrix, np.hstack((rotation_matrix, translation_vector)))
    point2d_homogeneous = np.dot(projection_matrix, point3d_homogeneous)

    # Normalize by the third coordinate
    point2d_normalized = point2d_homogeneous / point2d_homogeneous[2]

    # Extract the 2D map coordinates
    x, y = point2d_normalized[:2]

    return int(x), int(y)






















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
