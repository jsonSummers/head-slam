import pygame
import cv2
import numpy as np
from features import extract_orb_features

def display_image_with_orb_features(loader):
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