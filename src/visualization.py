import pygame
import cv2
import numpy as np

def display_images_as_video(loader):
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = pygame.surfarray.make_surface(np.rot90(frame))

            screen.blit(frame, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            clock.tick(rate_hz)

        else:
            running = False

    pygame.quit()