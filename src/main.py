import os
import argparse
import pygame
import numpy as np
import cv2
import yaml
from dataloader import DatasetLoader

def load_camera_info(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            camera_info = yaml.safe_load(stream)
            return camera_info
        except yaml.YAMLError as exc:
            print(exc)
            return None

def main(dataset_name, yaml_file):
    # Load camera information from the YAML file
    camera_info = load_camera_info(yaml_file)
    if camera_info is None:
        return

    resolution = camera_info.get('resolution', [752, 480])
    rate_hz = camera_info.get('rate_hz', 20)

    # Set up Pygame
    pygame.init()
    screen = pygame.display.set_mode(resolution)
    pygame.display.set_caption("SLAM Video")

    # Initialize the dataset loader
    dataset_path = os.path.join("../data/", dataset_name)
    loader = DatasetLoader(dataset_path)

    running = True

    clock = pygame.time.Clock()  # Create a clock to control frame rate

    while running:
        frame = loader.get_next_frame()

        if frame is not None:
            # Convert the frame to a Pygame surface
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = pygame.surfarray.make_surface(np.rot90(frame))

            # Display the frame
            screen.blit(frame, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            clock.tick(rate_hz)  # Limit the frame rate

        else:
            running = False

    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLAM Video Display")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset folder")
    parser.add_argument("yaml_file", type=str, help="Path to the YAML camera info file")
    args = parser.parse_args()
    main(args.dataset_name, args.yaml_file)