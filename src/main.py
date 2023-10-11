import os
import argparse
# import pygame
import numpy as np
import cv2
import yaml
from dataloader import DatasetLoader
from mapping import Map
from localization import Camera
from visualization import display_image_with_orb_features
from slam import run_slam
from test import test


def main(dataset_name):
    # Initialize the dataset loader
    dataset_path = os.path.join("../data/", dataset_name)
    loader = DatasetLoader(dataset_path)
    camera_info = loader.sensor_info
    resolution = camera_info.get('resolution', [752, 480])
    rate_hz = camera_info.get('rate_hz', 20)

    map = Map()
    camera = Camera(loader.camera_matrix)

    ### INSERT LOOP HERE

    #display_image_with_orb_features(loader)
    # display_image_with_orb_features(loader)
    #test(loader, map, camera)
    # run_slam(loader, map, camera)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLAM Video Display")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset folder")
    args = parser.parse_args()
    main(args.dataset_name)
