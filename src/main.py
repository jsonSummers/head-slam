import os
import argparse
import pygame
import numpy as np
import cv2
import yaml
from dataloader import DatasetLoader
from visualization import display_image_with_orb_features


def main(dataset_name):
    # Initialize the dataset loader
    dataset_path = os.path.join("../data/", dataset_name)
    loader = DatasetLoader(dataset_path)
    display_image_with_orb_features(loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLAM Video Display")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset folder")
    args = parser.parse_args()
    main(args.dataset_name)