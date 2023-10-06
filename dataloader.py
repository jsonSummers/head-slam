import os
import cv2
import yaml
import numpy as np

class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_paths = self._get_image_paths()
        self.sensor_info = self._load_sensor_info()  # Load sensor info

        # Initialize the current frame index
        self.current_frame = 0

    def _get_video_length(self):
        return len(self.image_paths)

    def _get_image_paths(self):
        image_extensions = ['.png', '.jpg', '.jpeg']  # Add more if needed
        image_paths = []

        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if any(file.endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))

        return sorted(image_paths)  # Sort for sequential access

    def _load_sensor_info(self):
        sensor_yaml_path = os.path.join(self.dataset_path, 'sensor.yaml')

        if os.path.isfile(sensor_yaml_path):
            with open(sensor_yaml_path, 'r') as stream:
                try:
                    sensor_info = yaml.safe_load(stream)
                    return sensor_info
                except yaml.YAMLError as exc:
                    print(exc)

        return None

    def get_current_frame(self):
        return self.current_frame

    def get_next_frame(self):
        if self.current_frame < len(self.image_paths):
            image_path = self.image_paths[self.current_frame]
            frame = cv2.imread(image_path)

            if frame is not None:
                self.current_frame += 1
                return frame
            else:
                return self.get_next_frame()  # Skip NoneType frames
        else:
            return None

    def load_3_frames(self):
        print(self.current_frame)
        if self.current_frame < (len(self.image_paths) - 2):
            image_path = self.image_paths[self.current_frame]
            image1 = cv2.imread(image_path)
            print(self.current_frame)
            image2 = self.get_next_frame()
            print(self.current_frame)
            image3 = self.get_next_frame()
            print(self.current_frame)
            self.current_frame += -2
            return image1, image2, image3
        else:
            return None

    def is_end_of_dataset(self):
        return self.current_frame >= len(self.image_paths)

    def reset(self):
        self.current_frame = 0
