import os
import cv2
import numpy as np

class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_paths = self._get_image_paths()

        # Initialize the current frame index
        self.current_frame = 0

    def _get_image_paths(self):
        image_extensions = ['.png', '.jpg', '.jpeg']  # Add more if needed
        image_paths = []

        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if any(file.endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))

        return sorted(image_paths)  # Sort for sequential access

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

    def is_end_of_dataset(self):
        return self.current_frame >= len(self.image_paths)

    def reset(self):
        self.current_frame = 0
