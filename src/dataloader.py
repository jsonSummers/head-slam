import os
import cv2

class ImageLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(('.jpg', '.png'))])
        self.current_index = 0

    def get_next_image(self):
        if self.current_index < len(self.image_paths):
            image_path = self.image_paths[self.current_index]
            image = cv2.imread(image_path)
            self.current_index += 1
            return image
        else:
            return None
