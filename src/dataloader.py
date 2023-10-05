import os
import cv2

def load_images_from_directory(directory):
    """
    Load image frames from a directory.

    Args:
        directory (str): Path to the directory containing image frames.

    Returns:
        List of image frames.
    """
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            img_path = os.path.join(directory, filename)
            image = cv2.imread(img_path)
            if image is not None:
                images.append(image)
    return images