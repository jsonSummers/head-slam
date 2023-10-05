import cv2
import argparse
from dataloader import load_images_from_directory

def main(dataset_dir):
    # Load image frames from the dataset directory
    frames = load_images_from_directory(dataset_dir)

    if not frames:
        print("No frames found in the specified directory.")
        return

    # Create a window for displaying the video
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 800, 600)

    # Loop through the frames and display them as a video
    for frame in frames:
        cv2.imshow('Video', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # Press Esc to exit the video window
            break

    # Release resources and close the window
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display image frames as a video.')
    parser.add_argument('dataset_dir', type=str, help='Path to the directory containing image frames.')

    args = parser.parse_args()
    main(args.dataset_dir)
