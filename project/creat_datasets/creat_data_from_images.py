import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from creat_non_uniformity import *

import keras
from keras.utils import load_img
from keras.utils import img_to_array


WIDTH = 256
HEIGHT = 256
START_INDEX = 1


def creat_data_from_images(images_dir, origin_path, noise_path):
    # Check origin and noise path
    if not os.path.exists(origin_path):
        print("Creat origin path!")
        os.makedirs(origin_path)

    if not os.path.exists(noise_path):
        print("Creat noise path!")
        os.makedirs(noise_path)

    # Load images
    images_path = os.listdir(images_dir)

    image_count = START_INDEX

    for img_path in images_path:
        img_full_path = os.path.join(images_dir, img_path)

        origin_img = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
        origin_img = cv2.resize(origin_img, (HEIGHT, WIDTH)) # Cac chieu cua anh nguoc so voi keras

        noise_img = creat_noise_image(origin_img)

        # Save images
        origin_images_path = os.path.join(origin_path, f"origin_{image_count}.jpg")
        cv2.imwrite(origin_images_path, origin_img)

        noise_images_path = os.path.join(noise_path, f"noise_{image_count}.jpg")
        cv2.imwrite(noise_images_path, noise_img)

        image_count += 1

        print(f"Saved origin {image_count} as {origin_images_path}")
        print(f"Saved noise  {image_count} as {noise_images_path}")

if __name__ == "__main__":
    # # Creat data to train
    # images_dir = 'val2017'
    # origin_path = 'datasets/original_images'
    # noise_path = 'datasets/noise_images'
    # creat_data_from_images(images_dir, origin_path, noise_path)

    # Creat data to test
    images_dir = 'BSD500'
    origin_path = 'BSDS500/origin'
    noise_path = 'BSDS500/noise'
    creat_data_from_images(images_dir, origin_path, noise_path)
