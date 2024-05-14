import os

import numpy as np
import matplotlib.pyplot as plt
import pandas
import pickle
import keras
from keras.utils import load_img
from keras.utils import img_to_array
from sklearn.model_selection import train_test_split


IMAGE_SIZE = (256, 256)
origin_path = 'datasets/original_images'
noise_path = 'datasets/noise_images'

# Load images from direction
def load_images(data_path):
    images = []
    images_path = os.listdir(data_path)
    for img_path in images_path:
        img_full_path = os.path.join(data_path, img_path)
        img = load_img(img_full_path, IMAGE_SIZE, color_mode = "grayscale")
        img = img_to_array(img)
        img = img / 255
        images.append(img)

    images = np.array(images)

    return  images

def show_imageset(imageset):
    f, ax = plt.subplots(1, 5)
    for i in range(1, 6):
        ax[i - 1].imshow(imageset[i].reshape(IMAGE_SIZE), cmap="gray")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists('data.dat'):
        # Load origin images
        original_images = load_images(origin_path)

        # Load noise images
        noise_images = load_images(noise_path)

        print(original_images.shape)
        show_imageset(noise_images)

        # Chia data thanh train, test
        noise_train, noise_test, origin_train, origin_test = train_test_split(noise_images, original_images, test_size=0.2, random_state=42)
        with open("data_25.dat", "wb") as f:
            pickle.dump([noise_train, noise_test, origin_train, origin_test], f)

    else:
        with open("data_25.dat", "rb") as f:
            arr = pickle.load(f)
            noise_train, noise_test, origin_train, origin_test = arr[0], arr[1], arr[2], arr[3]

            print(noise_train.shape)
            show_imageset(noise_train)
