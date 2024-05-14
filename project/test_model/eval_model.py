import os.path
import pickle

import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from keras.utils import load_img, img_to_array
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim

EVAL_DATASET = True
IMAGE_SIZE = (256, 256,1)
noise_path = "BSDS500/noise"
origin_path = "BSDS500/origin"

def PSRN(img_origin, img_reconstructed):
    # Tính toán PSNR
    psnr = peak_signal_noise_ratio(img_origin, img_reconstructed)
    # print("PSNR:", psnr)
    return psnr

def SSIM(img_origin, img_reconstructed):
    # Tính SSIM
    ssim_score, _ = ssim(img_origin/1.0, img_reconstructed/1.0, full=True, data_range=1.0)
    # print("SSIM score:", ssim_score)
    return ssim_score


def MAE(img_origin, img_reconstructed):
    mae = np.mean(np.abs(img_origin.astype(np.float32) - img_reconstructed.astype(np.float32)))
    # print("MAE score:", mae)
    return mae

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


if __name__ == '__main__':
    if EVAL_DATASET:
        print("Eval tren tap dataset")
        # Doc du lieu train, test tu file
        with open("data_25.dat", "rb") as f:
            arr = pickle.load(f)
            noise_test, origin_test = arr[1], arr[3]
    else:
        print("Eval tren tap BSDS500")
        noise_test = load_images(noise_path)
        origin_test = load_images(origin_path)

    # Load model
    model = load_model("aev2_enhan_best.h5", compile=False)

    pred_images = model.predict(noise_test)

    test_size = noise_test.shape[0]

    psrn = 0
    mae = 0


    for i in range(test_size):
        psrn += PSRN(origin_test[i], pred_images[i])
        mae += MAE(origin_test[i], pred_images[i])



    psrn_score = psrn/test_size
    mae_score = mae/test_size


    print("Mean psrn: ", psrn_score)
    print("Mean mae: ", mae_score)


    # Chon random 5 anh de khu nhieu
    s_id = 25
    e_id = 30

    pred_images = model.predict(noise_test[s_id: e_id])
    ori_images = origin_test[s_id: e_id]

    # Ve len man hinh de kiem tra
    for i in range(s_id, e_id):
        print("PSRN between GT image and predict image %d: " %i,PSRN(ori_images[i -s_id], pred_images[i - s_id]))
        new_image = cv2.blur(noise_test[i], (3, 3))
        plt.figure(figsize=(8, 3))
        plt.subplot(141)
        plt.imshow(pred_images[i - s_id].reshape(IMAGE_SIZE), cmap='gray')
        plt.title('Model')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(142)
        plt.imshow(new_image, cmap='gray')
        plt.title('Blur OpenCV (K3)')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(143)
        plt.imshow(ori_images[i -s_id].reshape(IMAGE_SIZE), cmap='gray')
        plt.title('Origin image')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(144)
        plt.imshow(noise_test[i], cmap='gray')
        plt.title('Noise image')
        plt.xticks([])
        plt.yticks([])

        plt.show()
