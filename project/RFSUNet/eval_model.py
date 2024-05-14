import pickle
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim

# ==============================================================================================================#
IMAGE_SIZE = (256, 256,1)
# ==============================================================================================================#

def PSRN(image1, image2):
    # Tính toán PSNR
    psnr = peak_signal_noise_ratio(image1, image2)
    # print("PSNR:", psnr)
    return psnr
# ==============================================================================================================#

def SSIM(image1, image2):
    # Tính SSIM
    ssim_score, _ = ssim(image1, image2, full=True)
    # print("SSIM score:", ssim_score)
    return ssim_score
# ==============================================================================================================#

def MAE(image1, image2):
    mae = np.mean(np.abs(image1.astype(np.float32) - image2.astype(np.float32)))
    # print("MAE score:", mae)
    return mae
# ==============================================================================================================#

if __name__ == '__main__':
    # Doc du lieu train, test tu file
    with open("data_25.dat", "rb") as f:
        arr = pickle.load(f)
        noise_test, origin_test = arr[1], arr[3]

    # Load model
    model = load_model("rfsunet_best.h5", compile=False)

    pred_images = model.predict(noise_test)

    test_size = noise_test.shape[0]

    psrn = PSRN(origin_test, pred_images)
    mae = MAE(origin_test, pred_images)
    # ssim = SSIM(origin_test, pred_images)

    psrn_avg = np.mean(psrn)
    mae_avg = np.mean(mae)
    # ssim_avg = np.mean(ssim)

    print("Mean psrn: ", psrn_avg)
    print("Mean mae: ", mae_avg)
    # print("Mean ssim: ", ssim_avg)

    # Chon random 5 anh de show
    s_id = 25
    e_id = 30

    pred_images = model.predict(noise_test[s_id: e_id])
    ori_images = origin_test[s_id: e_id]

    # Ve len man hinh de kiem tra
    for i in range(s_id, e_id):
        print("PSRN between GT image and predict image %d: " %i,PSRN(ori_images[i -s_id], pred_images[i - s_id]))
        plt.figure(figsize=(8, 3))
        plt.subplot(131)
        plt.imshow(pred_images[i - s_id].reshape(IMAGE_SIZE), cmap='gray')
        plt.title('Model')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(132)
        plt.imshow(ori_images[i -s_id].reshape(IMAGE_SIZE), cmap='gray')
        plt.title('Origin image')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(133)
        plt.imshow(noise_test[i], cmap='gray')
        plt.title('Noise image')
        plt.xticks([])
        plt.yticks([])

        plt.show()
