import cv2
import numpy as np
import matplotlib.pyplot as plt

NUM_LINES = 200
ALPHA_CONST = False
ALPHA_MIN = 20  # 15 - 30
ALPHA_MAX = 30  # 15 - 30
SNR_DB = 25
GAUSS_KERNEL = 3


# ===================================================================================================================
def add_line_noise(image, alpha, num_lines):
    noisy_image = np.copy(image).astype(np.float32)
    height, width = image.shape[:2]  # Load image by opencv

    for _ in range(num_lines):
        if np.random.rand(1) < 0.0:
            # Add horizontal line noise # chieu ngang
            row = np.random.randint(0, height)
            noisy_image[row, :] = np.clip((noisy_image[row, :] + np.round(np.random.rand(1) * 255 // alpha)), 0, 255)
        else:
            # Add vertical line noise
            col = np.random.randint(0, width)
            noisy_image[:, col] = np.clip((noisy_image[:, col] + np.round(np.random.rand(1) * 255 // alpha)), 0, 255)
    return noisy_image.astype(np.uint8)


# ===================================================================================================================

def add_awgn_noise(image, snr_db=SNR_DB):
    """
    Thêm nhiễu AWGN (Additive White Gaussian Noise) vào ảnh xám.

    Tham số:
        image (numpy.ndarray): Ảnh xám (ma trận NumPy).
        snr_db (float): Tỷ lệ tín hiệu đến nhiễu (SNR) trong đơn vị dB.

    Trả về:
        numpy.ndarray: Ảnh xám sau khi thêm nhiễu.
    """
    # Tính toán công suất của ảnh
    image_power = np.mean(np.abs(image) ** 2)

    # Tính toán công suất của nhiễu dựa trên SNR
    noise_power = image_power / (10 ** (snr_db / 10))

    # Tạo nhiễu Gaussian có phương sai noise_power
    noise = np.random.normal(scale=np.sqrt(noise_power), size=image.shape).astype(np.float32)

    # Thêm nhiễu vào ảnh
    noisy_image = (image + noise).astype(np.float32)

    # Đảm bảo giá trị pixel không vượt quá giới hạn của ảnh (0-255 cho ảnh uint8)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image


# ===================================================================================================================

def creat_noise_image(image):
    if ALPHA_CONST:
        alpha = ALPHA_MIN
    else:
        alpha = np.random.randint(ALPHA_MIN, ALPHA_MAX)

    num_lines = NUM_LINES

    # Creat line noise
    image_line_noise = add_line_noise(image, alpha, num_lines).astype(np.float32)

    # Gaussian filter whith kernel = gauss_kernel, gauss_kernel
    image_line_noise = cv2.GaussianBlur(image_line_noise, (GAUSS_KERNEL, GAUSS_KERNEL), 0)

    # Creat gaussian noise with mean = 0, standard_deviation
    noise_image = add_awgn_noise(image_line_noise, snr_db=25)

    return noise_image


# ===================================================================================================================


if __name__ == "__main__":
    # Load grayscale image
    image = cv2.imread("origin (1).jpg", cv2.IMREAD_GRAYSCALE)
    print("Size cua anh ban dau: ", image.shape)
    image = cv2.resize(image, (256, 256))
    print("Size cua anh sau khi resize: ", image.shape)

    for i in range(100):
        noise_image = creat_noise_image(image)
        cv2.imshow("Noise", noise_image)
        cv2.waitKey(1000)

    print(noise_image.shape)

    # Display original, line noise, and Gaussian noise images
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(122)
    plt.imshow(noise_image, cmap='gray')
    plt.title('noise_image')
    plt.show()
