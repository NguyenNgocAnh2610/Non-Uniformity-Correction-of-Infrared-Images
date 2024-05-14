import cv2
from keras.models import load_model
from eval_model import *

image_index = 3
WIDTH = 256
HEIGHT = 256

image_index = 7

if __name__ == '__main__':

    # model = load_model("aev2_best.h5")
    # model = load_model("dncnn_best.h5")
    model = load_model("aev2_enhan_best.h5")

    # Đọc và chuyển đổi ảnh đầu vào bằng OpenCV
    origin_image = cv2.imread(f"BSDS500/origin/origin_{image_index}.jpg", cv2.IMREAD_GRAYSCALE)
    # origin_image = cv2.imread(f"origin_{image_index}.jpg", cv2.IMREAD_GRAYSCALE)
    # origin_image = cv2.resize(origin_image, (HEIGHT,WIDTH))
    origin_image = origin_image.astype('float32') / 255.0  # Chuẩn hóa ảnh

    input_image_path = f"BSDS500/noise/noise_{image_index}.jpg"
    # input_image_path = f"noise_{image_index}.jpg"
    # input_image_path = "../test_image/test.jpg"


    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    # input_image = cv2.resize(input_image, (HEIGHT,WIDTH)) # nguoc lai so voi keras
    input_image = input_image.astype('float32') / 255.0  # Chuẩn hóa ảnh

    # Dự đoán ảnh tái tạo bằng cách sử dụng mô hình Autoencoder đã huấn luyện
    input_image_batch = input_image[None, ...]  # Thêm một chiều mới cho batch

    reconstructed_image = model.predict(input_image_batch)

    # Eval model
    print("MAE noise image and GT image: ", MAE(input_image, origin_image))
    print("MAE reconstructed image and GT image: ", MAE(origin_image, reconstructed_image.squeeze()))
    print("PSRN noise image and GT image: ", PSRN(input_image, origin_image))
    print("PSRN reconstructed image and GT image: ", PSRN(origin_image, reconstructed_image.squeeze()))
    print("SSIM noise image and GT image: ", SSIM(input_image, origin_image))
    print("SSIM reconstructed image and GT image: ", SSIM(origin_image, reconstructed_image.squeeze()))

    # Chuyển đổi ảnh tái tạo thành định dạng CV_8U
    reconstructed_image = np.clip((reconstructed_image * 255), 0, 255).astype('uint8')
    reconstructed_image = reconstructed_image.squeeze()

    cv2.imwrite('image_predict.jpg', reconstructed_image)

    # Hiển thị ảnh đầu vào và ảnh tái tạo bằng OpenCV
    cv2.imshow('Input Image', input_image)
    cv2.imshow('Origin Image', origin_image)
    cv2.imshow('Reconstructed Image', reconstructed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

