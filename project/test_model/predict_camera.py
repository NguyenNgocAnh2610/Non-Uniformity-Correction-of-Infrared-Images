import cv2
import os
import numpy as np
from keras.models import load_model
from creat_non_uniformity import *
from eval_model import *

video_path = 0
IMAGE_SIZE = (256, 256)

if __name__ == '__main__':
    model = load_model("cbdnet_best.h5", compile=False)
    video = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not video.isOpened():
        print("Can not open video from file!")
        exit()

    psrn = 0
    i = 0

    # Read video from file
    while True:
        ret, frame = video.read()

        if not ret:
            print("End video!")
            break
        # convert frame from RGB to GRAY
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # frame = cv2.resize(frame, IMAGE_SIZE)

        cv2.imshow("Video GT", frame)

        # input_image = cv2.resize(frame, IMAGE_SIZE)

        noise_image = creat_noise_image(frame)
        cv2.imshow("Video noise", noise_image)

        # input_image = input_image.astype('float32') / 255.0  # Chuẩn hóa ảnh
        input_image = noise_image.astype('float32') / 255.0  # Chuẩn hóa ảnh

        # Dự đoán ảnh tái tạo bằng cách sử dụng mô hình Autoencoder đã huấn luyện
        input_image_batch = input_image[None, ...]  # Thêm một chiều mới cho batch

        reconstructed_image = model.predict(input_image_batch)

        # Chuyển đổi ảnh tái tạo thành định dạng CV_8U
        reconstructed_image = np.clip((reconstructed_image * 255), 0, 255).astype('uint8')

        # Hiển thị ảnh đầu vào và ảnh tái tạo bằng OpenCV

        cv2.imshow('Reconstructed Image', reconstructed_image.squeeze())

        print("PSRN curent frame and noise: ", PSRN(frame, noise_image))
        print("PSRN curent frame and reconstructed: ", PSRN(frame, reconstructed_image.squeeze()))
        psrn += PSRN(frame, reconstructed_image.squeeze())
        i += 1

        # Wait for a key press. If 'q' is pressed, exit the loop
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    print("Mean PSRN: ", psrn / i)

    cv2.destroyAllWindows()
    video.release()
