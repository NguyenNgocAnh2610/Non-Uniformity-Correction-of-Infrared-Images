import os

import matplotlib.pyplot as plt
import pickle
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from keras.models import load_model
from LSC_model import *


# ==============================================================================================================#
n_epochs = 20
n_batch_size = 8  # AEv2
# ==============================================================================================================#
def plot_loss_accuracy(history):
    # Plot loss
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.jpg')
    plt.show()

    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.jpg')
    plt.show()
# ==============================================================================================================#


if __name__ == "__main__":
    with open("data_25.dat", "rb") as f:
        arr = pickle.load(f)
        noise_train, noise_test, origin_train, origin_test = arr[0], arr[1], arr[2], arr[3]
        print("Load data from binary file!")
        print(noise_train.shape)

    # CBDNet model
    if not os.path.exists('lsccnn_best.h5'):
        builder = ADresnetBuilder()
        LSC = builder.build_resnet18()
        LSC.compile(optimizer=Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1.e-8), loss='mae', metrics=['accuracy'])
        print("Creat new CBDNet model!")
        LSC.summary()

    else:
        LSC = load_model("lsccnn_best.h5", compile=False)  # load model de train tiep

        LSC.compile(optimizer=Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1.e-8), loss='mae', metrics=['accuracy'])
        print("Load pre_train model!")
        LSC.summary()

    # Định nghĩa callback để lưu trọng số tốt nhất
    lr_scheduler = LearningRateScheduler(schedule)

    checkpoint_best = ModelCheckpoint("lsccnn_best_2.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    history = LSC.fit(noise_train, origin_train, epochs=n_epochs, batch_size=n_batch_size,
                       validation_data=(noise_test, origin_test), callbacks=[checkpoint_best, lr_scheduler],
                       shuffle=True)

    LSC.save("lsccnn.h5")

    print(history.history.keys())  # In ra tất cả các metric đã được lưu
    plot_loss_accuracy(history)

