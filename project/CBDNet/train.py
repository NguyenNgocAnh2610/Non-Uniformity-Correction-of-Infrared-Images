import os

import matplotlib.pyplot as plt
import pickle
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from keras.models import load_model

from CBDNet_model import *

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
    if not os.path.exists('cbdnet.h5'):
        cbdnet = CBDNet()
        model_opt = Adam(lr=0.001)

        cbdnet.compile(optimizer=model_opt, loss='mae', metrics=['accuracy'])
        print("Creat new CBDNet model!")
        cbdnet.summary()

    else:
        cbdnet = load_model("cbdnet.h5")  # load model de train tiep

        model_opt = Adam(lr=0.001)
        cbdnet.compile(optimizer=model_opt, loss='mae', metrics=['accuracy'])
        print("Load pre_train model!")
        cbdnet.summary()

    # Định nghĩa callback để lưu trọng số tốt nhất
    lr_scheduler = LearningRateScheduler(schedule)

    checkpoint_best = ModelCheckpoint("cbdnet_best.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    history = cbdnet.fit(noise_train, origin_train, epochs=n_epochs, batch_size=n_batch_size,
                       validation_data=(noise_test, origin_test), callbacks=[checkpoint_best, lr_scheduler],
                       shuffle=True)

    cbdnet.save("cbdnet.h5")

    print(history.history.keys())  # In ra tất cả các metric đã được lưu
    plot_loss_accuracy(history)
