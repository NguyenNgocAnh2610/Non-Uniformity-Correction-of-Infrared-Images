import os

import matplotlib.pyplot as plt
import pickle
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from keras.models import load_model

from RFSUNet_model import *

# ==============================================================================================================#
n_epochs = 50
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

    # RFSUNet model
    if not os.path.exists('rfsunet.h5'):
        rfsunet = RFSUNet()
        model_opt = Adam(learning_rate=0.002)

        rfsunet.compile(optimizer=model_opt, loss='mae', metrics=['accuracy'])
        print("Creat new RFSUNet model!")
        rfsunet.summary()

    else:
        rfsunet = load_model("rfsunet.h5")  # load model de train tiep

        model_opt = Adam(learning_rate=0.002)
        rfsunet.compile(optimizer=model_opt, loss='mae', metrics=['accuracy'])
        print("Load pre_train model!")
        rfsunet.summary()

    # Định nghĩa callback để lưu trọng số tốt nhất
    lr_scheduler = LearningRateScheduler(schedule)

    checkpoint_best = ModelCheckpoint("rfsunet_best.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    history = rfsunet.fit(noise_train, origin_train, epochs=n_epochs, batch_size=n_batch_size,
                       validation_data=(noise_test, origin_test), callbacks=[checkpoint_best, lr_scheduler],
                       shuffle=True)

    rfsunet.save("rfsunet.h5")

    print(history.history.keys())  # In ra tất cả các metric đã được lưu
    plot_loss_accuracy(history)
