import os

import matplotlib.pyplot as plt
import pickle
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from tensorflow.keras.models import load_model


# ==============================================================================================================#
IMAGE_SIZE = (256, 256, 1)
origin_path = 'original_images'
noise_path = 'noise_images'
n_epochs = 40
n_batch_size = 4
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
    with open("data.dat", "rb") as f:
        arr = pickle.load(f)
        noise_train, noise_test, origin_train, origin_test = arr[0], arr[1], arr[2], arr[3]
        print("Load data from binary file!")
        print(noise_train.shape)

    if not os.path.exists('dncnn.h5'):
        dncnn = DnCNN(depth=17)
        model_opt = Adam(lr=0.001)

        dncnn.compile(optimizer=model_opt, loss='mse', metrics=['accuracy'])
        print("Creat new DnCNN model!")
        dncnn.summary()
    else:
        dncnn = load_model("dncnn.h5")  # load model de train tiep
        model_opt = Adam(lr=0.001)
        dncnn.compile(optimizer=model_opt, loss='mse', metrics=['accuracy'])
        print("Load pre_train model!")

        dncnn.summary()

    # Định nghĩa callback để lưu trọng số tốt nhất
    checkpoint_best = ModelCheckpoint("dncnn_best.h5", monitor='val_loss', save_best_only=True,
                                      mode='auto', verbose=1)

    lr_scheduler = LearningRateScheduler(schedule)

    history = dncnn.fit(noise_train, origin_train, epochs=n_epochs, batch_size=n_batch_size,
                        validation_data=(noise_test, origin_test), callbacks=[checkpoint_best, lr_scheduler],
                        shuffle=True)

    dncnn.save("dncnn.h5")

    print(history.history.keys())  # In ra tất cả các metric đã được lưu
    plot_loss_accuracy(history)

# ==============================================================================================================#
