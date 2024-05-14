import tensorflow as tf
from keras import layers
from keras.layers import Input, Conv2D, Concatenate, UpSampling2D, AveragePooling2D, Conv2DTranspose, Add
from keras.models import Model
import sys


# ==============================================================================================================#
from keras.optimizers import Adam


def upsample_and_sum(x1, x2, output_channels, in_channels, scope=None):
    pool_size = 2
    deconv_filter = Conv2DTranspose(output_channels, (pool_size, pool_size), strides=(pool_size, pool_size),
                                    padding='same', kernel_initializer='truncated_normal')(x1)
    deconv_output = Add()([deconv_filter, x2])
    return deconv_output


# ==============================================================================================================#
def FCN(input):
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    conv5 = Conv2D(1, (3, 3), activation='relu', padding='same')(conv4)
    return conv5


# ==============================================================================================================#
def UNet(input):
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)

    pool1 = AveragePooling2D((2, 2), padding='same')(conv1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)

    pool2 = AveragePooling2D((2, 2), padding='same')(conv2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

    up4 = upsample_and_sum(conv3, conv2, 128, 256, scope='deconv4')
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

    up5 = upsample_and_sum(conv4, conv1, 64, 128, scope='deconv5')
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

    out = Conv2D(1, (1, 1), activation=None, padding='same')(conv5)
    return out


# ==============================================================================================================#
# Total params: 4,361,794
# Trainable params: 4,361,794
# Non-trainable params: 0

def CBDNet():
    input = layers.Input(shape=(None, None, 1))
    noise_level = FCN(input)
    concat_img = Concatenate(axis=3)([input, noise_level])
    output = UNet(concat_img) + input
    # model = Model(inputs=input, outputs=[noise_level, output])
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


# ==============================================================================================================#
def schedule(epoch, lr):
    decay_rate = 0.5
    decay_step = 10
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr
# ==============================================================================================================#

if __name__ == "__main__":
    print(sys.executable)
    cbdnet = CBDNet()
    model_opt = Adam(lr=0.001)

    cbdnet.compile(optimizer=model_opt, loss='mae', metrics=['accuracy'])
    print("Creat new CBDNet model!")
    cbdnet.summary()
