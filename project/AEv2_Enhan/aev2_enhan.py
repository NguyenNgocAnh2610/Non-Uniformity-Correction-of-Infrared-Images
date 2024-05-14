from keras import Input, Model, layers
from keras.layers import Conv2D, Conv2DTranspose, Concatenate, Subtract, Add
from keras.optimizers import Adam
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio
import sys

# ==============================================================================================================#
# Total params: 518,209
# Trainable params: 518,209
# Non-trainable params: 0

def FCN(input):
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    conv5 = Conv2D(1, (3, 3), activation='relu', padding='same')(conv4)
    return conv5

def AEv2(input):
    # inputs = layers.Input(shape=(None, None, 1))
    # ========== encoder ==========
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, name='Conv0')(input)
    res0 = x

    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, name='Conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, name='Conv2')(x)
    res1 = x

    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, name='Conv3')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, name='Conv4')(x)
    res2 = x

    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, name='Conv5')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, name='Conv6')(x)
    res3 = x

    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, name='Conv7')(x)

    # ========== decoder ==========
    x = Conv2DTranspose(64, (3, 3), padding='same', strides=1, name='ConvT1')(x)
    x = x + res3

    x = Conv2DTranspose(64, (3, 3), padding='same', strides=1, name='ConvT2')(x)
    x = Conv2DTranspose(64, (3, 3), padding='same', strides=1, name='ConvT3')(x)
    x = x + res2

    x = Conv2DTranspose(64, (3, 3), padding='same', strides=1, name='ConvT4')(x)
    x = Conv2DTranspose(64, (3, 3), padding='same', strides=1, name='ConvT5')(x)
    x = x + res1

    x = Conv2DTranspose(64, (3, 3), padding='same', strides=1, name='ConvT6')(x)
    x = Conv2DTranspose(64, (3, 3), padding='same', strides=1, name='ConvT8')(x)
    x = x + res0

    x = Conv2D(1, (3, 3), activation=None, padding='same', strides=1, name='Conv_out')(x)

    # # model
    # model = Model(inputs=inputs, outputs=x)
    # model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

    return x

# def rdb_block(inputs, numLayers):
#     channels = inputs.get_shape()[-1]  # Get the amount of channels in our data, which is 1.
#
#     storedOutputs = [inputs]
#
#     for _ in range(numLayers):  # Here, "numLayers" represents the number of Conv2D layers
#         # that are used for the RDB feature extraction process.
#         localConcat = tf.concat(storedOutputs, axis=-1)
#         # localConcat = Concatenate(axis=-1)(storedOutputs)
#
#         out = Conv2D(filters=channels, kernel_size=3, padding="same",
#                      activation="relu")(localConcat)
#
#         storedOutputs.append(out)  # The outputs of each Conv2D layer are appended.
#
#     finalConcat = tf.concat(storedOutputs, axis=-1)
#     # finalConcat = Concatenate(axis=-1)(storedOutputs)
#     finalOut = Conv2D(filters=channels, kernel_size=1,  # This Conv2D layer is called "pointwise"
#                       padding="same", activation="relu")(finalConcat)  # convolution layer. It basically prepares
#     # the data to be added to the original input
#     finalOut = Add()([finalOut, inputs])  # and exit the RDB block to enter the next
#     # layer in the CNN.
#     return finalOut


def Enhan(input):
    # inputs = Input(shape=(None, None, channels))
    x1 = Conv2D(64, 7, padding='same', activation='relu')(input)
    x2 = Conv2D(32, 5, padding='same', activation='relu')(x1)
    x = Concatenate(axis=3)([x1, x2])

    x1 = Conv2D(32, 5, padding='same', activation='relu')(x)
    x2 = Conv2D(32, 3, padding='same', activation='relu')(x1)
    x = Concatenate(axis=3)([x1, x2])

    x1 = Conv2D(32, 3, padding='same', activation='relu')(x)
    x2 = Conv2D(32, 3, padding='same', activation='relu')(x1)
    x = Concatenate(axis=3)([x1, x2])

    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(1, 3, padding='same')(x)
    x = Add()([x, input])

    # model = Model(inputs = input, outputs=X)
    return x

def AEv2_Enhan():
    inputs = layers.Input(shape=(None, None, 1))

    noise_level = FCN(inputs)

    concat_img = Concatenate(axis=3)([inputs, noise_level])
    out_aev2 = inputs + AEv2(concat_img)

    aev2_enhan = Enhan(out_aev2)

    model = Model(inputs=inputs, outputs=aev2_enhan)
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
    # print(sys.executable)
    aev2 = AEv2_Enhan()
    model_opt = Adam(lr=0.001)

    aev2.compile(optimizer=model_opt, loss='mae', metrics=['accuracy'])
    print("Creat new AEv2_Enhan model!")
    aev2.summary()
