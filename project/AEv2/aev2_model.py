from keras import Input, Model, layers
from keras.layers import Conv2D, Conv2DTranspose


# ==============================================================================================================#
# Total params: 518,209
# Trainable params: 518,209
# Non-trainable params: 0
from skimage.metrics import peak_signal_noise_ratio


def AEv2():
    inputs = layers.Input(shape=(None, None, 1))
    # ========== encoder ==========
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, name='Conv0')(inputs)
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

    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same', strides=1, name='Conv_out')(x)

    # model
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model


# ==============================================================================================================#
def schedule(epoch, lr):
    decay_rate = 0.5
    decay_step = 20
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

# ==============================================================================================================#
def custom_loss(y_true,y_pred):
    res = peak_signal_noise_ratio(y_true, y_pred)
    return res
