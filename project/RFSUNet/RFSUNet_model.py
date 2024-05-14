import tensorflow as tf
from keras import layers, Model
from keras.layers import Input, Conv2D, Concatenate, UpSampling2D, AveragePooling2D, Conv2DTranspose, Add, MaxPooling2D, BatchNormalization, Activation, Subtract
from keras.optimizers import Adam

# =============================================================================================================#
def LRL_block_V2(input, filters):
    x1 = Conv2D(filters, (3, 3), padding='same', strides=1)(input)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation(activation='relu')(x1)

    x2 = Conv2D(filters, (3, 3), padding='same', strides=1)(x1)
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation(activation='relu')(x2)

    x3 = Conv2D(filters, (3, 3), padding='same', strides=1)(x2)
    x3 = BatchNormalization(axis=3)(x3)
    x3 = Activation(activation='relu')(x3)

    x4 = Conv2D(filters, (3, 3), padding='same', strides=1)(x3)
    x4 = BatchNormalization(axis=3)(x4)
    x4 = Activation(activation='relu')(x4)

    out = Subtract()([x1, x4])
    return out

def RFSUNet():
    input = layers.Input(shape=(None, None, 1))

    # Down
    lrl_1 = LRL_block_V2(input, filters=8)
    #
    max_pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(lrl_1)
    x = Conv2D(16, (3, 3), padding='same')(max_pool1)
    lrl_2 = LRL_block_V2(x, filters=16)

    max_pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(lrl_2)
    x = Conv2D(32, (3, 3), padding='same')(max_pool2)
    lrl_3 = LRL_block_V2(x, filters=32)

    max_pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(lrl_3)
    x = Conv2D(64, (3, 3), padding='same')(max_pool3)
    lrl_4 = LRL_block_V2(x, filters=64)

    max_pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(lrl_4)
    x = Conv2D(128, (3, 3), padding='same')(max_pool4)
    lrl_5 = LRL_block_V2(x, filters=128)

    # Up
    up_samp1 = UpSampling2D()(lrl_5)
    x = Conv2D(64, (3, 3), padding='same')(up_samp1)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    x = Concatenate(axis=-1)([lrl_4, x])
    x = Conv2D(64, (3, 3), padding='same')(x)
    lrl_6 = LRL_block_V2(x, filters=64)

    up_samp2 = UpSampling2D()(lrl_6)
    x = Conv2D(32, (3, 3), padding='same')(up_samp2)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    x = Concatenate(axis=3)([lrl_3, x])
    x = Conv2D(32, (3, 3), padding='same')(x)
    lrl_7 = LRL_block_V2(x, filters=32)

    up_samp3 = UpSampling2D()(lrl_7)
    x = Conv2D(16, (3, 3), padding='same')(up_samp3)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    x = Concatenate(axis=3)([lrl_2, x])
    x = Conv2D(16, (3, 3), padding='same')(x)
    lrl_8 = LRL_block_V2(x, filters=16)

    up_samp4 = UpSampling2D()(lrl_8)
    x = Conv2D(8, (3, 3), padding='same')(up_samp4)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    x = Concatenate(axis=3)([lrl_1, x])
    x = Conv2D(8, (3, 3), padding='same')(x)
    lrl_9 = LRL_block_V2(x, filters=8)

    x = Conv2D(8, (3, 3), padding='same')(lrl_9)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(1, (3, 3), padding='same')(x)

    x = Subtract()([input, x])

    model = Model(inputs=input, outputs=x)
    model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

    return model

# ==============================================================================================================#

# ==============================================================================================================#
def schedule(epoch, lr):
    decay_rate = 0.5
    decay_step = 20
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr
# ==============================================================================================================#
if __name__ == "__main__":
    rfsunet = RFSUNet()
    model_opt = Adam(lr=0.001)

    rfsunet.compile(optimizer=model_opt, loss='mse', metrics=['accuracy'])
    print("Creat new RFSUNetV2 model!")
    rfsunet.summary()

