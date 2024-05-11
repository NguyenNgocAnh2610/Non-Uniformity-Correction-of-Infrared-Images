import os

import numpy as np
from keras import Input, Model, layers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Conv2D, BatchNormalization, Subtract, Activation
from keras.optimizers import Adam

# ==============================================================================================================#
# Total params: 558,977
# Trainable params: 557,057
# Non-trainable params: 1,920

def DnCNN(depth, filters=64, image_channels=1, use_bnorm=True):
    layer_count = 0
    inputs = Input(shape=(None,None,image_channels), name='input' + str(layer_count))

    # Layer 1: Conv + ReLU, 64 filter of size 3x3
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='Orthogonal',
               name='Conv2D' + str(layer_count))(inputs)
    layer_count += 1
    x = Activation(activation='relu', name="ReLU" + str(layer_count))(x)

    # Layer 2 to depth-1: Conv+BN+ReLU, 64 filter of size 3x3x64
    for i in range(depth - 2):
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='Orthogonal',
                   name='Conv2D' + str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            x = BatchNormalization(axis=3, momentum=0, epsilon=0.0001, name="BN" + str(layer_count))(x)
        layer_count += 1
        x = Activation(activation='relu', name="ReLU" + str(layer_count))(x)

    # Last layer: Conv + ReLU, image_channels filter of size 3x3
    layer_count += 1
    x = Conv2D(filters=image_channels, kernel_size=(3, 3), strides=(1, 1), padding="same",
               kernel_initializer='Orthogonal',
               name='Conv2D' + str(layer_count))(x)
    layer_count += 1
    x = Subtract(name='subtract' + str(layer_count))([inputs, x]) # input - noise

    model = Model(inputs=inputs, outputs=x)

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
