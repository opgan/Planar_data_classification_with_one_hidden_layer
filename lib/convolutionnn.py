# pylint: disable=unused-import, unused-variable, unused-argument, no-name-in-module

import tensorflow as tf
import tensorflow.keras.layers as tfl  # pylint: disable= import-error
from lib.log import log


def build_conv_model(input_shape):
    conv_model = convolutional_model(input_shape)
    conv_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    conv_model.summary()

    return conv_model


def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes.
    Normally, functions should take these values as function parameters.

    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """

    input_img = tf.keras.Input(shape=input_shape)  # pylint: disable= no-member
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    # Z1 = None
    ## RELU
    # A1 = None
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    # P1 = None
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    # Z2 = None
    ## RELU
    # A2 = None
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    # P2 = None
    ## FLATTEN
    # F = None
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'"
    # outputs = None
    # YOUR CODE STARTS HERE
    Z1 = tfl.Conv2D(filters=8, kernel_size=(4, 4), strides=1, padding="same")(input_img)
    A1 = tfl.ReLU()(Z1)
    P1 = tfl.MaxPool2D(pool_size=(8, 8), strides=8, padding="same")(A1)
    Z2 = tfl.Conv2D(filters=16, kernel_size=(2, 2), strides=1, padding="same")(P1)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPool2D(pool_size=(4, 4), strides=4, padding="same")(A2)
    F = tfl.Flatten()(P2)
    outputs = tfl.Dense(units=6, activation="softmax")(F)
    # YOUR CODE ENDS HERE
    model = tf.keras.Model(  # pylint: disable= no-member
        inputs=input_img, outputs=outputs
    )  # pylint: disable= no-member
    return model
