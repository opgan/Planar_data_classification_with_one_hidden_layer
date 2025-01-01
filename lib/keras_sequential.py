# pylint: disable=unused-import, unused-variable, unused-argument, no-name-in-module 

import tensorflow as tf
import tensorflow.keras.layers as tfl # pylint: disable= import-error
from lib.log import log

def build_model():
    happy_model = happyModel()

    happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
    happy_model.summary()
    #log(happy_model.summary())

    return happy_model

def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    model = tf.keras.Sequential([  # pylint: disable=no-member
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            ## Conv2D with 32 7x7 filters and stride of 1
            ## BatchNormalization for axis 3
            ## ReLU
            ## Max Pooling 2D with default parameters
            ## Flatten layer
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            
            # YOUR CODE STARTS HERE
            tfl.ZeroPadding2D(padding=3, input_shape=(64, 64, 3)),  # Add ZeroPadding2D layer with padding of 3
            tfl.Conv2D(filters=32, kernel_size=(7, 7), strides=1),
            tfl.BatchNormalization(axis=3),
            tfl.ReLU(),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(units=1, activation='sigmoid') 
            # YOUR CODE ENDS HERE
        ])
    
    return model