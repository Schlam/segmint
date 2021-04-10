import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import VGG16

def encoder(inputs, base=None, train_base=False, **kwargs):    
    """

    """
    # If unspecified,
    if not base:    

        # load VGG as base model for encoder
        base = VGG16(weights="imagenet", 
                     include_top=False, 
                     input_tensor=inputs, 
                     **kwargs)

        # Set trainable to True or False
        base.trainable = train_base
    
    # Initialize loop variables
    x = inputs
    pool = []
        
    # For each layer
    for layer in base.layers[1:]:
        

        # feed the output from one layer into the next
        x = layer(x)
        if layer.name == "block1_conv2":

            first = x
            
        if 'pool' in layer.name and "5" not in layer.name:

            # Append pooling layer outputs to list 
            
            x = tf.keras.layers.BatchNormalization()(x)
            pool.append(x)
            
    return x, tuple([first] + pool)
    

def conv2d_block(inputs, filters, kernel_size=3, num_convolutions=2):
    """Conv2D Block
    Returns the output of a block of two Conv2D layers
    """
    
    x = inputs
    for _ in range(num_convolutions):
        
        # Convolve inputs twice, keeping same dimensions
        x = tf.keras.layers.Conv2D(filters, kernel_size, 
                                   padding="same", activation='relu')(x)
        
    return x

def decoder_block(
    inputs, conv, filters, 
    kernel_size=4, strides=2, dropout=.3, 
    padding="same", use_bias=False):
    """Decoder Block
    """

    # Upscale
    x = tf.keras.layers.Conv2DTranspose(filters, kernel_size, 
                                        strides=strides, padding=padding, 
                                        use_bias=use_bias)(inputs)    
    
    # Concatenate
    x = tf.keras.layers.concatenate([x, conv])
    
    # Add dropout and further convs
    x = tf.keras.layers.Dropout(dropout)(x)
    x = conv2d_block(x, filters, kernel_size)
    return x

def decoder(inputs, convs, kernel_size=4, strides=2, dropout=0.3, n_classes=1):
    """Decoder
    Upsamples the results from Encoder/Bottleneck blocks to 
    output a segmentation maskof the same size as the original image 
    """
    x = inputs
    
    f16, f32, f64, f128, f256 = convs[::-1]
    
    params = dict(kernel_size=kernel_size, strides=strides, dropout=dropout)
    
    x = inputs
    x = decoder_block(x, f16, filters=256, **params)
    x = decoder_block(x, f32, filters=128, **params)
    x = decoder_block(x, f64, filters=64, **params)
    x = decoder_block(x, f128, filters=32, **params)
    x = decoder_block(x, f256, filters=16, **params)

    
    if n_classes == 1:
    
        activation = "sigmoid"
    else:
    
        activation = "softmax"

    outputs = tf.keras.layers.Conv2D(n_classes, 
                                     kernel_size=1, 
                                     padding="same", 
                                     activation=activation)(x)
    return outputs

def unet(inputs, n_classes):
    """UNet
    Combines the encoder/decoder blocks (no bottleneck)
    """
    encoder_output, convolution_outputs = encoder(inputs)
    outputs = decoder(encoder_output, convolution_outputs, n_classes=n_classes)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def build_unet(inputs, n_classes): 
    """UNet
    Combines the encoder/decoder blocks (no bottleneck)
    """
    encoder_output, convolutions = encoder(inputs)
    outputs = decoder(encoder_output, convolutions, n_classes=n_classes)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
