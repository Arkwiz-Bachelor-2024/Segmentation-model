from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

"""
Method which returns a U-Net model 

Code inspired by:
https://keras.io/examples/vision/oxford_pets_image_segmentation/
"""


def UNET_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)

    return model


def ResNet_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    encoder = keras.applications.ResNet101(
        include_top=False, weights="imagenet", input_tensor=inputs
    )

    for layer in encoder.layers:
        layer.trainable = False

    x = encoder.output

    # Dropout
    x = layers.Dropout(0.5)(x)

    # Decoder/Upsampling
    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

    # Additional upsampling step to match the target dimension of 512x512
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(32, 3, padding="same")(
        x
    )  # Additional convolution for smoothing
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)

    return model


def DeeplabV3Plus(img_size, num_classes):

    def convolution_block(
        block_input, num_filters=256, kernel_size=3, dilation_rate=1, use_bias=False
    ):
        x = layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=keras.initializers.HeNormal(),
        )(block_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(
            x
        )  # Changed from ops.nn.relu to Keras layer function
        return x

    def DilatedSpatialPyramidPooling(dspp_input):
        dims = dspp_input.shape
        x = layers.AveragePooling2D(pool_size=(dims[1], dims[2]))(
            dspp_input
        )  # Adjusted indexing for batch shape
        x = convolution_block(x, kernel_size=1, use_bias=True)
        out_pool = layers.UpSampling2D(
            size=(dims[1] // x.shape[1], dims[2] // x.shape[2]),
            interpolation="bilinear",
        )(x)

        out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
        out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
        out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
        out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

        x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        output = convolution_block(x, kernel_size=1)
        return output

    model_input = keras.Input(shape=img_size + (3,))
    preprocessed = keras.applications.resnet50.preprocess_input(model_input)
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=preprocessed
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(img_size[0] // 4 // x.shape[1], img_size[1] // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(img_size[0] // x.shape[1], img_size[1] // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)

    return keras.Model(inputs=model_input, outputs=model_output)
