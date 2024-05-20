from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

"""
Method which returns a U-Net model 

Code inspired by:
https://keras.io/examples/vision/oxford_pets_image_segmentation/
"""


def UNET_model_skip(img_size, num_classes):

    def decoder_block(inputs, skip_features, num_filters):

        # Upsampling with 2x2 filter
        x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="valid")(
            inputs
        )

        # Copy and crop the skip features
        # to match the shape of the upsampled input
        skip_features = layers.resize(skip_features, size=(x.shape[1], x.shape[2]))

        x = layers.Concatenate()([x, skip_features])

        # Convolution with 3x3 filter followed by ReLU activation
        x = tf.keras.layers.Conv2D(num_filters, 3, padding="valid")(x)
        x = tf.keras.layers.Activation("relu")(x)

        # Convolution with 3x3 filter followed by ReLU activation
        x = tf.keras.layers.Conv2D(num_filters, 3, padding="valid")(x)
        x = tf.keras.layers.Activation("relu")(x)

        return x

    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(128, 3, strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(128, 3, strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Intialize list to store skip connections
    skip_connections = []
    skip_connections.append(x)  # Adds the first skip connection

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [256, 512, 1024]:

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        # Adds the skip connection except for the bottleneck
        if filters != 1024:
            skip_connections.append(x)  # Add this skip connection to the list

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [512, 256, 128]:

        # x = layers.UpSampling2D(2)(x)

        # Adds the skip connection after bottleneck
        # x = layers.Concatenate(axis=-1)([x, skip_connections.pop()])

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 2, 2, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.conv2dtranspose(filters, 3, padding="same")(x)
        x = layers.batchnormalization()(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual        # Adds the skip connection after bottleneck

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)

    return model


def UNET_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(256, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [512, 1024, 2048]:
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

    for filters in [2048, 1024, 512, 256]:

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


def UNet_model_skip_backbone(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    encoder = keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_tensor=inputs
    )

    for layer in encoder.layers:
        layer.trainable = False

    x = encoder.output

    # Upsample the output to retain as much semantic details as possible
    x = layers.UpSampling2D(2)(x)

    # Reduce size of feature maps to 512
    x = layers.Conv2D(1024, 1, padding="same")(x)

    # Upsample skip connections to fit the decoder
    # Decoder/Upsampling
    for filters in [1024, 512, 256, 128]:

        match filters:
            case 512:
                # 16x16x512
                skip = encoder.get_layer("conv5_block3_2_relu").output
            case 256:
                # 32x32x256
                skip = encoder.get_layer("conv4_block6_2_relu").output
            case 128:
                # 64x64x128
                skip = encoder.get_layer("conv3_block4_2_relu").output
            case 64:
                # 128x128x64
                skip = encoder.get_layer("conv2_block3_2_relu").output

        previous_block_activation = x  # Set aside next residual

        # Adds the skip connection
        if filters != 512:
            skip = layers.UpSampling2D(4)(skip)
            x = layers.Concatenate()([x, skip])

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        if filters != 32:
            x = layers.UpSampling2D(2)(x)
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)

            # Project residual
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)

    return model


# TODO OS4,multi-grid(1,2,1), Xception, freeze batch norm layers, unfreeze encoder,
# TODO tvernsky loss, crop size, multi-scaling, FCRF?, DPC?
def DeeplabV3Plus(img_size, num_classes):

    def convolution_block(
        block_input, num_filters=512, kernel_size=3, dilation_rate=1, use_bias=False
    ):
        x = layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            # Normal distribution initializer apparently
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
        weights="imagenet",
        include_top=False,
        input_tensor=preprocessed,
    )

    for layer in resnet50.layers:
        layer.trainable = False

    # Make the output to 2nd convultional block
    x = resnet50.get_layer("conv5_block3_2_relu").output

    # ASPP pyramid
    x = DilatedSpatialPyramidPooling(x)

    # 4x upsampling
    input_a = layers.UpSampling2D(
        size=(img_size[0] // 4 // x.shape[1], img_size[1] // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    # Concatenate the low-level features with the high-level features
    x = layers.Concatenate(axis=-1)([input_a, input_b])

    # Regular convolutions on global context from ASPP and low-level(early) features
    # 2 * 3x3 convolutions
    x = convolution_block(x)
    x = convolution_block(x)

    # 4x upsampling
    x = layers.UpSampling2D(
        size=(img_size[0] // x.shape[1], img_size[1] // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(
        num_classes, kernel_size=(1, 1), padding="same", activation="softmax"
    )(x)

    return keras.Model(inputs=model_input, outputs=model_output)
