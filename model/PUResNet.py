from keras.models import Model
from keras.layers import (
    Input,
    Convolution3D,
    UpSampling3D,
    BatchNormalization,
    Activation,
    Add,
    Concatenate,
)
from keras import backend as K
from keras.regularizers import l2
from tensorflow.python.keras.layers import Input as KerasTensor
from typing import Union, Tuple


class PUResNet(Model):
    """
    ResUNet Model Implementation. The output of the model is a sigmoid probability map with the same size as the input mol shape.

    It is a 3D array of probabilities that represents the likelihood of each atom in the mol being a part of a binding site or not.
    """

    def __init__(self, d: int = 36, f: int = 18, **kwargs) -> None:
        """Init PUResNet Model with input of shape (d, d, d, f). d -> input dimensionality, f -> features number"""

        # input layer
        inputs = Input((d, d, d, f), name="input")

        # encoder
        conv1 = self.conv_block(inputs, f, 2, (1, 1, 1))
        iden1_E = self.identity_block(conv1, f, 2)

        conv2 = self.conv_block(iden1_E, f * 2, 4, (2, 2, 2))
        iden2_E = self.identity_block(conv2, f * 2, 4)

        conv3 = self.conv_block(iden2_E, f * 4, 5, (2, 2, 2))
        iden3_E = self.identity_block(conv3, f * 4, 5)

        conv4 = self.conv_block(iden3_E, f * 8, 6, (3, 3, 3))
        iden4_E = self.identity_block(conv4, f * 8, 6)

        # bridge
        conv5 = self.conv_block(iden4_E, f * 16, 7, (3, 3, 3))
        iden5_B = self.identity_block(conv5, f * 16, 7)

        # decoder
        up4 = self.up_conv_block(iden5_B, f * 16, 8, (3, 3, 3))
        iden4_D = self.identity_block(up4, f * 16, 8, iden4_E)

        up3 = self.up_conv_block(iden4_D, f * 8, 9, (3, 3, 3))
        iden3_D = self.identity_block(up3, f * 8, 9, iden3_E)

        up2 = self.up_conv_block(iden3_D, f * 4, 10, (2, 2, 2))
        iden2_D = self.identity_block(up2, f * 4, 10, iden2_E)

        up1 = self.up_conv_block(iden2_D, f * 2, 11, (2, 2, 2))
        iden1_D = self.identity_block(up1, f * 2, 11, iden1_E)

        # output layer
        outputs = Convolution3D(
            filters=1,
            kernel_size=1,
            kernel_regularizer=l2(1e-4),
            activation="sigmoid",
            name="pocket",
        )(iden1_D)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def identity_block(
        self,
        input_tensor: KerasTensor,
        filters: int,
        level: int,
        encoder_tensor: Union[KerasTensor, None] = None,
    ) -> KerasTensor:
        """
        Make a convolution block of 3 convolutional layers with 1 batch normalization and 1 activation layers
        between each and an addition layer skip connection.

        And if this is a decoder block we pass `encoder_tensor` and use a concatenation layer skip connection between
        that tensor from the encoder identity block of the same channel dimension, and tensor of this decoder identity block.

        The rich skip connections in the RESUNET helps in better flow of information between different layers,
        which helps in better flow of gradients while training  (backpropagation).
        """

        is_channels_last = K.image_data_format() == "channels_last"
        bn_axis = 4 if is_channels_last else 1

        conv_name_base = f"IB_L{level}_Conv_"
        bn_name_base = f"IB_L{level}_BN_"

        # conv 1
        x = Convolution3D(
            filters=filters,
            kernel_size=1,
            name=conv_name_base + "A",
            kernel_regularizer=l2(1e-4),
        )(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "A")(x)
        x = Activation("relu")(x)

        # conv 2
        x = Convolution3D(
            filters=filters,
            kernel_size=3,
            padding="same",
            name=conv_name_base + "B",
            kernel_regularizer=l2(1e-4),
        )(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "B")(x)
        x = Activation("relu")(x)

        # conv 3
        x = Convolution3D(
            filters=filters,
            kernel_size=1,
            name=conv_name_base + "C",
            kernel_regularizer=l2(1e-4),
        )(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "C")(x)

        # addition layer skip connection
        x = Add()([x, input_tensor])
        x = Activation("relu")(x)

        # concatenation layer skip connection between encoder and decoder
        if encoder_tensor != None:
            x = Concatenate(axis=4)([x, encoder_tensor])

        return x

    def conv_block(
        self,
        input_tensor: KerasTensor,
        filters: int,
        level: int,
        strides: Tuple[int, int, int] = (2, 2, 2),
    ) -> KerasTensor:
        """
        Make a residual convolution block of 3 convolutional layers with 1 batch normalization and 1 activation layers
        between each and a residual connection in the end.

        The use of residual blocks helps in building a deeper network without worrying about the problem of vanishing
        gradient or exploding gradients. It also helps in easy training of the network.
        """
        is_channels_last = K.image_data_format() == "channels_last"
        bn_axis = 4 if is_channels_last else 1

        conv_name_base = f"CB_L{level}_Conv_"
        bn_name_base = f"CB_L{level}_BN_"

        # conv 1
        x = Convolution3D(
            filters,
            kernel_size=1,
            strides=strides,
            name=conv_name_base + "A",
            kernel_regularizer=l2(1e-4),
        )(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "A")(x)
        x = Activation("relu")(x)

        # conv 2
        x = Convolution3D(
            filters,
            kernel_size=3,
            padding="same",
            name=conv_name_base + "B",
            kernel_regularizer=l2(1e-4),
        )(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "B")(x)
        x = Activation("relu")(x)

        # conv 3
        x = Convolution3D(
            filters,
            kernel_size=1,
            name=conv_name_base + "C",
            kernel_regularizer=l2(1e-4),
        )(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "C")(x)

        # residual connection
        residue = Convolution3D(
            filters,
            kernel_size=1,
            strides=strides,
            name=conv_name_base + "R",
            kernel_regularizer=l2(1e-4),
        )(input_tensor)
        residue = BatchNormalization(axis=bn_axis, name=bn_name_base + "R")(residue)
        x = Add()([x, residue])
        x = Activation("relu")(x)

        return x

    def up_conv_block(
        self,
        input_tensor: KerasTensor,
        filters: int,
        level: int,
        size: Tuple[int, int, int] = (2, 2, 2),
        strides: Tuple[int, int, int] = (1, 1, 1),
        padding: str = "same",
    ) -> KerasTensor:
        """
        Make a residual upsampling convolution block of 1 upsampling layer, 3 convolutional layers with 1 batch
        normalization and 1 activation layers between each and a residual connection in the end.

        The use of residual blocks helps in building a deeper network without worrying about the problem of vanishing
        gradient or exploding gradients. It also helps in easy training of the network.
        """

        is_channels_last = K.image_data_format() == "channels_last"
        bn_axis = 4 if is_channels_last else 1

        up_conv_name_base = f"UB_L{level}_UP_"
        conv_name_base = f"UB_L{level}_Conv_"
        bn_name_base = f"UB_L{level}_BN_"

        # upsampling
        x = UpSampling3D(size, name=up_conv_name_base + "A")(input_tensor)

        # conv 1
        x = Convolution3D(
            filters,
            kernel_size=1,
            strides=strides,
            name=conv_name_base + "A",
            kernel_regularizer=l2(1e-4),
        )(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "A")(x)
        x = Activation("relu")(x)

        # conv 2
        x = Convolution3D(
            filters,
            kernel_size=3,
            padding=padding,
            name=conv_name_base + "B",
            kernel_regularizer=l2(1e-4),
        )(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "B")(x)
        x = Activation("relu")(x)

        # conv 3
        x = Convolution3D(
            filters,
            kernel_size=1,
            name=conv_name_base + "C",
            kernel_regularizer=l2(1e-4),
        )(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "C")(x)

        # residual connection
        shortcut = UpSampling3D(size, name=up_conv_name_base + "R")(input_tensor)
        shortcut = Convolution3D(
            filters,
            kernel_size=1,
            strides=strides,
            padding=padding,
            name=conv_name_base + "R",
            kernel_regularizer=l2(1e-5),
        )(shortcut)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + "R")(shortcut)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)

        return x
