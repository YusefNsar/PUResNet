from keras.models import Model
from keras.layers import (
    Input,
    Convolution3D,
    MaxPooling3D,
    UpSampling3D,
    concatenate,
    BatchNormalization,
    Activation,
    Add,
    Concatenate,
)
from keras import backend as K
from keras.regularizers import l2


class PUResNet(Model):
    def __init__(self, **kwargs) -> None:
        f = 18
        b_axis = 4
        inputs = Input((36, 36, 36, 18), name="input")
        x = self.conv_block(inputs, [f, f, f], stage=2, block="a", strides=(1, 1, 1))
        x = self.identity_block(x, [f, f, f], stage=2, block="b")
        x1 = self.identity_block(x, [f, f, f], stage=2, block="c")
        x = self.conv_block(
            x, [f * 2, f * 2, f * 2], stage=4, block="a", strides=(2, 2, 2)
        )
        x = self.identity_block(x, [f * 2, f * 2, f * 2], stage=4, block="b")
        x2 = self.identity_block(x, [f * 2, f * 2, f * 2], stage=4, block="f")
        x = self.conv_block(
            x, [f * 4, f * 4, f * 4], stage=5, block="a", strides=(2, 2, 2)
        )
        x = self.identity_block(x, [f * 4, f * 4, f * 4], stage=5, block="b")
        x3 = self.identity_block(x, [f * 4, f * 4, f * 4], stage=5, block="c")
        x = self.conv_block(
            x, [f * 8, f * 8, f * 8], stage=6, block="a", strides=(3, 3, 3)
        )
        x = self.identity_block(x, [f * 8, f * 8, f * 8], stage=6, block="b")
        x4 = self.identity_block(x, [f * 8, f * 8, f * 8], stage=6, block="c")
        x = self.conv_block(
            x, [f * 16, f * 16, f * 16], stage=7, block="a", strides=(3, 3, 3)
        )
        x = self.identity_block(x, [f * 16, f * 16, f * 16], stage=7, block="b")
        x = self.up_conv_block(
            x,
            [f * 16, f * 16, f * 16],
            stage=8,
            block="a",
            size=(3, 3, 3),
            padding="same",
        )
        x = self.identity_block(x, [f * 16, f * 16, f * 16], stage=8, block="b")
        x = Concatenate(axis=4)([x, x4])
        x = self.up_conv_block(
            x,
            [f * 8, f * 8, f * 8],
            stage=9,
            block="a",
            size=(3, 3, 3),
            stride=(1, 1, 1),
        )
        x = self.identity_block(x, [f * 8, f * 8, f * 8], stage=9, block="b")
        x = Concatenate(axis=4)([x, x3])
        x = self.up_conv_block(
            x,
            [f * 4, f * 4, f * 4],
            stage=10,
            block="a",
            size=(2, 2, 2),
            stride=(1, 1, 1),
        )
        x = self.identity_block(x, [f * 4, f * 4, f * 4], stage=10, block="b")
        x = Concatenate(axis=4)([x, x2])
        x = self.up_conv_block(
            x,
            [f * 2, f * 2, f * 2],
            stage=11,
            block="a",
            size=(2, 2, 2),
            stride=(1, 1, 1),
        )
        x = self.identity_block(x, [f * 2, f * 2, f * 2], stage=11, block="b")
        x = Concatenate(axis=4)([x, x1])
        outputs = Convolution3D(
            filters=1,
            kernel_size=1,
            kernel_regularizer=l2(1e-4),
            activation="sigmoid",
            name="pocket",
        )(x)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def identity_block(self, input_tensor, filters, stage, block, layer=None):
        filter1, filter2, filter3 = filters
        if K.image_data_format() == "channels_last":
            bn_axis = 4
        else:
            bn_axis = 1
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"
        x = Convolution3D(
            filters=filter1,
            kernel_size=1,
            name=conv_name_base + "2a",
            kernel_regularizer=l2(1e-4),
        )(input_tensor)
        if layer == None:
            x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
        x = Activation("relu")(x)
        x = Convolution3D(
            filters=filter2,
            kernel_size=3,
            padding="same",
            name=conv_name_base + "2b",
            kernel_regularizer=l2(1e-4),
        )(x)
        if layer == None:
            x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
        x = Activation("relu")(x)
        x = Convolution3D(
            filters=filter3,
            kernel_size=1,
            name=conv_name_base + "2c",
            kernel_regularizer=l2(1e-4),
        )(x)
        if layer == None:
            x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)
        x = Add()([x, input_tensor])
        x = Activation("relu")(x)
        return x

    def conv_block(self, input_tensor, filters, stage, block, strides=(2, 2, 2)):
        filters1, filters2, filters3 = filters

        if K.image_data_format() == "channels_last":
            bn_axis = 4
        else:
            bn_axis = 1
        residue = input_tensor
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"
        x = Convolution3D(
            filters1,
            kernel_size=1,
            strides=strides,
            name=conv_name_base + "2a",
            kernel_regularizer=l2(1e-4),
        )(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
        x = Activation("relu")(x)
        x = Convolution3D(
            filters2,
            kernel_size=3,
            padding="same",
            name=conv_name_base + "2b",
            kernel_regularizer=l2(1e-4),
        )(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
        x = Activation("relu")(x)
        x = Convolution3D(
            filters3,
            kernel_size=1,
            name=conv_name_base + "2c",
            kernel_regularizer=l2(1e-4),
        )(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)
        residue = Convolution3D(
            filters3,
            kernel_size=1,
            strides=strides,
            name=conv_name_base + "1",
            kernel_regularizer=l2(1e-4),
        )(input_tensor)
        residue = BatchNormalization(axis=bn_axis, name=bn_name_base + "1")(residue)
        x = Add()([x, residue])
        x = Activation("relu")(x)

        return x

    def up_conv_block(
        self,
        input_tensor,
        filters,
        stage,
        block,
        stride=(1, 1, 1),
        size=(2, 2, 2),
        padding="same",
        layer=None,
    ):
        filters1, filters2, filters3 = filters
        shortcut = input_tensor
        if K.image_data_format() == "channels_last":
            bn_axis = 4
        else:
            bn_axis = 1
        up_conv_name_base = "up" + str(stage) + block + "_branch"
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"
        x = UpSampling3D(size, name=up_conv_name_base + "2a")(input_tensor)
        x = Convolution3D(
            filters1,
            kernel_size=1,
            strides=stride,
            name=conv_name_base + "2a",
            kernel_regularizer=l2(1e-4),
        )(x)
        if layer == None:
            x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
        x = Activation("relu")(x)
        x = Convolution3D(
            filters2,
            kernel_size=3,
            padding=padding,
            name=conv_name_base + "2b",
            kernel_regularizer=l2(1e-4),
        )(x)
        if layer == None:
            x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
        x = Activation("relu")(x)
        x = Convolution3D(
            filters3,
            kernel_size=1,
            name=conv_name_base + "2c",
            kernel_regularizer=l2(1e-4),
        )(x)
        if layer == None:
            x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)
        shortcut = UpSampling3D(size, name=up_conv_name_base + "1")(input_tensor)
        shortcut = Convolution3D(
            filters3,
            kernel_size=1,
            strides=stride,
            padding=padding,
            name=conv_name_base + "1",
            kernel_regularizer=l2(1e-5),
        )(shortcut)
        if layer == None:
            shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + "1")(
                shortcut
            )
        x = Add()([x, shortcut])
        x = Activation("relu")(x)
        return x
