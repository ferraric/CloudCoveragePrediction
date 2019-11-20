import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv3D, Conv3DTranspose
import os


class Conv3dModel(Model):
    def __init__(self, config):
        super(Conv3dModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        self.conv1 = Conv3D(kernel_size = [3,3,3], filters=16,
                            #dilation_rate=[2, 2, 2],
                            strides=[2,2,2], padding="valid", activation="relu",
                            input_shape=(self.config.batch_size, 121, 59, 90, 2))
        self.conv2 = Conv3D(kernel_size = [3,3,3], filters=32,
                            #dilation_rate=[2, 2, 2],
                            strides=[2,2,2], padding="valid", activation="relu")
        self.conv3 = Conv3D(kernel_size = [3,3,3], filters=64,
                            #dilation_rate=[2, 2, 2],
                            #strides=[2,2,2],
                            padding="valid", activation="relu")
        self.deconv3 = Conv3DTranspose(kernel_size = [3,3,3], filters=32,
                                       #strides=[2,2,2],
                                       padding="valid", activation="relu",
                                       output_padding=[0,0,0])
        self.deconv2 = Conv3DTranspose(kernel_size = [3,3,3], filters=16,
                                       strides=[2,2,2], padding="valid", activation="relu",
                                       output_padding=[1,0,1])
        self.deconv1 = Conv3DTranspose(kernel_size=[3,3,3], filters=2,
                                       strides=[2,2,2], padding="valid", activation="relu",
                                       output_padding=[0,0,1])

        #self.pad = ZeroPadding2D(padding=((1, 0), (0, 0)))
        #self.conv1 = Conv2D(121, [3,3], padding="same", activation="relu", input_shape=(90, 59, 121))
        #self.max_pool = MaxPooling2D(padding="same")
        #self.conv2 = Conv2D(121, [3,3], padding="same", activation="relu")
        #self.up_sam = UpSampling2D((2, 2))
        #self.crop = Cropping2D(cropping=((1, 1), (1, 0)))

    def call(self, x):
        input = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        x = tf.keras.layers.average([x, input])
        return x