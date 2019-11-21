import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout


class UNetModel(Model):
    """This is a simple example class that shows how to use the project architecture for semantic segmentation. """
    def __init__(self, config):
        super(UNetModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        '''Build U-Net model'''
        self.inputlayer = InputLayer(input_shape=(90, 59, 121))

        self.conv11 = Conv2D(64, (3,3), activation='relu', padding='same')
        self.conv12 = Conv2D(64, (3,3), activation='relu', padding='same')
        self.pool1 = MaxPooling2D((2,2), strides=2)

        self.conv21 = Conv2D(128, (3,3), activation='relu', padding='same')
        self.conv22 = Conv2D(128, (3,3), activation='relu', padding='same')
        self.pool2 = MaxPooling2D((2,2), strides=2)

        self.conv31 = Conv2D(256, (3,3), activation='relu', padding='same')
        self.conv32 = Conv2D(256, (3,3), activation='relu', padding='same')
        self.pool3 = MaxPooling2D((2,2), strides=2)

        self.conv41 = Conv2D(512, (3,3), activation='relu', padding='same')
        self.conv42 = Conv2D(512, (3,3), activation='relu', padding='same')
        self.drop4 = Dropout(0.5)                                               # Not in Paper, but is in code
        self.pool4 = MaxPooling2D((2,2), strides=2)

        self.conv51 = Conv2D(1024, (3,3), activation='relu', padding='same')
        self.conv52 = Conv2D(1024, (3,3), activation='relu', padding='same')
        self.drop5 = Dropout(0.5)                                               # Not in Paper, but is in code
        self.upconv5 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')

        #self.concatenate6 = Concatenate([self.upconv5, self.drop4])
        self.conv61 = Conv2D(512, (3,3), activation='relu', padding='same')
        self.conv62 = Conv2D(512, (3,3), activation='relu', padding='same')
        self.upconv6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')

        #self.concatenate7 = Concatenate([self.upconv6, self.conv32])
        self.conv71 = Conv2D(256, (3,3), activation='relu', padding='same')
        self.conv72 = Conv2D(256, (3,3), activation='relu', padding='same')
        self.upconv7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')

        #self.concatenate8 = Concatenate([self.upconv7, self.conv22])
        self.conv81 = Conv2D(128, (3,3), activation='relu', padding='same')
        self.conv82 = Conv2D(128, (3,3), activation='relu', padding='same')
        self.upconv8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')

        #self.concatenate9 = Concatenate([self.upconv8, self.conv12])
        self.conv91 = Conv2D(64, (3,3), activation='relu', padding='same')
        self.conv92 = Conv2D(64, (3,3), activation='relu', padding='same')

        self.conv93 = Conv2D(3, (1,1), activation='softmax', padding='same')

    def call(self, x):
        #x = tf.reshape(x, [1, 512, 512, 1])
        x = self.inputlayer(x)
        x = self.conv11(x)
        x = self.conv12(x)
        conv12 = x
        x = self.pool1(x)

        x = self.conv21(x)
        x = self.conv22(x)
        conv22 = x
        x = self.pool2(x)

        x = self.conv31(x)
        x = self.conv32(x)
        conv32 = x
        x = self.pool3(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.drop4(x)
        drop4 = x
        x = self.pool4(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.drop5(x)
        x = self.upconv5(x)

        x = concatenate([x, drop4])
        x = self.conv61(x)
        x = self.conv62(x)
        x = self.upconv6(x)

        x = concatenate([x, conv32])
        x = self.conv71(x)
        x = self.conv72(x)
        x = self.upconv7(x)

        x = concatenate([x, conv22])
        x = self.conv81(x)
        x = self.conv82(x)
        x = self.upconv8(x)
        x = concatenate([x, conv12])
        x = self.conv91(x)
        x = self.conv92(x)

        x = self.conv93(x)

        return x
