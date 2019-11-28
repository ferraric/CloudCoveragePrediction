import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Conv3D, MaxPooling3D, Conv3DTranspose, concatenate, Dropout, BatchNormalization


class UNetModelQ(Model):
    def __init__(self, config):
        super(UNetModelQ, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        self.inputlayer = InputLayer(input_shape=(self.config.batch_size, 121, 59, 90, 7))

        self.conv11 = Conv3D(filters=32, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc11 = BatchNormalization(axis=-1)
        self.conv12 = Conv3D(filters=32, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc12 = BatchNormalization(axis=-1)
        self.pool1 = MaxPooling3D(pool_size=(2,2,2), strides=2)

        self.conv21 = Conv3D(filters=64, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc21 = BatchNormalization(axis=-1)
        self.conv22 = Conv3D(filters=64, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc22 = BatchNormalization(axis=-1)
        self.pool2 = MaxPooling3D(pool_size=(2,2,2), strides=2)

        self.conv31 = Conv3D(filters=128, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc31 = BatchNormalization(axis=-1)
        self.conv32 = Conv3D(filters=128, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc32 = BatchNormalization(axis=-1)
        self.pool3 = MaxPooling3D(pool_size=(2,2,2), strides=2)

        self.conv41 = Conv3D(filters=256, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc41 = BatchNormalization(axis=-1)
        self.conv42 = Conv3D(filters=256, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc42 = BatchNormalization(axis=-1)
        self.drop4 = Dropout(rate=0.5)                                               # Not in Paper, but is in code
        self.pool4 = MaxPooling3D(pool_size=(2,2,2), strides=2)

        self.conv51 = Conv3D(filters=512, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc51 = BatchNormalization(axis=-1)
        self.conv52 = Conv3D(filters=512, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc52 = BatchNormalization(axis=-1)
        self.drop5 = Dropout(rate=0.5)                                               # Not in Paper, but is in code
        self.upconv5 = Conv3DTranspose(filters=256, kernel_size=(2,2,2), strides=(2,2,2), padding='valid',
                                       output_padding=(1,1,1))

        #self.concatenate6 = Concatenate([self.upconv5, self.drop4])
        self.conv61 = Conv3D(filters=256, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc61 = BatchNormalization(axis=-1)
        self.conv62 = Conv3D(filters=256, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc62 = BatchNormalization(axis=-1)
        self.upconv6 = Conv3DTranspose(filters=128, kernel_size=(2,2,2), strides=(2,2,2), padding='same')

        #self.concatenate7 = Concatenate([self.upconv6, self.conv32])
        self.conv71 = Conv3D(filters=128, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc71 = BatchNormalization(axis=-1)
        self.conv72 = Conv3D(filters=128, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc72 = BatchNormalization(axis=-1)
        self.upconv7 = Conv3DTranspose(filters=64, kernel_size=(2,2,2), strides=(2,2,2), padding='valid',
                                       output_padding=(0,1,1))

        #self.concatenate8 = Concatenate([self.upconv7, self.conv22])
        self.conv81 = Conv3D(filters=64, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc81 = BatchNormalization(axis=-1)
        self.conv82 = Conv3D(filters=64, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc82 = BatchNormalization(axis=-1)
        self.upconv8 = Conv3DTranspose(filters=32, kernel_size=(2,2,2), strides=(2,2,2), padding='valid',
                                       output_padding=(1,1,0))

        #self.concatenate9 = Concatenate([self.upconv8, self.conv12])
        self.conv91 = Conv3D(filters=32, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc91 = BatchNormalization(axis=-1)
        self.conv92 = Conv3D(filters=32, kernel_size=(3,3,3), activation='relu', padding='same')
        self.batc92= BatchNormalization(axis=-1)

        self.conv93 = Conv3D(filters=21, kernel_size=(1,1,1), padding='same')

    def call(self, x):
        x = self.inputlayer(x)
        x = self.conv11(x)
        x = self.batc11(x)
        x = self.conv12(x)
        x = self.batc12(x)
        conv12 = x
        x = self.pool1(x)

        x = self.conv21(x)
        x = self.batc21(x)
        x = self.conv22(x)
        x = self.batc22(x)
        conv22 = x
        x = self.pool2(x)

        x = self.conv31(x)
        x = self.batc31(x)
        x = self.conv32(x)
        x = self.batc32(x)
        conv32 = x
        x = self.pool3(x)

        x = self.conv41(x)
        x = self.batc41(x)
        x = self.conv42(x)
        x = self.batc42(x)
        x = self.drop4(x)
        drop4 = x
        x = self.pool4(x)

        x = self.conv51(x)
        x = self.batc51(x)
        x = self.conv52(x)
        x = self.batc52(x)
        x = self.drop5(x)
        x = self.upconv5(x)

        x = concatenate([x, drop4])
        x = self.conv61(x)
        x = self.batc61(x)
        x = self.conv62(x)
        x = self.batc62(x)
        x = self.upconv6(x)

        x = concatenate([x, conv32])
        x = self.conv71(x)
        x = self.batc71(x)
        x = self.conv72(x)
        x = self.batc72(x)
        x = self.upconv7(x)

        x = concatenate([x, conv22])
        x = self.conv81(x)
        x = self.batc81(x)
        x = self.conv82(x)
        x = self.batc82(x)
        x = self.upconv8(x)
        x = concatenate([x, conv12])
        x = self.conv91(x)
        x = self.batc91(x)
        x = self.conv92(x)
        x = self.batc92(x)

        x = self.conv93(x)

        return x
