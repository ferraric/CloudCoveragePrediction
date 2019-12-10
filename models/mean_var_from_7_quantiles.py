from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
tf.keras.backend.set_floatx('float64')
class Model_mean_var(tf.Module):
       def __init__(self, config):
        super(Model_mean_var, self).__init__()
        self.config = config
        self.build_model()

       def build_model(self):
        config=self.config
        emb_size_month=config["emb_size_month"]
        emb_size_hours=config["emb_size_hour"]
        emb_size_lat_lon=config["emb_size_lat_lon"]
        emb_size_init_time=config["emb_size_init_time"]
        emb_size_lead_time=config["emb_size_lead_time"]
        layer_1=config["layer1"]
        layer_2=config["layer2"]
        layer_3=config["layer3"]
        layer_4=config["layer4"]
        layer_5=config["layer5"]
        layer_6=config["layer6"]
        layer_7=config["layer7"]
        layer_8=config["layer8"]
        layer_9=config["layer9"]
        layer_10=config["layer10"]
        self.month_embedding=Embedding(12,emb_size_month, input_length=1)
        self.hours_embedding=Embedding(24,emb_size_hours, input_length=1)
        self.lat_lon_emb=Embedding(5310,emb_size_lat_lon, input_length=1)
        self.init_time_embedding=Embedding(2,emb_size_init_time, input_length=1)
        self.lead_time_embedding=Embedding(121,emb_size_lead_time, input_length=1)
        self.dense11= Dense(layer_1,activation="relu",input_shape=(None,))
        self.dense12=Dense(layer_2,activation="relu",input_shape=(layer_1,))
        self.dense13=Dense(layer_3,activation="relu",input_shape=(layer_2,))
        self.dense14=Dense(layer_4,activation="relu",input_shape=(layer_3,))
        self.dense15=Dense(layer_5,activation="relu",input_shape=(layer_4,))
        self.dense16=Dense(layer_6,activation="relu",input_shape=(layer_5,))
        self.dense17=Dense(layer_7,activation="relu",input_shape=(layer_6,))
        self.dense18=Dense(layer_8,activation="relu",input_shape=(layer_7,))
        self.dense19=Dense(layer_9,activation="relu",input_shape=(layer_8,))
        self.dense110=Dense(layer_10,input_shape=(layer_9,))
        self.dense21= Dense(layer_1,activation="relu",input_shape=(None,))
        self.dense22=Dense(layer_2,activation="relu",input_shape=(layer_1,))
        self.dense23=Dense(layer_3,activation="relu",input_shape=(layer_2,))
        self.dense24=Dense(layer_4,activation="relu",input_shape=(layer_3,))
        self.dense25=Dense(layer_5,activation="relu",input_shape=(layer_4,))
        self.dense26=Dense(layer_6,activation="relu",input_shape=(layer_5,))
        self.dense27=Dense(layer_7,activation="relu",input_shape=(layer_6,))
        self.dense28=Dense(layer_8,activation="relu",input_shape=(layer_7,))
        self.dense29=Dense(layer_9,activation="relu",input_shape=(layer_8,))
        self.dense210=Dense(layer_10,input_shape=(layer_9,))
       @tf.function
       def __call__(self,x): 
        m_emb=self.month_embedding(x[:,7]-1)
        h_emb=self.hours_embedding(x[:,8])
        init_time_emb=self.init_time_embedding(x[:,9])
        lat_lon_emb=self.lat_lon_emb(x[:,10])
        lead_time_emb=self.lead_time_embedding(x[:,11])      
        concat_all=tf.concat([m_emb,h_emb,init_time_emb,lat_lon_emb,lead_time_emb,tf.reshape(x[:,0],[np.shape(x)[0],1]),tf.reshape(x[:,1],[np.shape(x)[0],1]),tf.reshape(x[:,2],[np.shape(x)[0],1]),tf.reshape(x[:,3],[np.shape(x)[0],1]),tf.reshape(x[:,4],[np.shape(x)[0],1]),tf.reshape(x[:,5],[np.shape(x)[0],1]),tf.reshape(x[:,6],[np.shape(x)[0],1])], 1)
        x11= self.dense11(concat_all)
        x12=self.dense12(x11)
        x13=self.dense13(x12)
        x14=self.dense14(x13)
        x15=self.dense15(x14)
        x16=self.dense16(x15)
        x17=self.dense17(x16)
        x18=self.dense18(x17)
        x19=self.dense19(x18)
        x110=self.dense110(x19)
        x21= self.dense21(concat_all)
        x22=self.dense22(x21)
        x23=self.dense23(x22)
        x24=self.dense24(x23)
        x25=self.dense25(x24)
        x26=self.dense26(x25)
        x27=self.dense27(x26)
        x28=self.dense28(x27)
        x29=self.dense29(x28)
        x210=self.dense210(x29)
        return x110,x210