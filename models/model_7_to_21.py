from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
tf.keras.backend.set_floatx('float64')
class Model_7_to_21(tf.Module):
       def __init__(self, config):
        super(Model_7_to_21, self).__init__()
        self.config = config
        self.build_model()

       def build_model(self):
        emb_size_month=self.config["emb_size_month"]
        emb_size_hours=self.config["emb_size_hour"]
        emb_size_lat_lon=self.config["emb_size_lat_lon"]
        emb_size_init_time=self.config["emb_size_init_time"]
        emb_size_lead_time=self.config["emb_size_lead_time"]
        layer_1=self.config["layer1"]
        layer_2=self.config["layer2"]
        layer_3=self.config["layer3"]
        layer_4=self.config["layer4"]
        layer_5=self.config["layer5"]
        layer_6=self.config["layer6"]
        layer_7=self.config["layer7"]
        layer_8=self.config["layer8"]
        layer_9=self.config["layer9"]
        layer_10=self.config["layer10"]
        self.month_embedding=Embedding(12,emb_size_month, input_length=1)
        self.hours_embedding=Embedding(24,emb_size_hours, input_length=1)
        self.lat_lon_emb=Embedding(5310,emb_size_lat_lon, input_length=1)
        self.init_time_embedding=Embedding(2,emb_size_init_time, input_length=1)
        self.lead_time_embedding=Embedding(121,emb_size_lead_time, input_length=1)
        self.model=tf.keras.Sequential()
        self.model.add(Dense(layer_1,activation="relu",input_shape=(None,)))
        self.model.add(Dense(layer_2,activation="relu",input_shape=(layer_1,)))
        self.model.add(Dense(layer_3,activation="relu",input_shape=(layer_2,)))
        self.model.add(Dense(layer_4,activation="relu",input_shape=(layer_3,)))
        self.model.add(Dense(layer_5,activation="relu",input_shape=(layer_4,)))
        self.model.add(Dense(layer_6,activation="relu",input_shape=(layer_5,)))
        self.model.add(Dense(layer_7,activation="relu",input_shape=(layer_6,)))
        self.model.add(Dense(layer_8,activation="relu",input_shape=(layer_7,)))
        self.model.add(Dense(layer_9,activation="relu",input_shape=(layer_8,)))
        self.model.add(Dense(layer_10,input_shape=(layer_9,)))
       @tf.function
       def __call__(self,x): 
        m_emb=self.month_embedding(x[:,7]-1)
        h_emb=self.hours_embedding(x[:,8])
        init_time_emb=self.init_time_embedding(x[:,9])
        lat_lon_emb=self.lat_lon_emb(x[:,10])
        lead_time_emb=self.lead_time_embedding(x[:,11])      
        concat_all=tf.concat([init_time_emb*m_emb,init_time_emb*lead_time_emb,m_emb,h_emb,init_time_emb,lat_lon_emb,lead_time_emb,tf.reshape(x[:,0],[np.shape(x)[0],1]),tf.reshape(x[:,1],[np.shape(x)[0],1]),tf.reshape(x[:,2],[np.shape(x)[0],1]),tf.reshape(x[:,3],[np.shape(x)[0],1]),tf.reshape(x[:,4],[np.shape(x)[0],1]),tf.reshape(x[:,5],[np.shape(x)[0],1]),tf.reshape(x[:,6],[np.shape(x)[0],1])], 1)
        out = self.model(concat_all)
        return out
