from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
tf.keras.backend.set_floatx('float64')
class ExampleModel(tf.Module):
    def __init__(self, config):
        super(ExampleModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        self.month_embedding=Embedding(12,200, input_length=1)
        self.hours_embedding=Embedding(24,200, input_length=1)
        self.lat_lon_emb=Embedding(5310,200, input_length=1)
        self.init_time_embedding=Embedding(2,200, input_length=1)
        self.lead_time_embedding=Embedding(121,200, input_length=1)
        self.dense6= Dense(1000,activation="relu",input_shape=(None,))#,kernel_regularizer=regularizers.l2(0.001))
        self.dense7=Dense(800,activation="relu",input_shape=(1000,))#,kernel_regularizer=regularizers.l2(0.001))
        self.dense3=Dense(1000,activation="relu",input_shape=(None,))#,kernel_regularizer=regularizers.l2(0.001))
        self.dense8=Dense(600,activation="relu",input_shape=(800,))#,kernel_regularizer=regularizers.l2(0.001))
        self.dense9=Dense(400,activation="relu",input_shape=(600,))#,kernel_regularizer=regularizers.l2(0.001))
        self.dense10=Dense(300,activation="relu",input_shape=(400,))#,kernel_regularizer=regularizers.l2(0.001))
        self.dense11=Dense(200,activation="relu",input_shape=(300,))#,kernel_regularizer=regularizers.l2(0.001))
        self.dense12=Dense(150,activation="relu",input_shape=(200,))
        self.dense13=Dense(50,activation="relu",input_shape=(150,))
        self.dense14=Dense(21,input_shape=(50,))

    @tf.function
    def __call__(self,x): 
        m_emb=self.month_embedding(x[:,7]-1)
        h_emb=self.hours_embedding(x[:,8])
        init_time_emb=self.init_time_embedding(x[:,9])
        lat_lon_emb=self.lat_lon_emb(x[:,10])
        lead_time_emb=self.lead_time_embedding(x[:,11])      
        concat_all=tf.concat([m_emb,h_emb,init_time_emb,lat_lon_emb,lead_time_emb,tf.reshape(x[:,0],[np.shape(x)[0],1]),tf.reshape(x[:,1],[np.shape(x)[0],1]),tf.reshape(x[:,2],[np.shape(x)[0],1]),tf.reshape(x[:,3],[np.shape(x)[0],1]),tf.reshape(x[:,4],[np.shape(x)[0],1]),tf.reshape(x[:,5],[np.shape(x)[0],1]),tf.reshape(x[:,6],[np.shape(x)[0],1])], 1)
        x21= self.dense6(concat_all)
        x31=self.dense7(x21)
        x41=self.dense8(x31)
        x51=self.dense9(x41)
        x61=self.dense10(x51)
        x71=self.dense11(x61)
        x81=self.dense12(x71)
        x91=self.dense13(x81)
        x911=self.dense14(x91)
        return x911

