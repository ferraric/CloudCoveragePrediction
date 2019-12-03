import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense
tf.keras.backend.set_floatx('float64')
import os


class SimpleModel(Model):
    def __init__(self, config):
        super(SimpleModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        self.w1 = tf.Variable(tf.random.normal(shape=[90, 59, 121, 2], mean=1.0));
        self.b1 = tf.Variable(tf.random.normal(shape=[90, 59, 121, 2]));

    def call(self, x):
        return tf.math.add(tf.math.multiply(self.w1, x), self.b1)

class LocalModel(Model):
    def __init__(self, config):
        super(LocalModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        #self.loc = LocallyConnected2D(filters=121, kernel_size=[1, 1], use_bias=True, input_shape=(90, 59, 121))
        self.month_embedding=Embedding(12,100, input_length=1)
        self.hours_embedding=Embedding(24,100, input_length=1)
        self.lat_lon_emb=Embedding(5310,150, input_length=1)
        self.init_time_embedding=Embedding(2,150, input_length=1)
        self.dense1 = Dense(300,activation="relu",input_shape=(None,))
        self.dense2=Dense(150,activation="sigmoid",input_shape=(300,))
        self.dense3=Dense(50,activation="sigmoid",input_shape=(150,))
        self.dense4=Dense(21, input_shape=(50, 1))
        #self.dense5=Dense(1,input_shape=(50,1))


    def call(self, x):
        #x = self.loc(x)
        #return x
        m_emb= self.month_embedding(x[:,0]-1)
        h_emb=self.hours_embedding(x[:,1])
        init_time_emb=self.init_time_embedding(x[:,2])
        lat_lon_emb=self.lat_lon_emb(x[:,3])
        concat_all=tf.concat([m_emb,h_emb,init_time_emb,lat_lon_emb,tf.reshape(x[:,4:25],[np.shape(x)[0],21])], 1)
        x2= self.dense1(concat_all)
        x3=self.dense2(x2)
        x4=self.dense3(x3)
        x5=self.dense4(x4)
        #x6=self.dense5(x4)
        return x5#, x6

    def log_model_architecture_to(self, experiment, input_shape):
        self.build(input_shape)
        model_architecture_path = os.path.join(self.config.summary_dir, "model_architecture")
        with open(model_architecture_path, "w") as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.summary(print_fn=lambda x: fh.write(x + "\n"))
        self.summary()
        experiment.log_asset(model_architecture_path)
