import os
import tensorflow as tf
import numpy as np
tf.keras.backend.set_floatx('float64')
class ExampleModel(Model):
    def __init__(self, config):
        super(ExampleModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        self.month_embedding=Embedding(12,100, input_length=1)
        self.hours_embedding=Embedding(24,100, input_length=1)
        self.lat_lon_emb=Embedding(5310,150, input_length=1)
        self.init_time_embedding=Embedding(2,150, input_length=1)
        self.dense1 = Dense(300,activation="relu",input_shape=(None,))
        self.dense2=Dense(150,activation="sigmoid",input_shape=(300,))
        self.dense3=Dense(50,activation="sigmoid",input_shape=(150,))
        self.dense4=Dense(1,input_shape=(50,1))
        self.dense5=Dense(1,input_shape=(50,1))
        #self.month_embedding2=Embedding(12,100, input_length=1)
        #self.hours_embedding2=Embedding(24,100, input_length=1)
        #self.lat_lon_emb2=Embedding(5310,150, input_length=1)
        #self.init_time_embedding2=Embedding(2,150, input_length=1)
        #self.dense12= Dense(300,activation="relu",input_shape=(None,))
        #self.dense22=Dense(150,activation="sigmoid",input_shape=(300,))
        #self.dense32=Dense(50,activation="sigmoid",input_shape=(150,))
        #self.dense42=Dense(1,input_shape=(50,1))
        #pass
    def call(self, x):
        #inputs_train=np.concatenate([CLCT_train,months_train,hours_train,init_time_train],axis=1)
        m_emb= self.month_embedding(x[:,2]-1)
        h_emb=self.hours_embedding(x[:,3])
        init_time_emb=self.init_time_embedding(x[:,4])
        lat_lon_emb=self.lat_lon_emb(x[:,5])
        concat_all=tf.concat([m_emb,h_emb,init_time_emb,lat_lon_emb,tf.reshape(x[:,0],[np.shape(x)[0],1]),tf.reshape(x[:,1],[np.shape(x)[0],1])], 1)
        x2= self.dense1(concat_all)
        x3=self.dense2(x2)
        x4=self.dense3(x3)
        x5=self.dense4(x4)
        x6=self.dense5(x4)
        #m_emb2= self.month_embedding2(x[:,1]-1)
        #h_emb2=self.hours_embedding2(x[:,2])
        #init_time_emb2=self.init_time_embedding2(x[:,3])
        #lat_lon_emb2=self.lat_lon_emb2(x[:,4])
        #concat_all2=tf.concat([m_emb2,h_emb2,init_time_emb2,lat_lon_emb2,tf.reshape(x[:,0],[np.shape(x)[0],1])], 1)
        #x22= self.dense12(concat_all2)
        #x32=self.dense22(x22)
        #x42=self.dense32(x32)
        #x52=self.dense42(x42)
        #x=tf.reshape(x[:,0],[np.shape(x)[0],1])
        return x5,x6

    def log_model_architecture_to(self, experiment, input_shape):
        self.build(input_shape)
        model_architecture_path = os.path.join(self.config.summary_dir, "model_architecture")
        with open(model_architecture_path, "w") as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.summary(print_fn=lambda x: fh.write(x + "\n"))
        print(self.summary())
        experiment.log_asset(model_architecture_path)