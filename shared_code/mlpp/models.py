import keras
from keras import layers


def shallow_mlp(n_inputs, n_outputs):
    inputs = keras.Input(shape=(n_inputs,), name='features')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(3, name='outputs')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
