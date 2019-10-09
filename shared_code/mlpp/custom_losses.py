from keras import backend as K


def gaussian_log_likelihood(y_true, y_pred):
    return -y_pred[1::2] - (y_true - y_pred[::2])**2/(2*K.exp(2*y_pred[1::2]))

