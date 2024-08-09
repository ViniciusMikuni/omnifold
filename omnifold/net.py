from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss

    t_loss = weights*tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(t_loss)
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = 1e-9
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * tf.math.log(y_pred) +
                         (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(t_loss)

        
def MLP(nvars,
        layer_sizes = [64,128,64],
        activation = 'gelu'):
    
    ''' Define a simple fully conneted model to be used during unfolding'''
    inputs = Input((nvars, ))
    layer = Dense(layer_sizes[0],activation=activation)(inputs)
    for layer_size in layer_sizes[1:]:
        layer = Dense(layer_size,activation=activation)(layer)
        
    outputs = Dense(1)(layer)
    model = Model(inputs = inputs, outputs = outputs)
    return model

