from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf

        
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

