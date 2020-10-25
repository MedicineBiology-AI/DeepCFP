import numpy as np
import h5py
from keras import backend as K
from keras import regularizers
from keras.layers import Layer
    
class AttentionLayer(Layer): 
    def __init__(self, **kwargs): 
        super(AttentionLayer, self).__init__(** kwargs) 
        
    def build(self, input_shape): 
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[-1], 1), 
                                 initializer='uniform', 
                                 regularizer=regularizers.l2(1e-5),
                                 trainable=True) 
        
        self.b = self.add_weight(name='att_bias', 
                                 shape=(1,), 
                                 initializer='uniform',
                                 regularizer=regularizers.l2(1e-5),
                                 trainable=True) 

        super(AttentionLayer, self).build(input_shape) 
        
    def call(self, inputs):
        f = K.tanh(K.dot(inputs, self.W) + self.b)
        a = K.softmax(K.batch_flatten(f)) 
        outputs = inputs*K.permute_dimensions(K.repeat(a, inputs.shape[-1]), [0, 2, 1])
        outputs = K.sum(outputs, axis=1)
        return outputs
    
    def compute_output_shape(self, input_shape): 
        return input_shape[0], input_shape[2]