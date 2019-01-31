import keras.backend as K
from keras.layers import Layer

class Repeat(Layer):
    def __init__(self, n=1):
        self.n = n
        super(Repeat, self).__init__()

    def build(self, input_shape):
        super(Repeat, self).build(input_shape)

    def call(self, x):
        y = K.squeeze(x, axis=2)
        y = K.repeat(y, self.n)
        y = K.permute_dimensions(y, pattern=(0,2,1))
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n)
