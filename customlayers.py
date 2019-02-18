import keras.backend as K
from keras.layers import Layer

class Repeat(Layer):
    """
    Repeat layer to avoid mismatch between a Pooling branch and Convolution branch

    n: Number of repeats (i.e., usually the filter size)
    """
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

class Expand(Layer):
    """
    Expand dimensions of the tensor by 1

    n (optional): size of empty dimension
    """
    def __init__(self, n=1):
        self.n = n
        super(Expand, self).__init__()

    def build(self, input_shape):
        super(Expand, self).build(input_shape)

    def call(self, x):
        y = K.expand_dims(x, axis=2)
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n)
