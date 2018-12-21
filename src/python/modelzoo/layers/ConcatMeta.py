import keras.backend as K
from keras.engine.topology import Layer


class ConcatMeta(Layer):
    def __init__(self, meta_t, **kwargs):
        self.meta_t = meta_t
        super(ConcatMeta, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(ConcatMeta, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        meta_t = K.expand_dims(self.meta_t, 0)
        meta_t = K.tile(meta_t, (K.shape(x)[0], 1, 1))
        return K.concatenate([x, meta_t])

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2] + 6
