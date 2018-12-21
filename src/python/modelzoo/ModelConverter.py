# manually put back imported modules
import keras.backend as K
from modelzoo.ModelFactory import ModelFactory

from modelzoo.graphconv import convert_model
from modelzoo.models.gatenet.PostprocessLayer import PostprocessLayer


class ModelConverter:

    def __init__(self, model_name, directory, output_format=['tflite', 'pb']):
        self.format = output_format
        self.model_name = model_name
        self.directory = directory

    def finalize(self, quantize=False):
        K.set_learning_phase(0)
        model = ModelFactory.build(self.model_name, batch_size=1, src_dir=self.directory)

        backend = model.net.backend
        out = backend.layers[-1].output
        input = backend.layers[0].input
        postprocessed = PostprocessLayer()(out)
        inference_model = backend  # Model(input, postprocessed)
        sess = K.get_session()
        convert_model(sess, model, out_name=self.directory + self.model_name, out_format=self.format,
                      quantize=quantize, input_shape=model.input_shape)
