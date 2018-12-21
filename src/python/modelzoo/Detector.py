from modelzoo.Postprocessor import Postprocessor
from modelzoo.Preprocessor import Preprocessor
from utils.ModelSummary import ModelSummary


class Detector:

    def __init__(self, model, preprocessor: Preprocessor, postprocessor: Postprocessor, summary: ModelSummary = None):
        self.summary = summary
        self.postprocessor = postprocessor
        self.preprocessor = preprocessor
        self.model = model

    def detect(self, sample):
        if not isinstance(sample, list):
            sample = [sample]
        return self.postprocessor.postprocess(self.model.predict(self.preprocessor.preprocess_batch(sample)))
