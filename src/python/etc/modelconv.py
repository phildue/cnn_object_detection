from modelzoo.ModelConverter import ModelConverter
from utils.workdir import cd_work

cd_work()
quantize = False

ModelConverter('simple_tf', 'out/test/').finalize(quantize)
