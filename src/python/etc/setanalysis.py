from utils.SetAnalysis import SetAnalysis
from utils.workdir import cd_work

cd_work()
s = SetAnalysis((416,416),'resource/ext/samples/various_environments20k/')
s.show_summary()

