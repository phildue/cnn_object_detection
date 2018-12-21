from samplegen.labelmaker.LabelMaker import LabelMaker

from utils.fileaccess.utils import create_dirs
from utils.workdir import cd_work

cd_work()

img_dir = 'resource/ext/samples/real_test/'
src_file = 'resource/video/eth.avi'
set_dir = 'resource/ext/samples/real_test_labeled/'

create_dirs([img_dir, set_dir])

LabelMaker(None, img_dir, step_size=1, set_dir=set_dir, start_idx=0, label_format='xml', image_format_in='bmp').parse()
