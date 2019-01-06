import os
from os.path import expanduser

import sys


def cd_work():
    print("Platform: ", sys.platform)
    # if sys.platform == 'win32' or sys.platform == 'win64':
    #     wd = 'e:/doc/study/thesis/mavv'
    # else:
    wd = expanduser('~') + '/code/cnn_object_detection'

    os.chdir(wd)
    cur_dir = os.getcwd()
    print("Working Directory: ", cur_dir)
    return cur_dir + '/'

def home():
    return expanduser('~')
