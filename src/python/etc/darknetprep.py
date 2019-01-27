import random

import numpy as np

from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.labelparser.YoloParser import DarknetParser
from utils.fileaccess.utils import create_dirs
from utils.image.Backend import crop, resize
from utils.image.Imageprocessing import show
from utils.labels.ImgLabel import ImgLabel
from utils.timing import tic, toc
from utils.workdir import cd_work


def filter(label):
    min_obj_size = 0.01
    max_obj_size = 1.0
    img_res = (416, 416)
    max_angle = 30
    objs_in_size = [obj for obj in label.objects if
                    min_obj_size < (obj.poly.height * obj.poly.width) / (img_res[0] * img_res[1]) < max_obj_size]

    max_aspect_ratio = 1.05 / (max_angle / 90)
    objs_within_angle = [obj for obj in objs_in_size if obj.poly.height / obj.poly.width < max_aspect_ratio]

    objs_in_view = []
    for obj in objs_within_angle:
        mat = np.array([[obj.poly.x_min, obj.poly.y_max],
                        [obj.poly.x_max, obj.poly.y_max]])
        if (len(mat[(mat[:, 0] < 0) | (mat[:, 0] > img_res[1])]) +
            len(mat[(mat[:, 1] < 0) | (mat[:, 1] > img_res[0])])) > 2:
            continue
        objs_in_view.append(obj)

    return ImgLabel(objs_in_size)


project_dir = cd_work()
set_name = 'yolo_43'
out_dir = 'resource/ext/samples/' + set_name
src_dir = 'resource/ext/samples/'

sets = [
    'daylight_course1',
    'daylight_course5',
    'daylight_course3',
    'iros2018_course1',
    'iros2018_course5',
    'iros2018_flights',
    'basement_course3',
    'basement_course1',
    'iros2018_course3_test',
    'various_environments20k',
    'resource/ext/samples/realbg20k'
]

sample_dir = out_dir + '/samples'
create_dirs([out_dir, sample_dir])
batch_size = 8
generator = GateGenerator([src_dir + s for s in sets], batch_size,
                          color_format='bgr', shuffle=False,
                          label_format='xml',
                          img_format='jpg',
                          start_idx=0,
                          valid_frac=0,
                          n_samples=None,
                          filter=filter,
                          max_empty=0.0,
                          forever=False)
reader = generator.generate()
n_images = generator.n_samples

writer = DarknetParser(out_dir + '/samples', color_format='bgr', image_format='jpg', img_norm=(416, 416))
n_images_filtered = 0

for i in range(0, n_images, batch_size):
    tic()
    try:
        batch = next(reader)
        imgs = []
        labels = []
        for img, label, _ in batch:
            img_, label_ = crop(img, (0, 52), (416, 416 - 52), label=label)
            img_, label_ = resize(img_, (240, 320), label=label_)
            show(img_, labels=label_)
            imgs.append(img_)
            labels.append(label_)

        writer.write(imgs, labels)
        toc(str(i) + '/' + str(n_images) + ' samples processed in ')
        n_images_filtered += batch_size
    except StopIteration:
        break

indeces = [i for i in range(n_images_filtered)]
random.shuffle(indeces)
valid_set = indeces[:int(0.1 * len(indeces))]
train_set = indeces[int(0.1 * len(indeces)):]

train_set_path = out_dir + '/train.txt'
valid_set_path = out_dir + '/valid.txt'

with open(valid_set_path, 'w') as f:
    for i in valid_set:
        f.write('{0:s}{1:s}/{2:05d}.jpg\n'.format(project_dir, sample_dir, i))

with open(train_set_path, 'w') as f:
    for i in train_set:
        f.write('{0:s}{1:s}/{2:05d}.jpg\n'.format(project_dir, sample_dir, i))

with open(out_dir + '/' + set_name + '.txt', 'w') as f:
    f.write('classes= 1\n')
    f.write('train= ' + project_dir + train_set_path + '\n')
    f.write('valid= ' + project_dir + valid_set_path + '\n')
    f.write('names= ' + project_dir + '/lib/darknet/data/gate.names.list\n')
    f.write('backup= ' + project_dir + '/out\n')
