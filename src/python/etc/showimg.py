from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.VocGenerator import VocGenerator
from utils.image.Backend import resize
from utils.image.Imageprocessing import show, LEGEND_POSITION
from utils.labels.ImgLabel import ImgLabel
from utils.workdir import cd_work

cd_work()


def show_voc():
    dataset = VocGenerator(batch_size=2, classes=['cat']).generate()
    for batch in dataset:
        for img, label, _ in batch:
            # img, label = resize(img, shape=(416, 416), label=label)
            show(img, 'resized', labels=label)


def show_img(path):
    def filter(label):

        objs_in_size = [obj for obj in label.objects if
                        0.01 < (obj.height * obj.width) / (416 * 416) < 1.2]

        max_aspect_ratio = 1.05 / (30 / 90)
        objs_within_angle = [obj for obj in objs_in_size if obj.height / obj.width < max_aspect_ratio]

        objs_in_view = []
        for obj in objs_within_angle:
            mat = obj.gate_corners.mat
            if (len(mat[(mat[:, 0] < 0) | (mat[:, 0] > 416)]) +
                len(mat[(mat[:, 1] < 0) | (mat[:, 1] > 416)])) > 2:
                continue
            objs_in_view.append(obj)

        return ImgLabel(objs_in_size)

    gate_generator = GateGenerator(path, 8, color_format='bgr', shuffle=False, label_format='xml', img_format='jpg',
                                   filter=None, remove_filtered=False, start_idx=0)

    for batch in gate_generator.generate():
        for img, label, _ in batch:
            # img, label = resize(img, (208, 208), label=label)
            print(label)
            show(img, 'img', labels=label, legend=LEGEND_POSITION, thickness=1)


def show_shot(path="samplegen/resource/shots/stream/"):
    shots, labels = ShotLoad(shot_path=path, img_format='bmp').get_shots(
        n_shots=200)
    for i, img in enumerate(shots):
        ann_img, label = resize(img, (500, 500), label=labels[i])
        show(ann_img, 'labeled', labels=label)


# show_shot(path="samplegen/resource/ext/samples/bebop_merge/")
show_img(path=['resource/ext/samples/iros2018_course_final_simple_17gates'])
# show_voc()
