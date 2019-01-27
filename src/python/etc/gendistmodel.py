from utils.fileaccess.utils import load_file
from utils.image.BarrelDistortion import BarrelDistortion
from utils.workdir import cd_work

cd_work()
# dataset = GateGenerator(directories=['resource/ext/samples/daylight_course1/'],
#                         batch_size=100, color_format='bgr',
#                         shuffle=False, start_idx=0, valid_frac=0,
#                         label_format='xml').generate()
# batch = next(dataset)

camera_calibration = load_file('resource/camera_calibration/jevois.pkl')

distortion = camera_calibration.distortion[0]
k1 = distortion[0]
k2 = distortion[1]
p1 = distortion[2]
p2 = distortion[3]
# k1 = 0.1
# k2 = 0
# p1 = 0.1
# p2 = 0

distortion_model = BarrelDistortion(img_shape=(240, 320),
                                    rad_dist_params=[k1, k2],
                                    tangential_dist_params=[p1, p2],
                                    max_iterations=100,
                                    distortion_radius=1,
                                    conv_thresh=0.0000001)
distortion_model.save('resource/camera_calibration/distortion_model_est.pkl')

# distortion_model = BarrelDistortion.from_file('resource/camera_calibration/distortion_model_est.pkl')
# for img, label, _ in batch:
#     img, label = resize(img, (240, 320), label=label)
#
#     tic()
#     img_d, label_d = distortion_model.undistort(img, label)
#
#     # img, label = resize(img, (320, 624), label=label)
#     # img_d, label_d = resize(img_d, (320, 624), label=label_d)
#     show(img, labels=label, t=1)
#     show(img_d, labels=label_d, name='undistorted')
