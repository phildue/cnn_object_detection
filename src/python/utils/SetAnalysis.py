import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.labels.Pose import Pose
from visuals.plots.BaseHist import BaseHist
from visuals.plots.BaseMultiPlot import BaseMultiPlot
from visuals.plots.BoxPlot import BoxPlot
from visuals.plots.Heatmap import Heatmap


class SetAnalysis:
    def __init__(self, img_shape, path):
        self.img_shape = img_shape
        if isinstance(path, list):
            self.labels = []
            for p in path:
                self.label_reader = DatasetParser.get_parser(p, 'xml', color_format='bgr')
                self.labels.extend(self.label_reader.read()[1])
        else:
            self.label_reader = DatasetParser.get_parser(path, 'xml', color_format='bgr')
            _, self.labels = self.label_reader.read()

    def get_label_map(self):

        y_limit = self.img_shape[0]
        label_sum = np.zeros(self.img_shape)
        for l in self.labels:
            label_map = np.zeros(self.img_shape)
            for o in l.objects:
                y_max = y_limit - o.poly.y_min
                y_min = y_limit - o.poly.y_max
                label_map[int(min((y_max, y_min))):int(max((y_max, y_min))), int(o.poly.x_min): int(o.poly.x_max)] = 1
            label_sum += label_map
        return label_sum

    def get_heatmap(self):
        return Heatmap(self.get_label_map(), x_label="Width", y_label="Height",
                       title="Heatmap of Object Locations/Sizes")

    def mean_n_objects(self):
        n_objects = 0
        for l in self.labels:
            n_objects += len(l.objects)

        print('Total Objects:' + str(n_objects))
        return n_objects / len(self.labels)

    def area_distribution(self):
        boxplot = BoxPlot(x_data=self.get_area(), y_label='Area', x_label='',
                          title='Distribution of Bounding Box Areas')
        return boxplot

    def area_distribution_hist(self):
        boxplot = BaseHist(y_data=self.get_area(), x_label='Area', y_label='Number of Boxes',
                           title='Histogram of Bounding Box Areas', n_bins=10)
        return boxplot

    def get_area(self):
        box_dims = self.get_box_dims()
        print(len(box_dims))
        area = box_dims[:, 0] * box_dims[:, 1]
        return area

    def get_box_dims(self):
        wh = []
        for label in self.labels:
            h, w = self.img_shape
            for b in label.objects:
                if 0 < b.poly.width < w\
                    and 0 < b.poly.height < h:
                    box_dim = np.array([b.poly.width, b.poly.height])
                    box_dim = np.expand_dims(box_dim, 0)
                    wh.append(box_dim)
        box_dims = np.concatenate(wh, 0)
        return box_dims

    def kmeans_anchors(self, n_boxes=5):
        box_dims = self.get_box_dims()
        # TODO move this to backend
        kmeans = KMeans(n_clusters=n_boxes).fit(box_dims)
        centers = kmeans.cluster_centers_
        scatter = BaseMultiPlot(x_data=[box_dims[:, 0], centers[:, 0]], y_data=[box_dims[:, 1], centers[:, 1]],
                                line_style=['bx', 'ro'],
                                x_label='width', y_label='height',y_lim=(0, self.img_shape[0]))
        return scatter, kmeans

    def show_summary(self):
        heat_map = self.get_heatmap()
        box_dims, _ = self.kmeans_anchors()
        mean_samples = self.mean_n_objects()
        area_distribution = self.area_distribution_hist()
        print("Mean samples: ", mean_samples)
        box_dims.show(False)
        heat_map.show(False)
        area_distribution.show(True)

    @staticmethod
    def assign_to_bin(pose: Pose, angle: str, angle_bins, distance_bins):
        dist = np.linalg.norm(pose.transvec / 3)

        if angle is 'yaw':
            angle_v = pose.yaw
        elif angle is 'roll':
            angle_v = pose.roll
        elif angle is 'pitch':
            angle_v = pose.pitch
        angle_v = np.degrees(angle_v)
        return SetAnalysis.assign_angle_dist_to_bin(angle_v,dist,angle_bins,distance_bins)

    @staticmethod
    def assign_angle_dist_to_bin(angle, dist, angle_bins, distance_bins):
        for i_dist in range(len(distance_bins) - 1):
            if distance_bins[i_dist] <= dist < distance_bins[i_dist + 1]:
                break
        for i_angle in range(len(angle_bins) - 1):
            if angle_bins[i_angle] <= angle < angle_bins[i_angle + 1]:
                break

        return i_dist, i_angle

    def pose_cluster(self, angle_bins, distance_bins):

        bins = np.zeros((len(distance_bins), len(angle_bins), 3))
        for i_angle, angle in enumerate(['yaw', 'pitch', 'roll']):
            for p in self.get_poses():
                bin_dist, bin_angle = SetAnalysis.assign_to_bin(p, angle, angle_bins, distance_bins)
                bins[bin_dist, bin_angle, i_angle] += 1

        return bins

    def get_poses(self):
        objects = []
        for l in self.labels:
            objects.extend(l.objects)
        poses = [o.pose for o in objects]
        return poses

    @staticmethod
    def show_pose_hist(poses,show=True):
        plt.title("Object Occurences in 3D", fontsize=12)
        yaws = np.array([np.degrees(p.yaw) for p in poses])
        yaws[yaws < 0] *= -1
        yaws[yaws > 180] = 360 - yaws[yaws > 180]
        yaws = 180 - yaws
        dists = [np.linalg.norm(p.transvec / 3) for p in poses]
        plt.hist2d(yaws, dists, bins=10)
        plt.colorbar()
        plt.xlabel("Relative Yaw Angle")
        plt.ylabel("Relative Distance")
        plt.show(show)
