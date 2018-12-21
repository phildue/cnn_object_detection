import numpy as np

from evaluation.DetectionEvaluator import DetectionEvaluator
from evaluation.DetectionResult import DetectionResult
from modelzoo.build_model import load_detector
from utils.ModelSummary import ModelSummary
from utils.SetAnalysis import SetAnalysis
from utils.fileaccess.GateGenerator import GateGenerator
from utils.fileaccess.utils import save_file, create_dirs, load_file
from utils.imageprocessing.Backend import imread
from utils.imageprocessing.Imageprocessing import show
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel
from utils.timing import tic, toc


def evalcluster_yaw_dist(labels_true, labels_pred, conf_thresh, n_bins_angle=10, n_bins_dist=10, iou_thresh=0.6):
    angles = np.linspace(0, 180, n_bins_angle)
    distances = np.linspace(0, 12, n_bins_dist)
    tp = np.zeros((n_bins_dist, n_bins_angle))
    fn = np.zeros((n_bins_dist, n_bins_angle))

    for i in range(len(labels_true)):
        m = DetectionEvaluator(min_box_area=0.001 * 416 * 416, max_box_area=2.0 * 416 * 416,
                               min_aspect_ratio=0,
                               max_aspect_ratio=100.0, iou_thresh=iou_thresh)
        label_pred = ImgLabel([obj for obj in labels_pred[i].objects if obj.confidence > conf_thresh])
        label_true = labels_true[i]
        m.evaluate(label_true, label_pred)

        for b in m.boxes_fn:
            yaw = np.degrees(b.pose.yaw)
            if yaw < 0:
                yaw *= -1
            if yaw > 180:
                yaw = 360 - yaw
            yaw = 180 - yaw
            dist = np.linalg.norm(b.pose.transvec / 3)
            i, j = SetAnalysis.assign_angle_dist_to_bin(yaw, dist, angles, distances)
            fn[i, j] += 1
        for i_b, b in enumerate(m.boxes_true):
            if i_b not in m.false_negatives_idx:
                yaw = np.degrees(b.pose.yaw)
                if yaw < 0:
                    yaw *= -1
                if yaw > 180:
                    yaw = 360 - yaw
                yaw = 180 - yaw
                dist = np.linalg.norm(b.pose.transvec / 3)
                i, j = SetAnalysis.assign_angle_dist_to_bin(yaw, dist, angles, distances)
                tp[i, j] += 1

    return fn, tp


def evalcluster_height_width(labels_true, labels_pred, conf_thresh, n_bins, iou_thresh=0.6):
    widths = np.linspace(0, 416, n_bins)
    heights = np.linspace(0, 416, n_bins)
    tp = np.zeros((n_bins, n_bins))
    fp = np.zeros((n_bins, n_bins))
    fn = np.zeros((n_bins, n_bins))
    evaluator = DetectionEvaluator(min_box_area=0.0 * 416 * 416, max_box_area=2.0 * 416 * 416,
                                   min_aspect_ratio=0,
                                   max_aspect_ratio=100.0, iou_thresh=iou_thresh)
    for i in range(len(labels_true)):

        label_pred = ImgLabel([obj for obj in labels_pred[i].objects if obj.confidence > conf_thresh])
        label_true = labels_true[i]
        evaluator.evaluate(label_true, label_pred)

        for b in evaluator.boxes_fn:
            w = b.poly.width
            h = b.poly.height
            for i_w in range(len(widths) - 1):
                if widths[i_w] <= w < widths[i_w + 1]:
                    break
            for i_h in range(len(heights) - 1):
                if heights[i_h] <= h < heights[i_h + 1]:
                    break

            fn[i_w, i_h] += 1

        for b in evaluator.boxes_fp:
            w = b.poly.width
            h = b.poly.height
            for i_w in range(len(widths) - 1):
                if widths[i_w] <= w < widths[i_w + 1]:
                    break
            for i_h in range(len(heights) - 1):
                if heights[i_h] <= h < heights[i_h + 1]:
                    break

            fp[i_w, i_h] += 1

        for b in evaluator.boxes_tp:
            w = b.poly.width
            h = b.poly.height
            for i_w in range(len(widths) - 1):
                if widths[i_w] <= w < widths[i_w + 1]:
                    break
            for i_h in range(len(heights) - 1):
                if heights[i_h] <= h < heights[i_h + 1]:
                    break

            tp[i_w, i_h] += 1

    return tp, fp, fn


def evalcluster_size(labels_true, labels_pred, conf_thresh, n_bins, iou_thresh=0.6, min_size=0, max_size=416):
    size = np.linspace(min_size, max_size, n_bins)
    tp = np.zeros((n_bins,))
    fp = np.zeros((n_bins,))
    fn = np.zeros((n_bins,))
    evaluator = DetectionEvaluator(min_box_area=0.0 * 416 * 416, max_box_area=2.0 * 416 * 416,
                                   min_aspect_ratio=0,
                                   max_aspect_ratio=100.0, iou_thresh=iou_thresh)
    for i in range(len(labels_true)):

        label_pred = ImgLabel([obj for obj in labels_pred[i].objects if obj.confidence > conf_thresh])
        label_true = labels_true[i]
        evaluator.evaluate(label_true, label_pred)

        for b in evaluator.boxes_fn:
            a = b.poly.area
            matched = False
            for i_w in range(len(size) - 1):
                if a < size[0]:
                    print('Area too small')
                    break

                if size[i_w] <= a < size[i_w + 1]:
                    matched = True
                    break
            if matched:
                fn[i_w] += 1
            else:
                print('FN Not matched: {}'.format(b))
                print(a)

        for b in evaluator.boxes_fp:
            a = b.poly.area
            matched = False
            for i_w in range(len(size) - 1):
                if a < size[0]:
                    print('Area too small')
                    break
                if size[i_w] <= a < size[i_w + 1]:
                    matched = True
                    break
            if matched:
                fp[i_w] += 1
            else:
                print('FP Not matched: {}'.format(b))
                print(a)
        for b in evaluator.boxes_tp:
            a = b.poly.area
            matched = False
            for i_w in range(len(size) - 1):
                if a < size[0]:
                    print('Area too small')
                    break
                if size[i_w] <= a < size[i_w + 1]:
                    matched = True
                    break
            if matched:
                tp[i_w] += 1
            else:
                print('TP Not matched: {}'.format(b))
                print(a)
    return tp, fp, fn


def evalscatter_wh(labels_true, labels_pred, conf_thresh, iou_thresh=0.6):
    evaluator = DetectionEvaluator(min_box_area=0.01 * 416 * 416, max_box_area=1.0 * 416 * 416,
                                   min_aspect_ratio=0,
                                   max_aspect_ratio=100.0, iou_thresh=iou_thresh)
    fn = []
    fp = []
    tp = []
    for i in range(len(labels_true)):

        label_pred = ImgLabel([obj for obj in labels_pred[i].objects if obj.confidence > conf_thresh])
        label_true = labels_true[i]
        evaluator.evaluate(label_true, label_pred)

        for b in evaluator.boxes_fn:
            w = b.poly.width
            h = b.poly.height
            fn.append((w, h))

        for b in evaluator.boxes_fp:
            w = b.poly.width
            h = b.poly.height
            fp.append((w, h))

        for i_b, b in enumerate(evaluator.boxes_true):
            if i_b not in evaluator.false_negatives_idx:
                w = b.poly.width
                h = b.poly.height
                tp.append((w, h))

    return tp, fp, fn


def evalset(labels_true, labels_pred, iou_thresh=0.6):
    evaluator = DetectionEvaluator(min_box_area=0.01 * 416 * 416, max_box_area=1.0 * 416 * 416,
                                   min_aspect_ratio=0.3,
                                   max_aspect_ratio=3.0, iou_thresh=iou_thresh)
    tp = []
    fp = []
    fn = []
    boxes_true = []
    results = []
    for i in range(len(labels_true)):
        evaluator.evaluate(labels_true[i], labels_pred[i])
        tp.extend(evaluator.boxes_tp)
        fp.extend(evaluator.boxes_fp)
        fn.extend(evaluator.boxes_fn)
        boxes_true.extend(evaluator.boxes_true)
        results.append(evaluator.result)
    sum_r = sum_results(results)
    return sum_r, tp, fp, fn, boxes_true


def evalcluster_size_ap(labels_true, labels_pred, bins, min_ar, max_ar,
                        min_obj_size, max_obj_size, iou_thresh=0.6, images=None, show_t=-1):
    evaluator = DetectionEvaluator(min_box_area=min_obj_size,
                                   max_box_area=max_obj_size,
                                   min_aspect_ratio=min_ar,
                                   max_aspect_ratio=max_ar, iou_thresh=iou_thresh)

    aps = []
    recalls = []
    precisions = []
    n_true = []
    for i_s in range(len(bins) - 1):
        results_bin = []
        n_true_bin = 0
        for i, l_true in enumerate(labels_true):
            l_true_bin = ImgLabel([o for o in l_true.objects if bins[i_s] < o.poly.area < bins[i_s + 1]])
            l_pred_bin = ImgLabel([o for o in labels_pred[i].objects if bins[i_s] < o.poly.area < bins[i_s + 1]])
            result = evaluator.evaluate(l_true_bin, l_pred_bin)
            if show_t > -1:
                evaluator.show(images[i], show_t)
            results_bin.append(result)
            n_true_bin += len(l_true_bin.objects)
        r = sum_results(results_bin)
        mean_pr, mean_rec, std_pr, std_rec = average_precision_recall([r])
        ap = np.mean(mean_pr)
        aps.append(ap)
        recalls.append(r.recall_conf)
        precisions.append(r.precision_conf)
        n_true.append(n_true_bin)

    return aps, n_true, recalls, precisions


def evalcluster_location_ap(labels_true, labels_pred, bins, iou_thresh=0.6, images=None, show_t=-1):
    evaluator = DetectionEvaluator(min_box_area=0.001 * 416 * 416,
                                   max_box_area=2.0 * 416 * 416,
                                   min_aspect_ratio=0,
                                   max_aspect_ratio=100.0, iou_thresh=iou_thresh)

    aps = []
    n_true = []
    for i_s in range(len(bins) - 1):
        results_bin = []
        n_true_bin = 0
        for i, l_true in enumerate(labels_true):
            l_true_bin = ImgLabel([o for o in l_true.objects if bins[i_s] <= o.poly.cx < bins[i_s + 1]])
            l_pred_bin = ImgLabel([o for o in labels_pred[i].objects if bins[i_s] <= o.poly.cx < bins[i_s + 1]])
            result = evaluator.evaluate(l_true_bin, l_pred_bin)
            if show_t > -1:
                evaluator.show(images[i], show_t)
            results_bin.append(result)
            n_true_bin += len(l_true_bin.objects)
        r = sum_results(results_bin)
        mean_pr, mean_rec, std_pr, std_rec = average_precision_recall([r])
        ap = np.mean(mean_pr)
        aps.append(ap)
        n_true.append(n_true_bin)

    return aps, n_true


def evaluate_labels(
        name,
        model_src,
        label_file,
        preprocessing: [ImgTransform] = None,
        min_box_area=0,
        max_box_area=2.0,
        iou_thresh=0.4,
        show=-1,
        result_path=None,
        result_file=None,
        min_aspect_ratio=0.0,
        max_aspect_ratio=100.0):
    exp_param_file = name + '_summary'

    exp_params = {'name': name,
                  'model': model_src,
                  'iou_thresh': iou_thresh,
                  'label_src': label_file,
                  'min_box_area': min_box_area,
                  'max_box_area': max_box_area}

    summary = ModelSummary.from_file(model_src + '/summary.pkl')

    metric = DetectionEvaluator(
        iou_thresh=iou_thresh,
        min_box_area=min_box_area * summary.img_size,
        max_box_area=max_box_area * summary.img_size,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )

    labels_true, labels_pred, image_files = load_predictions(label_file)
    results = []
    for i in range(len(labels_true)):
        img = imread(image_files[i], 'bgr')
        label = labels_true[i]
        if preprocessing:
            for p in preprocessing:
                img, label = p.transform(img, label)
        result = metric.evaluate(label, labels_pred[i])
        if show > -1:
            metric.show(img, t=show)
        results.append(result)

    output = {
        'results': results,
        'labels_true': labels_true,
        'labels_pred': labels_pred,
        'image_files': image_files
    }

    if result_path:
        create_dirs([result_path])
        if result_file is None:
            result_file = name
        save_file(output, result_file, result_path)
        save_file(exp_params, exp_param_file + '.txt', result_path)
        save_file(exp_params, exp_param_file + '.pkl', result_path)

    return output


def load_predictions(path: str):
    content = load_file(path)
    labels_true = content['labels_true']
    labels_pred = content['labels_pred']
    image_files = content['image_files']
    return labels_true, labels_pred, image_files


def preprocess_truth(img_files, labels_true, preprocessing: [ImgTransform]):
    labels_true_pp = []
    images_pp = []
    for i_l, l in enumerate(labels_true):
        img = imread(img_files[i_l], 'bgr')
        for p in preprocessing:
            img, l = p.transform(img, l)
        labels_true_pp.append(l)
        images_pp.append(img)

    return images_pp, labels_true_pp


def infer_on_set(
        model_src,
        image_source,
        batch_size,
        result_path,
        result_file,
        img_res=None,
        n_samples=None,
        color_format_dataset='bgr',
        preprocessing=None,
        image_format="jpg",
        show_t=-1):
    # Model
    conf_thresh = 0

    detector = load_detector(model_src, img_res, preprocessing)

    generator = GateGenerator(directories=image_source, batch_size=batch_size, img_format=image_format,
                              n_samples=n_samples,
                              shuffle=False, color_format=color_format_dataset, label_format='xml', start_idx=0)

    create_dirs([result_path])

    exp_params = {'model': model_src,
                  'conf_thresh': conf_thresh,
                  'image_source': image_source,
                  'color_format': color_format_dataset,
                  'n_samples': generator.n_samples,
                  'result_file': result_file,
                  'preprocessing': preprocessing}

    save_file(exp_params, 'test_summary' + '.pkl', result_path)
    save_file(exp_params, 'test_summary' + '.txt', result_path)
    n_batches = int(generator.n_samples / generator.batch_size)
    it = iter(generator.generate())
    labels_true = []
    labels_pred = []
    image_files = []
    for i in range(n_batches):
        batch = next(it)
        images = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        image_files_batch = [b[2] for b in batch]

        tic()
        predictions = detector.detect(images)
        if images[0].shape[0] != detector.model.input_shape[1] or \
                images[0].shape[1] != detector.model.input_shape[2]:
            print("Evaluator:: Labels have different size")

        if show_t > -1:
            for j, p in enumerate(predictions):
                l = p.copy()
                img_show = images[j].copy()
                l.objects = [o for o in l.objects if o.confidence > 0.01]
                if preprocessing:
                    for p in preprocessing:
                        img_show, _ = p.transform(img_show, ImgLabel([]))
                show(img_show, labels=l, t=show_t)
        # labels = [resize_label(l, images[0].shape[:2], self.model.input_shape) for l in labels]
        # for j in range(len(batch)):
        #     show(batch[j][0], labels=[predictions[j], labels[j]])

        labels_true.extend(labels)
        labels_pred.extend(predictions)
        image_files.extend(image_files_batch)
        toc("Evaluated batch {0:d}/{1:d} in ".format(i, n_batches))

        content = {'labels_true': labels_true,
                   'labels_pred': labels_pred,
                   'image_files': image_files}
        save_file(content, result_file, result_path)


def infer_detector_on_set(
        detector,
        image_source,
        batch_size,
        result_path,
        result_file,
        img_res=None,
        n_samples=None,
        color_format_dataset='bgr',
        preprocessing=None,
        image_format="jpg",
        show_t=-1):
    # Model
    conf_thresh = 0


    generator = GateGenerator(directories=image_source, batch_size=batch_size, img_format=image_format,
                              n_samples=n_samples,
                              shuffle=False, color_format=color_format_dataset, label_format='xml', start_idx=0)

    create_dirs([result_path])

    exp_params = {'conf_thresh': conf_thresh,
                  'image_source': image_source,
                  'color_format': color_format_dataset,
                  'n_samples': generator.n_samples,
                  'result_file': result_file,
                  'preprocessing': preprocessing}

    save_file(exp_params, 'test_summary' + '.pkl', result_path)
    save_file(exp_params, 'test_summary' + '.txt', result_path)
    n_batches = int(generator.n_samples / generator.batch_size)
    it = iter(generator.generate())
    labels_true = []
    labels_pred = []
    image_files = []
    for i in range(n_batches):
        batch = next(it)
        images = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        image_files_batch = [b[2] for b in batch]

        tic()

        predictions = detector.detect(images)
        if images[0].shape[0] != detector.model.input_shape[1] or \
                images[0].shape[1] != detector.model.input_shape[2]:
            print("Evaluator:: Labels have different size")

        if show_t > -1:
            for j, p in enumerate(predictions):
                l = p.copy()
                img_show = images[j].copy()
                l.objects = [o for o in l.objects if o.confidence > 0.01]
                if preprocessing:
                    for p in preprocessing:
                        img_show, _ = p.transform(img_show, ImgLabel([]))
                show(img_show, labels=l, t=show_t)
        # labels = [resize_label(l, images[0].shape[:2], self.model.input_shape) for l in labels]
        # for j in range(len(batch)):
        #     show(batch[j][0], labels=[predictions[j], labels[j]])

        labels_true.extend(labels)
        labels_pred.extend(predictions)
        image_files.extend(image_files_batch)
        toc("Evaluated batch {0:d}/{1:d} in ".format(i, n_batches))

        content = {'labels_true': labels_true,
                   'labels_pred': labels_pred,
                   'image_files': image_files}
        save_file(content, result_file, result_path)


def sum_results(detection_results: [DetectionResult]):
    """
    Sums list
    :param detection_results: list of results per image
    :return: sum
    """
    result_sum = detection_results[0]
    for d in detection_results[1:]:
        result_sum += d

    return result_sum


def average_precision_recall(detection_results: [DetectionResult], recall_levels=None):
    """
    Calculates average precision recall with interpolation. According to mAP of Pascal VOC metric.
    :param detection_results: list of results for each image
    :return: tensor(1,11): mean precision, tensor(1,11) mean recall
    """
    if recall_levels is None:
        recall_levels = np.linspace(0, 1.0, 11)
    precision = np.zeros((len(detection_results), len(recall_levels)))
    recall = np.zeros((len(detection_results), len(recall_levels)))
    for i, result in enumerate(detection_results):
        precision_raw = result.precision_conf
        recall_raw = result.recall_conf

        precision[i], recall[i] = interpolate(precision_raw, recall_raw, recall_levels)

    mean_pr = np.mean(precision, 0)
    std_pr = np.std(precision, 0)

    mean_rec = np.mean(recall, 0)
    std_rec = np.std(recall, 0)
    return mean_pr, mean_rec, std_pr, std_rec


def interpolate(precision_raw, recall_raw, recall_levels=None):
    if recall_levels is None:
        recall_levels = np.linspace(0, 1.0, 11)

    precision = np.zeros(shape=(len(recall_levels)))
    for i, r in enumerate(recall_levels):
        idx = recall_raw >= r
        if np.any(idx == True):
            precision[i] = np.max(precision_raw[idx])
        else:
            precision[i] = 0
    return precision, recall_levels.T


def load_result(netname, img_res, grid, layers, filters, old=True, filename='result_0.4.pkl'):
    if old:
        if filters == 16:
            folder_name = '{}{}x{}-{}layers'.format(netname, img_res[0], img_res[1], layers)
        else:
            folder_name = '{}{}x{}-{}layers-{}filters'.format(netname, img_res[0], img_res[1], layers, filters)
    else:
        folder_name = '{}{}x{}-{}x{}+{}layers+{}filters'.format(netname, img_res[0], img_res[1], grid[0],
                                                                grid[1], layers, filters)

    return load_file('out/2606/' + folder_name + '/results/' + filename)
