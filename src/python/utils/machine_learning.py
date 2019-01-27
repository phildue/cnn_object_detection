from sklearn.cluster import KMeans


def kmeans_anchors(n_boxes: [int], label_source, img_shape):
    if isinstance(label_source, list):
        labels = []
        for p in label_source:
            label_reader = DatasetParser.get_parser(p, 'xml', color_format='bgr')
            labels.extend(label_reader.read()[1])
    else:
        label_reader = DatasetParser.get_parser(label_source, 'xml', color_format='bgr')
        _, labels = label_reader.read()

    wh = []
    for label in labels:
        h, w = img_shape
        for b in label.objects:
            if 0 < b.poly.width < w \
                    and 0 < b.poly.height < h:
                box_dim = np.array([b.poly.width, b.poly.height])
                box_dim = np.expand_dims(box_dim, 0)
                wh.append(box_dim)
    box_dims = np.concatenate(wh, 0)

    kmeans = KMeans(n_clusters=np.sum(n_boxes)).fit(box_dims)
    centers = kmeans.cluster_centers_
    centers = np.round(centers, 2)
    centers = np.sort(centers,0)[::-1]
    anchors = []
    idx = 0
    for n in n_boxes:
        anchors.append(centers[idx:idx + n])
        idx += n

    return np.array(anchors)

