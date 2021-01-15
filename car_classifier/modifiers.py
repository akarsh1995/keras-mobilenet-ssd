import numpy as np
import tensorflow as tf

def one_hot_class_label(classname, label_maps):
    """ Turn classname to one hot encoded label.

    Args:
        - classname: String representing the classname
        - label_maps: A list of strings containing all the classes

    Returns:
        - A numpy array of shape (len(label_maps), )

    Raises:
        - Classname does not includes in label maps
    """
    assert classname in label_maps, "classname must be included in label maps"
    temp = np.zeros((len(label_maps)), dtype=np.int)
    temp[label_maps.index(classname)] = 1
    return temp


def corner_to_center(boxes):
    """ Convert bounding boxes from center format (xmin, ymin, xmax, ymax) to corner format (cx, cy, width, height)

    Args:
        - boxes: numpy array of tensor containing all the boxes to be converted

    Returns:
        - A numpy array or tensor of converted boxes
    """
    temp = boxes.copy()
    width = np.abs(boxes[..., 0] - boxes[..., 2])
    height = np.abs(boxes[..., 1] - boxes[..., 3])
    temp[..., 0] = boxes[..., 0] + (width / 2)  # cx
    temp[..., 1] = boxes[..., 1] + (height / 2)  # cy
    temp[..., 2] = width  # xmax
    temp[..., 3] = height  # ymax
    return temp


def iou(box_group1, box_group2):
    """ Calculates the intersection over union (aka. Jaccard Index) between two boxes.
    Boxes are assumed to be in corners format (xmin, ymin, xmax, ymax)

    Args:
    - box_group1: boxes in group 1
    - box_group2: boxes in group 2

    Returns:
    - A numpy array of shape (len(box_group1), len(box_group2)) where each value represents the iou between a box in box_group1 to a box in box_group2

    Raises:
    - The shape of box_group1 and box_group2 are not the same.

    Code References:
    - https://stackoverflow.com/questions/28723670/intersection-over-union-between-two-detections/41660682
    """
    assert box_group1.shape == box_group2.shape, "The two boxes array must be the same shape."
    xmin_intersect = np.maximum(box_group1[..., 0], box_group2[..., 0])
    ymin_intersect = np.maximum(box_group1[..., 1], box_group2[..., 1])
    xmax_intersect = np.minimum(box_group1[..., 2], box_group2[..., 2])
    ymax_intersect = np.minimum(box_group1[..., 3], box_group2[..., 3])

    intersect = (xmax_intersect - xmin_intersect) * (ymax_intersect - ymin_intersect)
    box_group1_area = (box_group1[..., 2] - box_group1[..., 0]) * (box_group1[..., 3] - box_group1[..., 1])
    box_group2_area = (box_group2[..., 2] - box_group2[..., 0]) * (box_group2[..., 3] - box_group2[..., 1])
    union = box_group1_area + box_group2_area - intersect
    res = intersect / union

    # set invalid ious to zeros
    res[xmax_intersect < xmin_intersect] = 0
    res[ymax_intersect < ymin_intersect] = 0
    res[res < 0] = 0
    res[res > 1] = 0
    return res


def center_to_corner(boxes):
    """ Convert bounding boxes from center format (cx, cy, width, height) to corner format (xmin, ymin, xmax, ymax)

    Args:
        - boxes: numpy array of tensor containing all the boxes to be converted

    Returns:
        - A numpy array or tensor of converted boxes
    """
    temp = boxes.copy()
    temp[..., 0] = boxes[..., 0] - (boxes[..., 2] / 2)  # xmin
    temp[..., 1] = boxes[..., 1] - (boxes[..., 3] / 2)  # ymin
    temp[..., 2] = boxes[..., 0] + (boxes[..., 2] / 2)  # xmax
    temp[..., 3] = boxes[..., 1] + (boxes[..., 3] / 2)  # ymax
    return temp


def match_gt_boxes_to_default_boxes(
    gt_boxes,
    default_boxes,
    match_threshold=0.5,
    neutral_threshold=0.3
):
    """ Matches ground truth bounding boxes to default boxes based on the SSD paper.

    'We begin by matching each ground truth box to the default box with the best jaccard overlap (as in MultiBox [7]).
    Unlike MultiBox, we then match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)'

    Args:
        - gt_boxes: A numpy array or tensor of shape (num_gt_boxes, 4). Structure [cx, cy, w, h]
        - default_boxes: A numpy array of tensor of shape (num_default_boxes, 4). Structure [cx, cy, w, h]
        - threshold: A float representing a target to decide whether the box is matched
        - default_boxes: A numpy array of tensor of shape (num_default_boxes, 4). Structure [cx, cy, w, h]

    Returns:
        - matches: A numpy array of shape (num_matches, 2). The first index in the last dimension is the index
          of the ground truth box and the last index is the default box index.
        - neutral_boxes: A numpy array of shape (num_neutral_boxes, 2). The first index in the last dimension is the index
          of the ground truth box and the last index is the default box index.

    Raises:
        - Either the shape of ground truth's boxes array or the default boxes array is not 2

    Code References:
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/ssd_encoder_decoder/matching_utils.py

    Paper References:
        - Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.Y., & Berg, A. C. (2016).
          SSD: Single Shot MultiBox Detector. https://arxiv.org/abs/1512.02325
    """

    assert len(gt_boxes.shape) == 2, "Shape of ground truth boxes array must be 2"
    assert len(default_boxes.shape) == 2, "Shape of default boxes array must be 2"

    # convert gt_boxes and default_boxes to [xmin, ymin, xmax, ymax]
    gt_boxes = center_to_corner(gt_boxes)
    default_boxes = center_to_corner(default_boxes)

    num_gt_boxes = gt_boxes.shape[0]
    num_default_boxes = default_boxes.shape[0]

    matches = np.zeros((num_gt_boxes, 2), dtype=np.int)

    # match ground truth to default box with highest iou
    for i in range(num_gt_boxes):
        gt_box = gt_boxes[i]
        gt_box = np.tile(
            np.expand_dims(gt_box, axis=0),
            (num_default_boxes, 1)
        )
        ious = iou(gt_box, default_boxes)
        matches[i] = [i, np.argmax(ious)]

    # match default boxes to ground truths with overlap higher than threshold
    gt_boxes = np.tile(np.expand_dims(gt_boxes, axis=1), (1, num_default_boxes, 1))
    default_boxes = np.tile(np.expand_dims(default_boxes, axis=0), (num_gt_boxes, 1, 1))
    ious = iou(gt_boxes, default_boxes)
    ious[:, matches[:, 1]] = 0

    matched_gt_boxes_idxs = np.argmax(ious, axis=0)  # for each default boxes, select the ground truth box that has the highest iou
    matched_ious = ious[matched_gt_boxes_idxs, list(range(num_default_boxes))]  # get iou scores between gt and default box that were selected above
    matched_df_boxes_idxs = np.nonzero(matched_ious >= match_threshold)[0]  # select only matched default boxes that has iou larger than threshold
    matched_gt_boxes_idxs = matched_gt_boxes_idxs[matched_df_boxes_idxs]

    # concat the results of the two matching process together
    matches = np.concatenate([
        matches,
        np.concatenate([
            np.expand_dims(matched_gt_boxes_idxs, axis=-1),
            np.expand_dims(matched_df_boxes_idxs, axis=-1)
        ], axis=-1),
    ], axis=0)

    ious[:, matches[:, 1]] = 0

    # find neutral boxes (ious that are higher than neutral_threshold but below threshold)
    # these boxes are neither background nor has enough ious score to qualify as a match.
    background_gt_boxes_idxs = np.argmax(ious, axis=0)
    background_gt_boxes_ious = ious[background_gt_boxes_idxs, list(range(num_default_boxes))]
    neutral_df_boxes_idxs = np.nonzero(background_gt_boxes_ious >= neutral_threshold)[0]
    neutral_gt_boxes_idxs = background_gt_boxes_idxs[neutral_df_boxes_idxs]
    neutral_boxes = np.concatenate([
        np.expand_dims(neutral_gt_boxes_idxs, axis=-1),
        np.expand_dims(neutral_df_boxes_idxs, axis=-1)
    ], axis=-1)

    return matches, neutral_boxes


def encode_bboxes(y, epsilon=10e-5):
    """ Encode the label to a proper format suitable for training SSD network.

    Args:
        - y: A numpy of shape (num_default_boxes, num_classes + 12) representing a label sample.

    Returns:
        - A numpy array with the same shape as y but its gt boxes values has been encoded to the proper SSD format.

    Paper References:
        - Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.Y., & Berg, A. C. (2016).
          SSD: Single Shot MultiBox Detector. https://arxiv.org/abs/1512.02325

    Webpage References:
        - https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/

    Code References:
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/ssd_encoder_decoder/ssd_input_encoder.py
    """
    gt_boxes = y[:, -12:-8]
    df_boxes = y[:, -8:-4]
    variances = y[:, -4:]
    encoded_gt_boxes_cx = ((gt_boxes[:, 0] - df_boxes[:, 0]) / (df_boxes[:, 2])) / np.sqrt(variances[:, 0])
    encoded_gt_boxes_cy = ((gt_boxes[:, 1] - df_boxes[:, 1]) / (df_boxes[:, 3])) / np.sqrt(variances[:, 1])
    encoded_gt_boxes_w = np.log(epsilon + gt_boxes[:, 2] / df_boxes[:, 2]) / np.sqrt(variances[:, 2])
    encoded_gt_boxes_h = np.log(epsilon + gt_boxes[:, 3] / df_boxes[:, 3]) / np.sqrt(variances[:, 3])
    y[:, -12] = encoded_gt_boxes_cx
    y[:, -11] = encoded_gt_boxes_cy
    y[:, -10] = encoded_gt_boxes_w
    y[:, -9] = encoded_gt_boxes_h
    return y


def get_number_default_boxes(aspect_ratios, extra_box_for_ar_1=True):
    """ Get the number of default boxes for each grid cell based on the number of aspect ratios
    and whether to add a extra box for aspect ratio 1

    Args:
    - aspect_ratios: A list containing the different aspect ratios of default boxes.
    - extra_box_for_ar_1: Whether to add a extra box for aspect ratio 1.

    Returns:
    - An integer for the number of default boxes.
    """
    num_aspect_ratios = len(aspect_ratios)
    return num_aspect_ratios + 1 if (1.0 in aspect_ratios) and extra_box_for_ar_1 else num_aspect_ratios


def generate_default_boxes_for_feature_map(
    feature_map_size,
    image_size,
    offset,
    scale,
    next_scale,
    aspect_ratios,
    variances,
    extra_box_for_ar_1
):
    """ Generates a 4D Tensor representing default boxes.

    Note:
    - The structure of a default box is [xmin, ymin, xmax, ymax]

    Args:
    - feature_map_size: The size of the feature map. (must be square)
    - image_size: The size of the input image. (must be square)
    - offset: The offset for the center of the default boxes. The order is (offset_x, offset_y)
    - scale: The current scale of the default boxes.
    - next_scale: The next scale of the default boxes.
    - aspect_ratios: A list of aspect ratios representing the default boxes.
    - variance: ...
    - extra_box_for_ar_1: Whether to add an extra box for default box with aspect ratio 1.

    Returns:
    - A 4D numpy array of shape (feature_map_size, feature_map_size, num_default_boxes, 8)

    Raises:
    - offset does not have a len of 2

    Code References:
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_layers/keras_layer_AnchorBoxes.py
    """
    assert len(offset) == 2, "offset must be of len 2"
    grid_size = image_size / feature_map_size
    offset_x, offset_y = offset
    num_default_boxes = get_number_default_boxes(
        aspect_ratios,
        extra_box_for_ar_1=extra_box_for_ar_1
    )
    # get all width and height of default boxes
    wh_list = []
    for ar in aspect_ratios:
        if ar == 1.0 and extra_box_for_ar_1:
            wh_list.append([
                image_size * np.sqrt(scale * next_scale) * np.sqrt(ar),
                image_size * np.sqrt(scale * next_scale) * (1 / np.sqrt(ar)),
            ])
        wh_list.append([
            image_size * scale * np.sqrt(ar),
            image_size * scale * (1 / np.sqrt(ar)),
        ])
    wh_list = np.array(wh_list, dtype=np.float)
    # get all center points of each grid cells
    cx = np.linspace(offset_x * grid_size, image_size - (offset_x * grid_size), feature_map_size)
    cy = np.linspace(offset_y * grid_size, image_size - (offset_y * grid_size), feature_map_size)
    cx_grid, cy_grid = np.meshgrid(cx, cy)
    cx_grid, cy_grid = np.expand_dims(cx_grid, axis=-1), np.expand_dims(cy_grid, axis=-1)
    cx_grid, cy_grid = np.tile(cx_grid, (1, 1, num_default_boxes)), np.tile(cy_grid, (1, 1, num_default_boxes))
    #
    default_boxes = np.zeros((feature_map_size, feature_map_size, num_default_boxes, 4))
    default_boxes[:, :, :, 0] = cx_grid
    default_boxes[:, :, :, 1] = cy_grid
    default_boxes[:, :, :, 2] = wh_list[:, 0]
    default_boxes[:, :, :, 3] = wh_list[:, 1]
    # clip overflow default boxes
    # default_boxes = center_to_corner(default_boxes)
    # x_coords = default_boxes[:, :, :, [0, 2]]
    # x_coords[x_coords >= image_size] = image_size - 1
    # x_coords[x_coords < 0] = 0
    # default_boxes[:, :, :, [0, 2]] = x_coords
    # y_coords = default_boxes[:, :, :, [1, 3]]
    # y_coords[y_coords >= image_size] = image_size - 1
    # y_coords[y_coords < 0] = 0
    # default_boxes[:, :, :, [1, 3]] = y_coords
    # default_boxes = corner_to_center(default_boxes)
    #
    default_boxes[:, :, :, [0, 2]] /= image_size
    default_boxes[:, :, :, [1, 3]] /= image_size
    #
    variances_tensor = np.zeros_like(default_boxes)
    variances_tensor += variances
    default_boxes = np.concatenate([default_boxes, variances_tensor], axis=-1)
    return default_boxes


def decode_predictions(
    y_pred,
    input_size,
    nms_max_output_size=400,
    confidence_threshold=0.01,
    iou_threshold=0.45,
    num_predictions=10
):
    """"""
    # decode bounding boxes predictions
    cx = y_pred[..., -12] * y_pred[..., -4] * y_pred[..., -6] + y_pred[..., -8]
    cy = y_pred[..., -11] * y_pred[..., -3] * y_pred[..., -5] + y_pred[..., -7]
    w = tf.exp(y_pred[..., -10] * tf.sqrt(y_pred[..., -2])) * y_pred[..., -6]
    h = tf.exp(y_pred[..., -9] * tf.sqrt(y_pred[..., -1])) * y_pred[..., -5]
    # convert bboxes to corners format (xmin, ymin, xmax, ymax) and scale to fit input size
    xmin = (cx - 0.5 * w) * input_size
    ymin = (cy - 0.5 * h) * input_size
    xmax = (cx + 0.5 * w) * input_size
    ymax = (cy + 0.5 * h) * input_size
    # concat class predictions and bbox predictions together
    y_pred = tf.concat([
        y_pred[..., :-12],
        tf.expand_dims(xmin, axis=-1),
        tf.expand_dims(ymin, axis=-1),
        tf.expand_dims(xmax, axis=-1),
        tf.expand_dims(ymax, axis=-1)], -1)
    #
    batch_size = tf.shape(y_pred)[0]  # Output dtype: tf.int32
    num_boxes = tf.shape(y_pred)[1]
    num_classes = y_pred.shape[2] - 4
    class_indices = tf.range(1, num_classes)
    # Create a function that filters the predictions for the given batch item. Specifically, it performs:
    # - confidence thresholding
    # - non-maximum suppression (NMS)
    # - top-k filtering

    def filter_predictions(batch_item):
        # Create a function that filters the predictions for one single class.
        def filter_single_class(index):

            # From a tensor of shape (n_boxes, n_classes + 4 coordinates) extract
            # a tensor of shape (n_boxes, 1 + 4 coordinates) that contains the
            # confidnece values for just one class, determined by `index`.
            confidences = tf.expand_dims(batch_item[..., index], axis=-1)
            class_id = tf.fill(dims=tf.shape(confidences), value=float(index))
            box_coordinates = batch_item[..., -4:]

            single_class = tf.concat([class_id, confidences, box_coordinates], -1)

            # Apply confidence thresholding with respect to the class defined by `index`.
            threshold_met = single_class[:, 1] > confidence_threshold
            single_class = tf.boolean_mask(tensor=single_class,
                                           mask=threshold_met)

            # If any boxes made the threshold, perform NMS.
            def perform_nms():
                scores = single_class[..., 1]

                # `tf.image.non_max_suppression()` needs the box coordinates in the format `(ymin, xmin, ymax, xmax)`.
                xmin = tf.expand_dims(single_class[..., -4], axis=-1)
                ymin = tf.expand_dims(single_class[..., -3], axis=-1)
                xmax = tf.expand_dims(single_class[..., -2], axis=-1)
                ymax = tf.expand_dims(single_class[..., -1], axis=-1)
                boxes = tf.concat([ymin, xmin, ymax, xmax], -1)
                maxima_indices = tf.image.non_max_suppression(boxes=boxes,
                                                              scores=scores,
                                                              max_output_size=nms_max_output_size,
                                                              iou_threshold=iou_threshold,
                                                              name='non_maximum_suppresion')
                maxima = tf.gather(params=single_class,
                                   indices=maxima_indices,
                                   axis=0)
                return maxima

            def no_confident_predictions():
                return tf.constant(value=0.0, shape=(1, 6))

            single_class_nms = tf.cond(tf.equal(tf.size(single_class), 0), no_confident_predictions, perform_nms)

            # Make sure `single_class` is exactly `self.nms_max_output_size` elements long.
            padded_single_class = tf.pad(tensor=single_class_nms,
                                         paddings=[[0, nms_max_output_size - tf.shape(single_class_nms)[0]], [0, 0]],
                                         mode='CONSTANT',
                                         constant_values=0.0)

            return padded_single_class

        # Iterate `filter_single_class()` over all class indices.
        filtered_single_classes = tf.map_fn(fn=lambda i: filter_single_class(i),
                                            elems=tf.range(1, num_classes),
                                            dtype=tf.float32,
                                            parallel_iterations=128,
                                            back_prop=False,
                                            swap_memory=False,
                                            infer_shape=True,
                                            name='loop_over_classes')

        # Concatenate the filtered results for all individual classes to one tensor.
        filtered_predictions = tf.reshape(tensor=filtered_single_classes, shape=(-1, 6))

        # Perform top-k filtering for this batch item or pad it in case there are
        # fewer than `self.top_k` boxes left at this point. Either way, produce a
        # tensor of length `self.top_k`. By the time we return the final results tensor
        # for the whole batch, all batch items must have the same number of predicted
        # boxes so that the tensor dimensions are homogenous. If fewer than `self.top_k`
        # predictions are left after the filtering process above, we pad the missing
        # predictions with zeros as dummy entries.
        def top_k():
            return tf.gather(params=filtered_predictions,
                             indices=tf.nn.top_k(filtered_predictions[:, 1], k=num_predictions, sorted=True).indices,
                             axis=0)

        def pad_and_top_k():
            padded_predictions = tf.pad(tensor=filtered_predictions,
                                        paddings=[[0, num_predictions - tf.shape(filtered_predictions)[0]], [0, 0]],
                                        mode='CONSTANT',
                                        constant_values=0.0)
            return tf.gather(params=padded_predictions,
                             indices=tf.nn.top_k(padded_predictions[:, 1], k=num_predictions, sorted=True).indices,
                             axis=0)

        top_k_boxes = tf.cond(tf.greater_equal(tf.shape(filtered_predictions)[0], num_predictions), top_k, pad_and_top_k)

        return top_k_boxes
    # Iterate `filter_predictions()` over all batch items.
    output_tensor = tf.map_fn(fn=lambda x: filter_predictions(x),
                              elems=y_pred,
                              dtype=None,
                              parallel_iterations=128,
                              back_prop=False,
                              swap_memory=False,
                              infer_shape=True,
                              name='loop_over_batch')
    return output_tensor
