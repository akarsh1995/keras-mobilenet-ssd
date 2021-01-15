from pathlib import Path
from random import choices
from typing import Any, List

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from dataclasses import dataclass
from matplotlib import pyplot as plt
import tensorflow as tf
from car_classifier.modifiers import one_hot_class_label, match_gt_boxes_to_default_boxes, encode_bboxes, \
    generate_default_boxes_for_feature_map


@dataclass
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def coordinates(self):
        return self.xmin, self.ymin, self.xmax, self.ymax


@dataclass
class Annotation:
    image: Any
    bbox: BBox
    img_class: str
    class_name: str

    def __post_init__(self):
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')

    def get_resized_annotation(self, resized_shape=(256, 256)):
        resized_height, resized_width = resized_shape
        original_width, original_height = self.image.size
        scale_height = original_height / resized_height
        scale_width = original_width / resized_width
        img = self.image.resize((resized_height, resized_width))
        return Annotation(img,
                          BBox(int(self.bbox.xmin / scale_width),
                               int(self.bbox.ymin / scale_height),
                               int(self.bbox.xmax / scale_width),
                               int(self.bbox.ymax / scale_height)),
                          self.img_class,
                          self.class_name)

    def image_label_tensors(self):
        return np.array(self.image), (self.bbox.coordinates, self.img_class)


class DataAnnotationReader:

    def __init__(self, file_path, test_train_root, test=True):
        if test:
            self._img_dir = test_train_root.joinpath('test')
        else:
            self._img_dir = test_train_root.joinpath('train')
        self._df = pd.read_csv(file_path, header=None, index_col=0)
        self._all_imgs = {
            path.name: path for path in self._img_dir.rglob("*.jpg")
        }
        self.all_keys = self._df.index
        self.ALL_LABELS = ['__background__'] + file_path.parent.joinpath('names.csv').read_text().splitlines()

    @property
    def n_samples(self) -> int:
        return len(self._df)

    @property
    def n_classes(self):
        return len(self.ALL_LABELS)

    def __getitem__(self, key):
        image = self.get_image(key)
        *bbox_, class_label = self._df.loc[key]
        return Annotation(image,
                          BBox(*bbox_),
                          class_label,
                          self.ALL_LABELS[class_label])

    def __iter__(self):
        for img_name in self._all_imgs.keys():
            yield self[img_name]

    def __len__(self):
        return len(self._all_imgs)

    def get_image(self, img_name):
        with Image.open(self._all_imgs[img_name]) as img:
            img.load()
        return img

    def sample_annos(self, n=10):
        names = choices(list(self._all_imgs.keys()), k=n)
        return [self[name] for name in names]


class Dataloader(tf.keras.utils.Sequence):

    def __init__(self, anno_reader: DataAnnotationReader,
                 batch_size: int, config, preprocess_func, augment=True, workers=4):
        self.anno_reader = anno_reader
        self.batch_size = batch_size
        self.shuffle = True
        self.indices = range(0, anno_reader.n_samples)
        self._keys = anno_reader.all_keys
        self._workers = workers
        self.perform_augmentation = augment
        self.num_classes = anno_reader.n_classes
        self.label_maps = anno_reader.ALL_LABELS
        training_config = config['training']
        model_config = config['model']
        self.input_size = model_config["input_size"]
        self.match_threshold = training_config["match_threshold"]
        self.neutral_threshold = training_config["neutral_threshold"]
        self.extra_box_for_ar_1 = model_config["extra_box_for_ar_1"]
        self.default_boxes_config = model_config["default_boxes"]
        self.input_template = self.__get_input_template()
        self.on_epoch_end()
        self._preprocess_callback = preprocess_func

    def __getitem__(self, index):
        batch = [
            self.anno_reader[key]
            for key in self._keys[index:index + self.batch_size]
        ]
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __len__(self):
        return len(self.anno_reader) // self.batch_size

    def __get_input_template(self):
        scales = np.linspace(
            self.default_boxes_config["min_scale"],
            self.default_boxes_config["max_scale"],
            len(self.default_boxes_config["layers"])
        )
        mbox_conf_layers = []
        mbox_loc_layers = []
        mbox_default_boxes_layers = []
        for i, layer in enumerate(self.default_boxes_config["layers"]):
            layer_default_boxes = generate_default_boxes_for_feature_map(
                feature_map_size=layer["size"],
                image_size=self.input_size,
                offset=layer["offset"],
                scale=scales[i],
                next_scale=scales[i+1] if i+1 <= len(self.default_boxes_config["layers"]) - 1 else 1,
                aspect_ratios=layer["aspect_ratios"],
                variances=self.default_boxes_config["variances"],
                extra_box_for_ar_1=self.extra_box_for_ar_1
            )
            layer_default_boxes = np.reshape(layer_default_boxes, (-1, 8))
            layer_conf = np.zeros((layer_default_boxes.shape[0], self.num_classes))
            layer_conf[:, 0] = 1  # all classes are background by default
            mbox_conf_layers.append(layer_conf)
            mbox_loc_layers.append(np.zeros((layer_default_boxes.shape[0], 4)))
            mbox_default_boxes_layers.append(layer_default_boxes)
        mbox_conf = np.concatenate(mbox_conf_layers, axis=0)
        mbox_loc = np.concatenate(mbox_loc_layers, axis=0)
        mbox_default_boxes = np.concatenate(mbox_default_boxes_layers, axis=0)
        template = np.concatenate([mbox_conf, mbox_loc, mbox_default_boxes], axis=-1)
        template = np.expand_dims(template, axis=0)
        return np.tile(template, (self.batch_size, 1, 1))

    def __get_data(self, batch: List[Annotation]):
        X = []
        y = self.input_template.copy()

        batch = [anno.get_resized_annotation((self.input_size, self.input_size)) for anno in batch]
        for batch_idx, anno in enumerate(batch):
            image, bboxes, classes = (
                np.array(anno.image),
                np.array([anno.bbox.coordinates]),
                [anno.class_name]
            )

            # if self.perform_augmentation:
            #     image, bboxes, classes = self.__augment(
            #         image=image,
            #         bboxes=bboxes,
            #         classes=classes
            #     )

            input_img = self.process_input_fn(image)

            gt_classes = np.zeros((bboxes.shape[0], self.num_classes))
            gt_boxes = np.zeros((bboxes.shape[0], 4))
            default_boxes = y[batch_idx, :, -8:]

            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                cx = ((bbox[0] + bbox[2]) / 2) / self.input_size
                cy = ((bbox[1] + bbox[3]) / 2) / self.input_size
                width = (abs(bbox[2] - bbox[0])) / self.input_size
                height = (abs(bbox[3] - bbox[1])) / self.input_size
                gt_boxes[i] = [cx, cy, width, height]
                gt_classes[i] = one_hot_class_label(classes[i], self.label_maps)

            matches, neutral_boxes = match_gt_boxes_to_default_boxes(
                gt_boxes=gt_boxes,
                default_boxes=default_boxes[:, :4],
                match_threshold=self.match_threshold,
                neutral_threshold=self.neutral_threshold
            )
            # # set matched ground truth boxes to default boxes with appropriate class
            y[batch_idx, matches[:, 1], self.num_classes: self.num_classes + 4] = gt_boxes[matches[:, 0]]
            y[batch_idx, matches[:, 1], 0: self.num_classes] = gt_classes[matches[:, 0]]  # set class scores label
            # set neutral ground truth boxes to default boxes with appropriate class
            y[batch_idx, neutral_boxes[:, 1], self.num_classes: self.num_classes + 4] = gt_boxes[neutral_boxes[:, 0]]
            y[batch_idx, neutral_boxes[:, 1], 0: self.num_classes] = np.zeros((self.num_classes))  # neutral boxes have a class vector of all zeros
            # encode the bounding boxes
            y[batch_idx] = encode_bboxes(y[batch_idx])
            X.append(input_img)

        X = np.array(X, dtype=np.float)

        return X, y

    def process_input_fn(self, x):
        return self._preprocess_callback(x)

    # def __augment(self, image, bboxes, classes):
    #     augmentations = [
    #         photometric.random_brightness,
    #         photometric.random_contrast,
    #         photometric.random_hue,
    #         photometric.random_lighting_noise,
    #         photometric.random_saturation,
    #         geometric.random_expand,
    #         geometric.random_crop,
    #         geometric.random_horizontal_flip,
    #         geometric.random_vertical_flip,
    #     ]
    #     augmented_image, augmented_bboxes, augmented_classes = image, bboxes, classes
    #     for aug in augmentations:
    #         augmented_image, augmented_bboxes, augmented_classes = aug(
    #             image=augmented_image,
    #             bboxes=augmented_bboxes,
    #             classes=augmented_classes
    #         )

    #     return augmented_image, augmented_bboxes, augmented_classes

class PlotGrid:
    def __init__(self, size=(20, 20), columns=4):
        self.size = size
        self.cols = columns

    def plot_grid(self, annos: List[Annotation], title: str):
        imgs = [anno.get_resized_annotation(self.size) for anno in annos]
        n_images = len(imgs)
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle(title)
        rows = n_images // self.cols + 1 if n_images % self.cols != 0 else 0
        for i in range(len(imgs)):
            img_anno = imgs[i].image
            img = np.array(img_anno)
            self.draw_bbox(img, imgs[i].bbox.coordinates)
            fig.add_subplot(rows, self.cols, i + 1)
            plt.axis('off')
            plt.title(' '.join(imgs[i].class_name.split(' ')[:2]))
            plt.imshow(img)
        image_save_path = Path(title.lower() + '.png')
        plt.savefig(image_save_path)
        print('Image {} saved as {}'.format(title, image_save_path))

    def draw_bbox(self, img, bbox):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
