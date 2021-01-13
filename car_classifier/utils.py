from multiprocessing import Pool
from pathlib import Path
from random import choices
from typing import Any, List

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from dataclasses import dataclass
from matplotlib import pyplot as plt


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
        self.ALL_LABELS = file_path.parent.joinpath('names.csv').read_text().splitlines()

    def __getitem__(self, key):
        image = self.get_image(key)
        *bbox_, class_label = self._df.loc[key]
        return Annotation(image, BBox(*bbox_), class_label, self.ALL_LABELS[class_label - 1])

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


class Dataloader:

    def __init__(self, anno_reader, batch_size, workers=4):
        self.anno_reader = anno_reader
        self.batch_size = batch_size
        self._keys = anno_reader.all_keys
        self._workers = workers

    def __iter__(self):
        with Pool(self._workers) as pool:
            for batch in pool.imap_unordered(self.get_nth_batch, range(0, len(self.anno_reader), self.batch_size)):
                yield batch

    def get_nth_batch(self, batch_id):
        return [self.anno_reader[key].image_label_tensors() for key in self._keys[batch_id:batch_id+self.batch_size]]

    def preprocess(image):
        return image


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
