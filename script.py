from pathlib import Path
import os
from car_classifier.utils import DataAnnotationReader, Dataloader, PlotGrid

dataset_dir = Path(os.getenv('DIR_DATASETS') or 'dataset').joinpath('car_classification')

data_dir = dataset_dir.joinpath('car_data')
test_dir = data_dir.joinpath('test')
train_dir = data_dir.joinpath('train')

csv_test_label = dataset_dir.joinpath('anno_test.csv')
csv_train_label = dataset_dir.joinpath('anno_train.csv')


BATCH_SIZE = 10

def main():
    train_annotations_reader = DataAnnotationReader(csv_train_label, data_dir, False)
    test_annotations_reader = DataAnnotationReader(csv_test_label, data_dir, True)
    p = PlotGrid(size=(100, 100), columns=4)
    p.plot_grid(train_annotations_reader.sample_annos(n=10), title='Training Sample')
    p.plot_grid(test_annotations_reader.sample_annos(n=10), title='Testing Sample')
    dl = Dataloader(train_annotations_reader, 10)
    count = 0
    for batch in dl:
        print(count, len(batch))
        count += 1

if __name__ == '__main__':
    main()
