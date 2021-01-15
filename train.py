from pathlib import Path
import os
from car_classifier.utils import DataAnnotationReader, Dataloader, PlotGrid
import matplotlib.pyplot as plt
from tensorflow.keras.applications import  mobilenet_v2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from car_classifier.losses import SSD_LOSS
from car_classifier.networks import SSD_MOBILENETV2
from car_classifier.config import config as mobilenet_v2_config_json

dataset_dir = Path(os.getenv('DIR_DATASETS') or 'dataset').joinpath('car_classification')
data_dir = dataset_dir.joinpath('car_data')
test_dir = data_dir.joinpath('test')
train_dir = data_dir.joinpath('train')

csv_test_label = dataset_dir.joinpath('anno_test.csv')
csv_train_label = dataset_dir.joinpath('anno_train.csv')

training_config = mobilenet_v2_config_json['training']
model_config = mobilenet_v2_config_json['model']
learning_rate = 10e-3


def main():
    epochs = 200
    batch_size = 150
    checkpoint_frequency = 50
    out_dir = Path(os.getenv('DIR_DL_MODELS') or '.').joinpath('car_classifier')
    out_dir.mkdir(parents=True, exist_ok=True)
    train_annotations_reader = DataAnnotationReader(csv_train_label, data_dir, False)
    test_annotations_reader = DataAnnotationReader(csv_test_label, data_dir, True)
    p = PlotGrid(size=(100, 100), columns=4)
    p.plot_grid(train_annotations_reader.sample_annos(n=10), title='Training Sample')
    p.plot_grid(test_annotations_reader.sample_annos(n=10), title='Testing Sample')
    dl = Dataloader(train_annotations_reader, 10,
                    config=mobilenet_v2_config_json,
                    preprocess_func=mobilenet_v2.preprocess_input,
                    augment=False)

    # make model
    model = SSD_MOBILENETV2(config=mobilenet_v2_config_json,
                            label_maps=train_annotations_reader.ALL_LABELS[1:])
    loss = SSD_LOSS(
        alpha=training_config["alpha"],
        min_negative_boxes=training_config["min_negative_boxes"],
        negative_boxes_ratio=training_config["negative_boxes_ratio"]
    )
    optimizer = SGD(
        lr=learning_rate,
        momentum=0.9,
        decay=0.0005,
        nesterov=False
    )
    model.compile(optimizer=optimizer, loss=loss.compute)

    # training starts here
    history = model.fit(
        x=dl,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=len(train_annotations_reader)//batch_size,
        callbacks=[
            ModelCheckpoint(
                os.path.join(f'{out_dir}', 'cp_{epoch:02d}_{loss:.4f}.h5'),
                save_weights_only=True,
                save_freq=(len(train_annotations_reader)//batch_size) * checkpoint_frequency
            ),
        ]
    )

    model.save_weights(os.path.join(out_dir, "model.h5"))
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(os.path.join(out_dir, "training_graph.png"))

if __name__ == '__main__':
    main()
