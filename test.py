import cv2
import argparse
from tensorflow.keras.applications import mobilenet_v2
import numpy as np
from car_classifier.networks import SSD_MOBILENETV2
from car_classifier.config import config
from pathlib import Path
import os

dataset_dir = Path(os.getenv('DIR_DATASETS') or 'dataset').joinpath('car_classification')
label_maps_path = dataset_dir.joinpath('names.csv')
parser = argparse.ArgumentParser(description='run inference on an input image.')

parser.add_argument('--input_image', type=str, help='path to the input image.')
parser.add_argument('--weights', type=str, help='path to the weight file.')
parser.add_argument('--confidence_threshold', type=float, help='the confidence score a detection should match in order to be counted.', default=0.1)
parser.add_argument('--num_predictions', type=int, help='the number of detections to be output as final detections', default=10)
args = parser.parse_args()

assert args.num_predictions > 0, "num_predictions must be larger than zero"
assert args.confidence_threshold > 0, "confidence_threshold must be larger than zero."
assert args.confidence_threshold <= 1, "confidence_threshold must be smaller than or equal to 1."

with open(label_maps_path, "r") as file:
    label_maps = [line.strip("\n") for line in file.readlines()]

input_size = config["model"]["input_size"]
model_config = config["model"]

model = SSD_MOBILENETV2(
    config,
    label_maps,
    is_training=False,
    num_predictions=args.num_predictions)
process_input_fn = mobilenet_v2.preprocess_input

model.load_weights(args.weights)

image = cv2.imread(args.input_image)
display_image = image.copy()
image_height, image_width, _ = image.shape
height_scale, width_scale = input_size/image_height, input_size/image_width
image = cv2.resize(image, (input_size, input_size))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = process_input_fn(image)

image = np.expand_dims(image, axis=0)
y_pred = model.predict(image)

bbox_line_width = int((image_width * 0.5) / 300)
text_scale = (image_width * 0.3) / 300

for i, pred in enumerate(y_pred[0]):
    classname = label_maps[int(pred[0]) - 1].upper()
    confidence_score = pred[1]
    print(f"-- {classname}: {'%.2f' % (confidence_score * 100)}%")
    if confidence_score < 1 and confidence_score > args.confidence_threshold:
        xmin = max(int(pred[2] / width_scale), bbox_line_width)
        ymin = max(int(pred[3] / height_scale), bbox_line_width)
        xmax = min(int(pred[4] / width_scale), image_width-bbox_line_width)
        ymax = min(int(pred[5] / height_scale), image_height-bbox_line_width)
        cv2.putText(
            display_image,
            f"{classname} {'%.2f' % (confidence_score * 100)}%",
            (int(xmin + (text_scale * 10)), int(ymin + (text_scale * 20))),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            (100, 100, 255),
            bbox_line_width, 1)
        cv2.rectangle(
            display_image,
            (xmin, ymin),
            (xmax, ymax),
            (0, 255, 0),
            bbox_line_width)

cv2.imshow("image", display_image)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
