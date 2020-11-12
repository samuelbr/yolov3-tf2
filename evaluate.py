from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from tqdm import tqdm

flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    dataset = load_tfrecord_dataset(
        FLAGS.tfrecord, FLAGS.classes, FLAGS.size)

    total_matched = 0
    total_count = 0
    total_iou = 0
    false_detected = 0

    for img_raw, _label in tqdm(dataset):
        _label = _label.numpy()
        true_labels = np.sum(np.sum(_label, axis=1) > 0)

        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)

        boxes, scores, classes, nums = yolo(img)

        total_count += true_labels

        for i in range(nums[0]):
            iou = 0
            for i_true in range(true_labels):
                if _label[i_true][-1] != int(classes[0][i]):
                    continue
                iou_current = bb_intersection_over_union(_label[i_true][:4], np.array(boxes[0][i]))
                if iou_current > iou:
                    iou = iou_current

            if iou > 0:
                total_matched += 1
            else:
                false_detected += 1
            total_iou += iou
    logging.info(f'Total objects: {total_count}\n\t matched: {total_matched} ratio: {total_matched / total_count:.3}\n\t false detected: {false_detected}\n\tavgIOU: {total_iou / total_count:.3} matched avgIOU: {total_iou / total_matched:.3}')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
