from absl.logging import log
import numpy as np
import tensorflow as tf
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
from pathlib import Path
from PIL import Image
from lxml import etree
from io import BytesIO

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

flags.DEFINE_string('data_dir', '', 'path to image files')
flags.DEFINE_string('output_file', './data/cars.tfrecord', 'output dataset')
flags.DEFINE_integer('target_size', 416, 'target image size')
flags.DEFINE_boolean('allow_empty', True, 'Allow examples without annotations')

def recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.
    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree
    Returns:
        Python dictionary holding XML contents.
    """
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def pascal_load(annotation_file_path):
    """
    Read image from Pascal dataset
    Args:
        annotation_file_path: Annotation file path
    Returns: PIL Image, BoundingBoxes, Annotations object
    """
    data = etree.fromstring(open(annotation_file_path).read())
    annotations = recursive_parse_xml_to_dict(data)['annotation']

    image_folder = annotation_file_path.parent.parent.joinpath('images')
    image_path = image_folder.joinpath(annotations['filename'])


    items = []
    if 'object' in annotations:
        for obj in annotations['object']:
            bndbox = obj['bndbox']
            xmin = int(float(bndbox['xmin']))
            ymin = int(float(bndbox['ymin']))
            xmax = int(float(bndbox['xmax']))
            ymax = int(float(bndbox['ymax']))

            items.append((xmin, ymin, xmax, ymax))

    img = np.array(Image.open(image_path).convert('RGB'))
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1, y1, x2, y2) for x1, y1, x2, y2 in items
    ], shape=img.shape)

    return img, bbs, annotations

def pascal_build_example(img, bndbox, annotation, class_map, allow_empty=False, min_bbox_width = 0, min_bbox_height = 0):
    """
        Built tf.train.Example from image - use Pascal format.
        Arsgs:
            img: Image numpy array
            bndbox: BoundingBoxes
            annotation: Image annotations
            class_map: Classes map
    """
    key = hashlib.sha256(img).hexdigest()

    height = img.shape[0]
    width = img.shape[1]

    img_raw = BytesIO()
    Image.fromarray(img).convert('RGB').save(img_raw, format='jpeg')

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    views = []
    difficult_obj = []
    if 'object' in annotation:
        for idx, obj in enumerate(annotation['object']):
            if bndbox[idx].is_out_of_image(img.shape):
                continue
            box_norm = bndbox[idx].clip_out_of_image(img.shape)

            if box_norm.width < min_bbox_width or box_norm.height < min_bbox_height:
                continue
            
            difficult = bool(int(obj['difficult']))
            difficult_obj.append(int(difficult))

            xmin.append(float(box_norm.x1) / width)
            ymin.append(float(box_norm.y1) / height)
            xmax.append(float(box_norm.x2) / width)
            ymax.append(float(box_norm.y2) / height)

            obj_name = obj['name'].encode('utf8')
            if not (obj_name in class_map):
                class_map[obj_name] = len(class_map)

            classes_text.append(obj_name)
            classes.append(class_map[obj_name])
            truncated.append(int(obj['truncated']))
            views.append(obj['pose'].encode('utf8'))

    if not allow_empty and len(xmin) == 0:
        return None
 
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw.getvalue()])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example

def main(_argv):
    data_dir = Path(FLAGS.data_dir)

    annotation_files = list(data_dir.glob('annotations/*.xml'))
    logging.info(f'Loaded {len(annotation_files)} annotations')

    class_map = {'backgroud'.encode('utf8'): 0}

    with tf.io.TFRecordWriter(FLAGS.output_file) as writer:

        for annotation_file in annotation_files:
            img, bbs, annotations = pascal_load(annotation_file)
            example = pascal_build_example(img, bbs, annotations, class_map, allow_empty=FLAGS.allow_empty)
            if example:
                writer.write(example.SerializeToString())

if __name__ == '__main__':
    app.run(main)