from absl import app, flags, logging
from absl.flags import FLAGS

from pathlib import Path

from pascal_to_tfrecord import recursive_parse_xml_to_dict
from lxml import etree
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from tqdm import tqdm 

import numpy as np

flags.DEFINE_string('input_dir', None, 'Input directory')
flags.DEFINE_string('output_dir', './output', 'Output directory')

def output_preview_image(annotation):
    xml = etree.fromstring(open(annotation).read().encode())
    annotation = recursive_parse_xml_to_dict(xml)
    
    image_file = Path(FLAGS.input_dir).joinpath('images').joinpath(annotation['annotation']['filename'])

    image = Image.open(image_file)

    bndboxes = [x['bndbox'] for x in annotation['annotation']['object']]

    bbs = BoundingBoxesOnImage([
        BoundingBox(int(b['xmin']), int(b['ymin']), int(b['xmax']), int(b['ymax'])) for b in bndboxes
    ], shape=(image.width, image.height))

    image_bbs = bbs.draw_on_image(np.array(image))

    output_dir = Path(FLAGS.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    output_file = output_dir.joinpath(image_file.name)
    image_bbs = Image.fromarray(image_bbs)
    image_bbs.save(output_file)

def main(_argv):
    input_dir = Path(FLAGS.input_dir)
    annotations = input_dir.glob('annotations/*.xml')

    for annotation in tqdm(annotations):
        output_preview_image(annotation)

if __name__ == '__main__':
    app.run(main)