import glob
import os.path
from PIL import Image
import numpy as np
import tensorflow as tf


INPUT_ALL_DATA = './data'
INPUT_TRAIN_DATA = './data/train'
INPUT_TEST_DATA = './data/valid'
OUTPUT_TRAIN_FILE = './data/train.tfrecords'
OUTPUT_TEST_FILE = './data/valid.tfrecords'

num_records = 10


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_img_data(sub_dirs, INPUT_DATA):
    current_label = 0
    is_root_dir = True
    print("file directory: " + INPUT_DATA)

    record_name = INPUT_DATA + ".tfrecords"
    writer = tf.io.TFRecordWriter(record_name)

    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        file_list = []
        dir_name = os.path.basename(sub_dir)

        file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + "jpg")
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue
        index = 0
        for file_name in file_list:
            image = np.array(Image.open(file_name).resize((224, 224))).astype(np.float32) / 128 - 1
            image = image.tostring()

            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(image),
                'label': _int64_feature(current_label)
            }))
            index = index + 1
            if index % 50 == 0:
                break
            writer.write(tf_example.SerializeToString())
            print("处理文件索引%d index%d"%(current_label,index))
        current_label += 1
    writer.close()


def create_image_lists():

    sub_dirs = [x[0] for x in os.walk(INPUT_TRAIN_DATA)]
    get_img_data(sub_dirs, INPUT_TRAIN_DATA)

    sub_test_dirs = [x[0] for x in os.walk(INPUT_TEST_DATA)]
    get_img_data(sub_test_dirs, INPUT_TEST_DATA)


def main():
    create_image_lists()
    print('success')


if __name__ == '__main__':
    main()