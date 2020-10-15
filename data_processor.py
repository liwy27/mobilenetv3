import glob
import os.path
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


def get_img_data(sub_dirs, INPUT_DATA, sess):
    current_label = 0
    is_root_dir = True
    print("file directory: " + INPUT_DATA)

    record_name = INPUT_DATA + "-00.tfrecords"
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
            image_raw_data = tf.gfile.FastGFile(file_name, 'rb').read()
            image = tf.io.decode_jpeg(image_raw_data, channels=3)
            image = tf.cast(image, tf.float32) / 128. - 1
            image = tf.image.resize_images(image, [224, 224])
            image_value = sess.run(image)
            image_raw = image_value.tostring()

            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(image_raw),
                'label': _int64_feature(current_label)
            }))

            index = index + 1
            if index % 50 == 0:
                writer.close()
                record_name = INPUT_DATA + "-%.2d.tfrecords" % (index // 50)
                print(record_name)
                writer = tf.io.TFRecordWriter(record_name)
            writer.write(tf_example.SerializeToString())

            print("处理文件索引%d index%d"%(current_label,index))
        current_label += 1


def create_image_lists(sess):

    sub_dirs = [x[0] for x in os.walk(INPUT_TRAIN_DATA)]
    get_img_data(sub_dirs, INPUT_TRAIN_DATA, sess)

    sub_test_dirs = [x[0] for x in os.walk(INPUT_TEST_DATA)]
    writer_test = tf.io.TFRecordWriter(OUTPUT_TEST_FILE)
    get_img_data(sub_test_dirs, INPUT_TEST_DATA, sess)


def main():
    with tf.Session() as sess:
        create_image_lists(sess)
        print('success')


if __name__ == '__main__':
    main()