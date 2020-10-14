import tensorflow as tf
import tf_slim as slim
import collections
import mobilenet_v3
import matplotlib.pyplot as plt
import re
import cv2 as cv


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_paths", "./data/train.tfrecords", "The train data dir.")
tf.app.flags.DEFINE_string("eval_paths", "./data/valid.tfrecords", "The eval data dir.")
# 本次程序运行，必须往该 HDFS 目录至少写一个文件
tf.app.flags.DEFINE_string("model_path", "./model", "The output directory where the model checkpoints will be written.")
# 如果单个训练任务会触发多次程序运行，如分 Part 训练或天级/小时级更新，上一次成功的运行对应的 model_path 将通过如下命令行参数传入
tf.app.flags.DEFINE_string("init_checkpoint", "./mobilenet_v3/model.ckpt-388500",
                    "Initial checkpoint (usually from a pre-trained BERT model).")
tf.app.flags.DEFINE_integer('batch_size', 16, 'Instance count in a batch.')
tf.app.flags.DEFINE_integer('num_epochs', 200, 'Iteration count over training data.')
tf.app.flags.DEFINE_integer("eval_num_epochs", 1, "Total batch size for eval.")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "The initial learning rate for Adam.")

# 打开 estimator 日志
tf.logging.set_verbosity(tf.logging.INFO)


def input_fn_builder(input_file, num_epochs, batch_size):
    """Creates an `input_fn` closure to be passed to Estimator."""

    name_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record):
        example = tf.parse_single_example(record, name_to_features)
        example['image_raw'] = tf.decode_raw(example['image_raw'], tf.float32)
        example['image_raw'] = tf.reshape(example['image_raw'], [224, 224, 3])

        return example

    def input_fn(params):
        """The actual input function."""
        # For training, we want a lot of parallel reading and shuffling.
        d = tf.data.TFRecordDataset(input_file).map(_decode_record)
        d = d.shuffle(buffer_size=100).batch(batch_size)
        d = d.repeat(num_epochs)
        return d

    return input_fn


# def create_model(is_training, img_content, label_id):
#     """Creates a classification model."""
#     logits, endpoints = mobilenet_v3.mobilenet(img_content, num_classes=2, conv_defs=mobilenet_v3.V3_SMALL)
#
#     with tf.variable_scope("loss"):
#         loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_id)
#         prob = tf.nn.softmax(logits, axis=-1)
#
#         return loss, logits, prob

def mobilenet(input_tensor):
    with slim.arg_scope(mobilenet_v3.training_scope()):
        logits, endpoints = mobilenet_v3.mobilenet(input_tensor, num_classes=2, conv_defs=mobilenet_v3.V3_SMALL)
        return logits

def model_fn_builder(init_checkpoint, learning_rate):
    """Returns `model_fn` closure for Estimator."""

    def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
        """Compute the union of the current variables and checkpoint variables."""
        assignment_map = {}
        initialized_variable_names = {}

        name_to_variable = collections.OrderedDict()
        for var in tvars[:-2]:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
            name_to_variable[name] = var

        init_vars = tf.train.list_variables(init_checkpoint)

        assignment_map = collections.OrderedDict()
        for x in init_vars:
            (name, var) = (x[0], x[1])
            if name not in name_to_variable:
                continue
            assignment_map[name] = name
            initialized_variable_names[name] = 1
            initialized_variable_names[name + ":0"] = 1

        return (assignment_map, initialized_variable_names)

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        img_content = features["image_raw"]
        label_id = features["label"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # loss, logits, prob = create_model(is_training, img_content, label_id)
        logits = mobilenet(img_content)
        tf.losses.softmax_cross_entropy(tf.one_hot(label_id, 2), logits, weights=1.0)

        tvars = tf.trainable_variables()
        print(tvars)

        (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(
                loss=tf.losses.get_total_loss(),
                global_step=tf.train.get_global_step()
            )
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=tf.losses.get_total_loss(),
                train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.accuracy(
                predictions=tf.argmax(logits, 1),
                labels=label_id,
                name="acc_op"
            )
            eval_metric_ops = {
                "my_metric": accuracy
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                eval_metric_ops=eval_metric_ops,
                loss=tf.losses.get_total_loss())

        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"result": tf.argmax(logits, 1)})

        return output_spec

    return model_fn


def serving_input_receiver_fn():

    features = {}
    features["image_raw"] = tf.compat.v1.placeholder(tf.float32, shape=[None, 224, 224, 3], name="image_raw")
    features["label"] = tf.compat.v1.placeholder(tf.int32, shape=[None], name="label")
    return tf.estimator.export.ServingInputReceiver(features, features)


def main():

    train_input_fn = input_fn_builder(FLAGS.train_paths, FLAGS.num_epochs, FLAGS.batch_size)
    eval_input_fn = input_fn_builder(FLAGS.eval_paths, FLAGS.eval_num_epochs, FLAGS.batch_size)
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=1, log_step_count_steps=20)

    model_fn = model_fn_builder(FLAGS.init_checkpoint, FLAGS.learning_rate)

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_path, config=run_config)

    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=tf.estimator.TrainSpec(input_fn=train_input_fn),
        eval_spec=tf.estimator.EvalSpec(input_fn=eval_input_fn)
    )

    if estimator.config.is_chief:
        estimator.export_saved_model(
            export_dir_base=FLAGS.model_path + '/exported',
            serving_input_receiver_fn=serving_input_receiver_fn
        )


def read_tfrecord_pipeline(input_file):

    def convert_tfrecord(record):
        name_to_features = {
            "image_raw": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        }

        example = tf.parse_single_example(record, name_to_features)
        example['image_raw'] = tf.decode_raw(example['image_raw'], tf.float32)
        example['image_raw'] = tf.reshape(example['image_raw'], [224, 224, 3])

        return example

    d = tf.data.TFRecordDataset(input_file).map(convert_tfrecord, 4)
    d = d.batch(1).repeat(1)

    it = d.make_one_shot_iterator()
    data = it.get_next()

    sess = tf.Session()
    while True:
        try:
            a = sess.run(data)
            print(a['image_raw'].shape)
            b = a['image_raw'][0]
            b = b[:, :, (2, 1, 0)]
            cv.imshow("img", b)
            cv.waitKey()

        except tf.errors.OutOfRangeError:
            break


if __name__ == '__main__':
    # read_tfrecord_pipeline(FLAGS.train_paths)
    main()
