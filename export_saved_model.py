#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: limohan
"""
import os
import shutil
import json
import requests
import base64
import tensorflow as tf
from tensorflow.contrib import predictor

gpu_options = tf.GPUOptions(allow_growth=True)
tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)


# ================================= 常规方式构建的model ==================================
class Model(object):
    def __init__(self):
        self._inputs = tf.placeholder(tf.int32, shape=(None, FLAGS.max_seq_length), name='inputs')
        self._labels = tf.placeholder(tf.int32, name='labels')


def export(checkpoint_file, saved_model_dir, signature_key='classification_model'):
    """
    Builds a prediction graph and exports the model.
    模型导出
    """
    # dir tf_serving_model

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(config=tf_config)
        with sess.as_default():
            # 读取模型
            saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            # 选择需要的tensor
            _input = graph.get_tensor_by_name('inputs:0')
            _probs = graph.get_tensor_by_name('probs:0')

            # saved_model_dir = os.path.join(saved_model_dir, '1')
            # remove exist files before export
            if os.path.exists(saved_model_dir):
                shutil.rmtree(saved_model_dir)

            # 模型导出
            builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
            inputs = {
                'input': tf.saved_model.utils.build_tensor_info(_input)
            }
            outputs = {
                'preds': tf.saved_model.utils.build_tensor_info(_probs)
            }
            signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs, outputs, method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

            builder.add_meta_graph_and_variables(
                sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
                signature_def_map={signature_key: signature}, clear_devices=True)

            builder.save()
            print(f'Model export to path: {saved_model_dir}, signature_key is: {signature_key}')


# tensorflow saved_model
with tf.Session(config=tf_config) as sess:
    # saved_model_dir = os.path.join(cfg.saved_model_dir, '1')
    loaded_model = tf.saved_model.load(sess, tags=[tf.saved_model.tag_constants.SERVING],
                                       export_dir=saved_model_dir)

    output = sess.run('ouput_feat:0', feed_dict={'input:0': word_ids})

# contrib predictor
input_dict = {'input': word_ids}
model = predictor.from_saved_model(model_path)
output = model(input_dict)


# tensorflow_serving payload
def tf_serving_request():
    word_ids = [0, 0, 0, 0, 0, 0]
    payload = {'signature_name': 'classification_model', # 与export_model的signature_name保持一致
            'inputs': {'input': word_ids}}
    model_name = 'my_model'
    tf_serving_url = f'http://127.0.0.1:8501/v1/models/{model_name}:predict'

    r = requests.post(tf_serving_url, data=json.dumps(payload))

    # word_ids: batch_size * num_unroll
    # batch = 1: word_ids: [[59, ..., 0]]
    # batch > 1: word_ids: [[59, ..., 0],[48, ..., 0], ... , [78, ..., 0]]


# ======================== tf_estimator构建的model approach 1 ============================
def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example."""
    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           name='input_example_tensor')

    receiver_tensors = {'examples': serialized_tf_example}
    feature_spec = {"unique_ids": tf.FixedLenFeature([], tf.int64),
                    "input_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                    "input_mask": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                    "input_type_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                    }
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

# checkpoint_path: The checkpoint path to export.  If `None` (the default),
# the most recent checkpoint found within the model directory is chosen.
# assets_extra: Extra assets may be written into the SavedModel via the assets_extra argument. The simple case
# of copying a single file without renaming it is specified as
# {'my_asset_file.txt': '/path/to/my_asset_file.txt'}
estimator.export_saved_model(FLAGS.output_dir, serving_input_receiver_fn)


def serialize_example(example, max_seq_length, tokenizer):
    # ex_index>5 避免打印信息
    feature = convert_single_example(example,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["input_type_ids"] = create_int_feature(feature.input_type_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    example_proto = tf_example.SerializeToString()

    return example_proto


# predict with saved model
example_proto = serialize_example(example, FLAGS.max_seq_length, tokenizer)

# one_batch: [example_proto], batches: [example_proto1, example_proto2, ...]
input_dict = {'examples': [example_proto]}
model = predictor.from_saved_model(model_path)
output = model(input_dict)


# tensorflow_serving payload
def tf_serving_request():
    payload = {}

    example_proto = serialize_example(example, FLAGS.max_seq_length, tokenizer)
    payload['signature_name'] = 'serving_default'

    # Input Tensors in row ("instances") or columnar ("inputs") format. A request can have either of them but NOT both.
    # "instances": <value>|<(nested)list>|<list-of-objects>
    #  "inputs": <value>|<(nested)list>|<object>

    # one_batch
    payload['instances'] = [{'examples': {'b64': base64.b64encode(example_proto).decode('utf-8')}}]
    # batches:
    # payload['instances'] = [{'examples': {'b64': base64.b64encode(example_proto1).decode('utf-8')}},
    #                         {'examples': {'b64': base64.b64encode(example_proto2).decode('utf-8')}}
    #                         ]

    model_name = 'my_model'
    tf_serving_url = f'http://127.0.0.1:8501/v1/models/{model_name}:predict'

    r = requests.post(tf_serving_url, data=json.dumps(payload))

# ======================== tf_estimator构建的model approach 2 ============================
features_spec = {
    "unique_ids": tf.placeholder(tf.int64, name="unique_ids"),
    "input_ids": tf.placeholder(tf.int64, shape=[None, FLAGS.max_seq_length], name="input_ids"),
    "input_mask": tf.placeholder(tf.int64, shape=[None, FLAGS.max_seq_length], name="input_mask"),
    "input_type_ids": tf.placeholder(tf.int64, shape=[None, FLAGS.max_seq_length], name="input_type_ids"),
}

serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features_spec)
estimator.export_saved_model(FLAGS.output_dir, serving_input_receiver_fn)

feature = convert_single_example(example,
                                 max_seq_length, tokenizer)

input_dict = {}
input_dict["unique_ids"] = feature.unique_id
input_dict["input_ids"] = feature.input_ids
input_dict["input_mask"] = feature.input_mask
input_dict["input_type_ids"] = feature.input_type_ids

model = predictor.from_saved_model(model_path)
output = model(input_dict)

