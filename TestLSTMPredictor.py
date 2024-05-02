from unittest import TestCase

import tensorflow as tf
from tensorflow.python.client import device_lib

from LSTMPredictor import LSTMPredictor


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


class TestTensorFlow(TestCase):

    def test_devices(self):
        print(tf.config.list_physical_devices())
        print(tf.config.list_logical_devices())
        print(get_available_devices())

    def test_train_cpu(self):
        with tf.device('/cpu:0'):
            lp = LSTMPredictor()
            lp.EPOCHS = 1
            data, model = lp.run_model_fit()
            lp.run_model_evaluation(data, model)

    def test_run_gpu(self):
        with tf.device('/gpu:0'):
            lp = LSTMPredictor()
            lp.EPOCHS = 1
            data, model = lp.run_model_fit()
            lp.run_model_evaluation(data, model)

