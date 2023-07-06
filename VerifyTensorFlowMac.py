from unittest import TestCase

import tensorflow as tf
from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU' or x.device_type == 'CPU']


def run_tf():
    # this model runs clearly faster on GPU than CPU (66 sec versus 344 sec per Epoch)
    cifar = tf.keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    model = tf.keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_shape=(32, 32, 3),
        classes=100, )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=64)
class TestTensorFlow(TestCase):

    def test_devices(self):
        print(tf.config.list_physical_devices())
        print(tf.config.list_logical_devices())
        print(get_available_devices())

    def test_run_cpu(self):
        with tf.device('/cpu:0'):
            run_tf()

    def test_run_gpu(self):
        with tf.device('/gpu:0'):
            run_tf()

