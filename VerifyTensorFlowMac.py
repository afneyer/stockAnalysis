from unittest import TestCase

import tensorflow as tf

def run_tf():

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

    def test_run(self):
        run_tf()

