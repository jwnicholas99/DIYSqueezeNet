import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
from model import SqueezeNet

# Prepare a directory to store all the checkpoints.
checkpoint_dir = './ckpt'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def make_model():
    # Create a new linear regression model.
    #model = tf.keras.applications.mobilenet.MobileNet(weights=None, input_shape=(96, 96, 1), alpha = 0.25, classes=6)

    my_model = tf.keras.Sequential([ #37-40
        tf.keras.layers.experimental.preprocessing.Resizing(72, 72, interpolation='nearest'),
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
        # tf.keras.applications.mobilenet.MobileNet(weights=None, input_shape=(96, 96, 1), alpha = 0.39, include_top = False),
        # tf.keras.layers.GlobalAveragePooling2D(),
        # tf.keras.layers.Dense(6)
        SqueezeNet(6)
        ])
    my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    my_model.build((None, 96, 96, 1))
    my_model.summary()
    return my_model

def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + '/' + name
                   for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Restoring from', latest_checkpoint)
        return tf.keras.models.load_model(latest_checkpoint)
    print('Creating a new model')
    return make_model()

if __name__ == "__main__":
    print(tf.__version__)
    train_dir = pathlib.Path('intel6/seg_train')
    test_dir = pathlib.Path('intel6/seg_test')

    # Generate batches from image files.
    batch_size = 32
    esp32_image_dims = (96, 96)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        color_mode = "grayscale",
        image_size=esp32_image_dims,
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=123,
        color_mode = "grayscale",
        image_size=esp32_image_dims,
        batch_size=batch_size)

    # Further preprocessing:
    def process(image,label):
        #image = tf.cast(image, tf.float32)
        # image = tf.keras.applications.mobilenet.preprocess_input(image)
        image = tf.cast((image/127.5) - 1.0, tf.float32)
        return image,label
    train_ds = train_ds.map(process)
    val_ds = val_ds.map(process)

    # lowest_value = float('inf')
    # highest_value = float('-inf')
    # for images, labels in train_ds.take(3):
    #     for im_index in range(10):
    #         for i in range(96):
    #             for j in range(96):
    #                 pixel = images[im_index].numpy()[i, j, 0]
    #                 lowest_value = min(pixel, lowest_value)
    #                 highest_value = max(pixel, highest_value)
    # print("Range after processing: ")
    # print(" ", lowest_value, "to", highest_value)

    # Caching best practices
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Training:
    model = make_or_restore_model()
    my_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + '/ckpt-loss={loss:.2f}',
            period = 200),
    ]
    model.fit(train_ds, epochs=2000000, validation_data=val_ds, callbacks=my_callbacks)



