import numpy as np
import tensorflow as tf
import pickle
import os
#import tensorflow.keras.preprocessing.image as ImagePreprocess

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_databatch(data_folder, idx, img_size=32, is_train=True):
    if is_train:
        data_file = os.path.join(data_folder, 'train_data_batch_')
        d = unpickle(data_file + str(idx))
    else:
        data_file = os.path.join(data_folder, 'val_data')

    
    x = d['data']
    y = d['labels']
    x = x/np.float32(255)
    data_size = x.shape[0]

    return (tf.reshape(tf.convert_to_tensor(x, dtype=tf.float64), shape=[data_size, img_size, img_size, 3]),
            tf.math.subtract(tf.convert_to_tensor(y, dtype=tf.int64), tf.Constant(1)))

def get_data(train_file_path, test_file_path, batch_size=1000):
    train_set = train_file_path
    for i in range(10):
        train_data = tf.data.Dataset.from_generator(load_databatch, output_types=(tf.float64, tf.int64),
                                                    args=(tf.Constant(train_file_path), tf.Constant(i+1)))
        if type(train_set) == str:
            train_set = train_data
        else:
            train_set.concatenate(train_data)
    validation_data = tf.data.Dataset.from_generator(load_databatch, output_types=(tf.float64, tf.int64),
                                                    args=(tf.Constant(test_file_path)))
    train_set = train_set.map(tf.image.random_flip_left_right)
    train_set = train_set.map(tf.image.random_flip_up_down)
    train_set = train_set.shuffle()
    train_set = train_set.batch(batch_size)
    validation_data = validation_data.batch(batch_size)
    
    """train_datagen = ImagePreprocess.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30)
    test_datagen = ImagePreprocess.ImageDataGenerator(rescale=1./255)
    

    train_generator = train_datagen.flow(
        train_data[:, 0], train_data[:, 1],
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')
    
    validation_generator = test_datagen.flow(
        validation_data[:, 0], validation_data[:, 1],
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')
    """
    
    return train_set, validation_data

if __name__ == "__main__":
    train_gen, val_gen = get_data('Imagenet32_train', 'Imagenet32_val')
