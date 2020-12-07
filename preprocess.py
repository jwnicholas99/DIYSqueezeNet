import numpy as np
import tensorflow as tf
import pickle
import os
#import tensorflow.keras.preprocessing.image as ImagePreprocess

def unpickle(file):
    if type(file) != str:
        file = file.numpy().decode('utf-8')
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_databatch(data_folder, img_size=32, is_train=True):
    if is_train:
        data_file = data_folder #os.path.join(data_folder, 'train_data_batch_')
        d = unpickle(data_file)# + str(idx))
    else:
        data_file = os.path.join(data_folder, 'val_data')
        d = unpickle(data_file)

    x = d['data']
    y = d['labels']
    x = x/np.float32(255)
    data_size = x.shape[0]
    x = np.reshape(x, [data_size, img_size, img_size, 3])
    return [tf.reshape(tf.convert_to_tensor(x, dtype=tf.float64), shape=[data_size, img_size, img_size, 3]),
            tf.math.subtract(tf.convert_to_tensor(y, dtype=tf.float64), tf.constant(1.0, dtype=tf.float64))]

def get_data(train_file_path, test_file_path, batch_size=1000):
    train_set = [os.path.join(train_file_path, 'train_data_batch_' + str(i+1)) for i in range(10)]
    train_files_dataset = tf.data.Dataset.list_files(train_set)
    train_data = train_files_dataset.interleave(lambda x: tf.data.Dataset.from_tensor_slices(
            tuple(tf.py_function(func=load_databatch, inp=[x], Tout=[tf.float64, tf.float64])
            )), num_parallel_calls=2)
    #for i in range(10):
    #    train_undata = load_databatch(train_file_path, idx=i+1)
    #    train_data = tf.data.Dataset.from_tensor_slices(train_undata)
    #    if type(train_set) == str:
    #        train_set = train_data
    #    else:
    #        train_set.concatenate(train_data)
    #val_undata = load_databatch(test_file_path, is_train=False)
    #print('yes?')
    validation_data = tf.data.Dataset.from_tensor_slices(tuple(load_databatch(test_file_path, is_train=False)))
    
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
    
    return train_data, validation_data, 1000
