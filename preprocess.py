import numpy as np
import tensorflow as tf
import pickle
import os
import tensorflow.keras.preprocessing.image as ImagePreprocess

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_databatch(data_folder, idx, img_size=32):
    data_file = os.path.join(data_folder, 'train_data_batch_')

    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']
    data_size = x.shape[0]
    X_train = x
    Y_train = np.array(y)
    """mean_image = d['mean']

    x = x/np.float32(255)
    mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)"""

    return tf.reshape(tf.convert_to_tensor(X_train), shape=[data_size, 8, 8, 3]), Y_train.astype('int32')#,
        #mean=mean_image)

def get_data(file_path):
    #train_data = tf.io.read_file(file_path)
    #print(train_data)
    x_train, y_train = load_databatch('Imagenet8_train', 1)
    
    train_datagen = ImagePreprocess.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30)
    test_datagen = ImagePreprocess.ImageDataGenerator(rescale=1./255)
    train_datagen.fit(x_train)
    print(x_train[:10])

    """train_generator = train_datagen.flow_from_directory(
        'Imagenet8_train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')"""
    
    """validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')"""
    #return train_generator, validation_generator

if __name__ == "__main__":
    #mydict = load_databatch('Imagenet8_train', 1)
    #print(mydict)
    get_data('asdf')


    """
    model.fit(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
    """
