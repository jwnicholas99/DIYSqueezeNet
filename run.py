import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

from pruning import prune_model
from quantization import pre_quantize, quantize, save_model, load_model, display_diff
from huffman_opt import encode, decode, compare_encoding_size 
from model import SqueezeNet
import pathlib

IMAGE_DIM = 150
NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_EPOCHS = 4
IS_CALTECH = 0

# Prepare a directory to store all the checkpoints.
checkpoint_dir = './ckpt'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def make_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
        SqueezeNet(num_classes)
        ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    model.build((None, IMAGE_DIM, IMAGE_DIM, 3))
    model.summary()

    return model

def make_or_restore_model(num_classes):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + '/' + name
                   for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Restoring from', latest_checkpoint)
        return tf.keras.models.load_model(latest_checkpoint)
    print('Creating a new model')
    return make_model(num_classes)

def load_intel6():
    train_data_dir = pathlib.Path('intel6/seg_train')
    test_data_dir = pathlib.Path('intel6/seg_test')
    
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        seed=123,
        image_size=(IMAGE_DIM, IMAGE_DIM),
        batch_size=BATCH_SIZE)

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        seed=123,
        image_size=(IMAGE_DIM, IMAGE_DIM),
        batch_size=BATCH_SIZE)

    # Caching best practices
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
    test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

    return train_data, test_data

def load_caltech256():
    train_data_dir = pathlib.Path('256_ObjectCategories')
    
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        seed=123,
        image_size=(IMAGE_DIM, IMAGE_DIM),
        batch_size=BATCH_SIZE)
    
    # Caching best practices
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)

    return train_data

def main():
    # Load dataset:
    print("-" * 30)
    print("[+] Start Loading Dataset")
    if IS_CALTECH:
        train_data, load_caltech256()
    else:
        train_data, test_data = load_intel6()
    model = make_or_restore_model(NUM_CLASSES)


    # Training:
    print("-" * 30)
    print("[+] Start Training")
    my_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + '/ckpt-loss={loss:.2f}',
            period = 2),
    ]
    if IS_CALTECH:
        model.fit(train_data, epochs=NUM_EPOCHS, validation_split=0.1, callbacks=my_callbacks)
    else:
        model.fit(train_data, epochs=NUM_EPOCHS, validation_data=test_data, callbacks=my_callbacks)


    # Prune model:
    print("-" * 30)
    print("[+] Start Pruning")
    if IS_CALTECH:
        model = prune_model(model, train_data)
    else:
        model = prune_model(model, train_data, test_data)


    # Pre-quanitzation. Fill in parameters to increase accuracy.
    """
    q_aware_model = pre_quantize(model,
             optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
             metrics=['accuracy'],
             train_data=train_data,
             num_shards=5,
             batch_size=64,
             epochs=1,
             validation_split=0.1)
    """

    # Quantization Call
    print("-" * 30)
    print("[+] Start Quantization")
    quantized_tflite_model = quantize(model)

    # Save model
    file_path = 'DIYSqueezeNet.tflite'
    save_model(quantized_tflite_model, file_path)
    display_diff(model, quantized_tflite_model)

    # Load model
    interpreter = load_model(file_path)

    print("TODO: See which of these layers are important. Look for quantization layer.")
    for x in enumerate(interpreter.get_tensor_details()):
        print("[{}]: {}".format(x[0], x[1]["name"]), flush=True)
        print(x[1]["shape"])

    important_tensor = interpreter.get_tensor(52)

    # Huffman Encoding
    print("-" * 30)
    print("[+] Start Huffman Encoding")
    code, codec = encode(important_tensor)
    output_weights = decode(code, codec, shape=important_tensor.shape)
    compare_encoding_size(code, output_weights)

    assert(np.all(tf.equal(important_tensor, output_weights)))


if __name__ == "__main__":
    print(tf.__version__)
    main()
