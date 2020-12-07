import tensorflow as tf

from preprocess import get_data
from quantization import pre_quantize, quantize, save_model, load_model
from huffman_opt import encode, decode
from model import SqueezeNet
import pathlib
import os

IMAGE_DIM = 150 # Unmodified size of caltech images
NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_EPOCHS = 10
train_data_dir = pathlib.Path('intel6/seg_train')
test_data_dir = pathlib.Path('intel6/seg_test')

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

    start_learning_rate = 0.4
    end_learning_rate = 1e-4
    decay_steps = 10000
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        start_learning_rate,
        decay_steps,
        end_learning_rate,
        power=0.5)
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=learning_rate_fn),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    """
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


def main():
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

    model = make_or_restore_model(NUM_CLASSES)
    # Training:
    my_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + '/ckpt-loss={loss:.2f}',
            period = 2),
    ]
    model.fit(train_data, epochs=NUM_EPOCHS, validation_data=test_data, callbacks=my_callbacks)

    # Pre-quanitzation. Fill in parameters to increase accuracy.
    # q_aware_model = pre_quantize(model,
    #          optimizer='adam',
    #          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #          metrics=['accuracy'],
    #          train_data=train_images,
    #          train_labels=train_labels,
    #          subset_size=1000,
    #          batch_size=500,
    #          epochs=1,
    #          validation_split=0.1)

    # Quantization Call
    quantized_tflite_model = quantize(model)

    # Save model
    file_path = 'DIYSqueezeNet.tflite'
    save_model(quantized_tflite_model, file_path)

    # Load model
    interpreter = load_model(file_path)

    print("TODO: See which of these layers are important. Look for quantization layer.")
    for x in enumerate(interpreter.get_tensor_details()):
        print("[{}]: {}".format(x[0], x[1]), flush=True)

    important_tensor = interpreter.get_tensor(0)

    # Huffman Encoding
    code, codec = encode(important_tensor)
    output_weights = decode(code, codec, shape=important_tensor.shape)

    assert(np.all(tf.equal(important_tensor, output_weights)))


if __name__ == "__main__":
    print(tf.__version__)
    main()
