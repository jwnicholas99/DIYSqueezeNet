import tensorflow as tf

from preprocess import get_data
from model import SqueezeNet
import os

IMAGE_DIM = 32
BATCH_SIZE = 32
NUM_EPOCHS = 10

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

    start_learning_rate = 0.04
    end_learning_rate = 0.00001
    decay_steps = 10000
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        start_learning_rate,
        decay_steps,
        end_learning_rate,
        power=0.5)

    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=learning_rate_fn),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
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

if __name__ == "__main__":
    print(tf.__version__)
    train_data, test_data, num_classes = get_data('Imagenet8_train', 'Imagenet8_val')
    model = make_or_restore_model(num_classes)
    train_data = train_data.batch(BATCH_SIZE)
    test_data = test_data.batch(BATCH_SIZE)

    # Caching best practices
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
    test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

    # Training:
    model = make_or_restore_model(num_classes)
    my_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + '/ckpt-loss={loss:.2f}',
            period = 200),
    ]
    model.fit(train_data, epochs=NUM_EPOCHS, validation_data=test_data, callbacks=my_callbacks)


## OLD CODE: 


# def train(model, train_data):
#     '''
#     Trains the model on all of the inputs and labels for one epoch. Images are 
#     shuffled. Images should already have been randomnly transformed in get_data()
#     :param model: model to run
#     :param inputs: train inputs (all inputs to use for training), 
#     shape (num_inputs, width, height, num_channels)
#     :param labels: train labels (all labels to use for training), 
#     shape (num_labels, num_classes)
#     '''
#     train_data = train_data.shuffle(1281167)
#     #train_data = train_data.map(tf.image.random_flip_left_right)
#     #train_data = train_data.map(tf.image.random_flip_up_down)

#     for batch in train_data:
#         print(batch)
#         batch_x = batch[0]
#         batch_y = tf.cast(batch[1], tf.int32)

#         with tf.GradientTape() as tape:
#             probs = model.call(batch_x)
#             loss = model.loss(probs, batch_y)
#         print("Loss: ", loss)
#         gradients = tape.gradient(loss, model.trainable_variables)
#         model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return

# def test(model, test_data):
#     """
#     Tests the model on the test inputs and labels. 
#     :param model: SqueezeNet model to run
#     :param inputs: test data (all images to be tested), 
#     shape (num_inputs, width, height, num_channels)
#     :param labels: test labels (all corresponding labels),
#     shape (num_labels, num_classes)
#     :return top1_accuracy and top5_accuracy
#     """
#     top1_num_corr = 0
#     top5_num_corr = 0
#     label_length = tf.shape(test_data[1])[0]

#     for batch in test_data:
#         batch_x = batch[0]
#         batch_y = tf.cast(batch[1], tf.int32)

#         probs = model.call(batch_x)
#         top1_acc = model.top1_accuracy(probs, batch_y)
#         top5_acc = model.top5_accuracy(probs, batch_y)
#         top1_num_corr += top1_acc * len(batch_y)
#         top5_num_corr += top5_acc * len(batch_y)
#     top1_acc = top1_num_corr / label_length
#     top5_acc = top5_num_corr / label_length
#     return top1_acc, top5_acc

# def main():
#     train_data, test_data, num_classes = get_data('Imagenet8_train', 'Imagenet8_val')
#     model = SqueezeNet(num_classes)
#     train_data = train_data.batch(model.batch_size)
#     test_data = test_data.batch(model.batch_size)
#     for _ in range(10):
#         train(model, train_data)
#         top1_acc, top5_acc = test(model, test_data)
#         print("Top 1 Accuracy: ", top1_acc)
#         print("Top 5 Accuracy: ", top5_acc)
#     return


# if __name__=='__main__':
#     main()
