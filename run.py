import tensorflow as tf

from preprocess import get_data
from model import SqueezeNet

def train(model, train_data):
    '''
    Trains the model on all of the inputs and labels for one epoch. Images are 
    shuffled. Images should already have been randomnly transformed in get_data()
    :param model: model to run
    :param inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    '''
    train_data = train_data.shuffle(1281167)
    #train_data = train_data.map(tf.image.random_flip_left_right)
    #train_data = train_data.map(tf.image.random_flip_up_down)

    for batch in train_data:
        batch_x = batch[0]
        batch_y = tf.cast(batch[1], tf.int32)

        with tf.GradientTape() as tape:
            probs = model.call(batch_x)
            loss = model.loss(probs, batch_y)
        #print("Loss: ", loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        break
    return

def test(model, test_data):
    """
    Tests the model on the test inputs and labels. 
    :param model: SqueezeNet model to run
    :param inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return top1_accuracy and top5_accuracy
    """
    top1_num_corr = 0
    top5_num_corr = 0
    label_length = 0

    for batch in test_data:
        batch_x = batch[0]
        batch_y = tf.cast(batch[1], tf.int32)
        label_length += len(batch_y)

        probs = model.call(batch_x)
        top1_acc = model.top1_accuracy(probs, batch_y)
        top5_acc = model.top5_accuracy(probs, batch_y)
        print(top1_acc)
        #print(top1_num_corr)
        top1_num_corr += top1_acc * len(batch_y)
        top5_num_corr += top5_acc * len(batch_y)
    top1_acc = top1_num_corr / label_length
    top5_acc = top5_num_corr / label_length
    return top1_acc, top5_acc

def main():
    train_data, test_data, num_classes = get_data('Imagenet32_train', 'Imagenet32_val')
    model = SqueezeNet(num_classes)
    train_data = train_data.batch(model.batch_size)
    test_data = test_data.batch(model.batch_size)
    for _ in range(5):
        train(model, train_data)
        top1_acc, top5_acc = test(model, test_data)
        print("Top 1 Accuracy: ", top1_acc)
        print("Top 5 Accuracy: ", top5_acc)
    return


if __name__=='__main__':
    main()
