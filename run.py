import tensorflow as tf

from preprocess import get_data
from model import SqueezeNet

def train(model, inputs, labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. Images are 
    shuffled. Images should already have been randomnly transformed in get_data()
    :param model: model to run
    :param inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    '''
    indices = tf.random.shuffle(tf.range(len(labels)))
    inputs = tf.gather(inputs, indices)
    labels = tf.gather(labels, indices)

    for start in range(0, len(labels), model.batch_size):
        end = start + model.batch_size
        batch_x = inputs[start:end]
        batch_y = labels[start:end]

        with tf.GradientTape() as tape:
            probs = model.call(batch_x)
            loss = model.loss(probs, batch_y)
        print("Loss: ", loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return

def test(model, inputs, labels):
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

    for start in range(0, len(inputs), model.batch_size):
        end = start + model.batch_size
        batch_x = inputs[start:end]
        batch_y = inputs[start:end]

        probs = model.call(batch_x)
        top1_acc = model.top1_accuracy(probs, batch_y)
        top5_acc = model.top5_accuracy(probs, batch_y)
        top1_num_corr += top1_acc * len(batch_y)
        top5_num_corr += top5_acc * len(batch_y)
    top1_acc = top1_num_corr / len(labels)
    top5_acc = top5_num_corr / len(labels)
    return top1_acc, top5_acc

def main():
    train_inputs, train_labels, test_inputs, test_labels, num_classes = get_data()

    model = SqueezeNet(num_classes)

    for _ in range(10):
        train(model, train_inputs, train_labels)
        top1_acc, top5_acc = test(model, test_inputs, test_labels)
        print("Top 1 Accuracy: ", top1_acc)
        print("Top 5 Accuracy: ", top5_acc)
    return


if __name__=='__main__':
    main()
