import os
import argparse
import zipfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from model import SqueezeNet

def load_model(args):
    model = SqueezeNet(args['num_class'])
    model.load_weights(args['filepath'])
    return model

def prune_model(model):
    model = tfmot.sparsity.keras.prune_low_magnitude(model)
    train_inputs, train_labels, test_inputs, test_labels, num_classes = get_data()

    # pruning hyperparams
    num_epochs = 5
    step_callback = tfmot.sparsity.keras.UpdatePruningStep()
    step_callback.set_model(model)

    # prune model
    step_callback.on_train_begin()
    for _ in range(num_epochs):
        indices = tf.random.shuffle(tf.range(len(train_labels)))
        train_inputs = tf.gather(train_inputs, indices)
        train_labels = tf.gather(train_labels, indices)

        for start in range(0, len(train_labels), model.batch_size):
            step_callback.on_train_batch_begin()
            end = start + model.batch_size
            batch_x = train_inputs[start:end]
            batch_y = train_labels[start:end]

            with tf.gradienttape() as tape:
                probs = model.call(batch_x)
                loss = model.loss(probs, batch_y)
            print("loss: ", loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        step_callback.on_epoch_end()

    return model

def export_model(model, args):
    model_for_export = tfmot.sparsity.keras.strip_pruning(model)
    tf.keras.models.save_model(model_for_export, args['outpath'], include_optimizer=False)

    with zipfile.ZipFile(args['zippath'], 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(args['outpath'])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Utility for pruning networks")
    parser.add_argument('--filepath', type=str, help="filepath of network weights")
    parser.add_argument('--outpath', type=str, help="filepath to save pruned model")
    parser.add_argument('--zippath', type=str, help="filepath to zip pruned model")
    parser.add_argument('--num_class', type=int, help="number of classes")
    args = vars(parser.parse_args())

    if not os.path.exists(args['filepath']):
        print("File not found")
        return

    model = load_model(args)
    model = prune_model(model)
    export_model(model, args)
