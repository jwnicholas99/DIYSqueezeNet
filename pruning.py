import os
import argparse
import zipfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from model import SqueezeNet

def load_model(args):
    model = tf.keras.models.load_model(args['filepath'])
    squeezenet = model.get_layer(index=2)
    squeezenet.save_weights("saved_weights")
    model = SqueezeNet(6)
    model.load_weights("saved_weights")
    model.wrap_layer_pruning()
    model = tf.keras.Sequential([
        model
        ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    model.build((None, 150, 150, 3))
    model.summary()
    model = model.strip_model_prune()
    """
    model = tf.keras.models.load_model(args['filepath'])
    squeeze = model.get_layer(index=2)
    for layer in squeeze.layers:
        print(layer)
        tfmot.sparsity.keras.prune_low_magnitude(layer)
    
    def apply_pruning_to_layers(layer):
        if layer == squeeze:
            return tf.keras.models.clone_model(layer, clone_function=apply_pruning_to_layers)
        
        elif not isinstance(layer, tf.keras.layers.experimental.preprocessing.RandomFlip) \
        and not isinstance(layer, tf.keras.layers.experimental.preprocessing.RandomRotation):
            return tfmot.sparsity.keras.prune_low_magnitude(layer)
        return layer
    
    model = tf.keras.models.clone_model(model, clone_function=apply_pruning_to_layers)
    """
    return model

def prune_model(model):
    # pruning hyperparams
    num_epochs = 5
    train_data_dir = "intel6/seg_train"
    test_data_dir = "intel6/seg_test"
    IMAGE_DIM = 150
    BATCH_SIZE = 32

    # prune model
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

    # Training:
    my_callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
    ]
    model.fit(train_data, epochs=num_epochs, validation_data=test_data, callbacks=my_callbacks)

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
    args = vars(parser.parse_args())

    if not os.path.exists(args['filepath']):
        print("File not found")

    model = load_model(args)
    model = prune_model(model)
    export_model(model, args)
