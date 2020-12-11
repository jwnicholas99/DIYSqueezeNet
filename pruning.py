import os
import tempfile
import argparse
import zipfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from model import SqueezeNet

def load_model(args):
    """
    :param args: Takes in filepath of network weights, filepath to save network
    weights, filepath to save pruned weights,filepath to save to pruned network,
    filepath to save zip model.

    return model: Returns properly loaded model
    
    Recreates model using the loaded model from the filepath for saved weights.
    Build and compile the model and then return it as a SqueezeNet type model.
    

    """
    model = tf.keras.models.load_model(args['filepath'])
    squeezenet = model.get_layer(index=2)
    squeezenet.save_weights("saved_weights")
    model = SqueezeNet(6)
    model.load_weights("saved_weights")
    
    model.wrap_layer_pruning()
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
        model
        ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    model.build((None, 150, 150, 3))

    return model

def get_gzipped_model_size(model):
    _, pruned_keras_file = tempfile.mkstemp('.h5')
    #tf.keras.models.save_model(model, pruned_keras_file, include_optimizer=False)
    model.save_weights(pruned_keras_file)
    
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(pruned_keras_file)
    return os.path.getsize(zipped_file)

def prune_model(model, train_data, test_data=None):
    """
    :param model: loaded SqueezeNet model
    :param train_data: tf.data.Dataset train data
    :param test_data: tf.data.Dataset test data

    :return model: Pruned weights SqueezeNet model
    
    Takes in the SqueezeNet model, looks at the third layer which is the model 
    layer, and then uses the class method to wrap the layers for pruning. We 
    then unwrap the model and rebuild it with its previous preprocessing layers
    and then return it.

    """
    # pruning hyperparams
    num_epochs = 10
    BATCH_SIZE = 64

    # Get original model size
    og_size = get_gzipped_model_size(model.get_layer(index=2))

    # Create prune model
    squeezenet = model.get_layer(index=2)
    #squeezenet.save_weights("saved_weights")
    #model = SqueezeNet(6)
    #model.load_weights("saved_weights")
    
    squeezenet.wrap_layer_pruning()
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
        model
        ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    model.build((None, 150, 150, 3))

    # Training:
    my_callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
    ]
    if test_data is None:
        model.fit(train_data, epochs=num_epochs, validation_split=0.1, callbacks=my_callbacks)
    else:
        model.fit(train_data, epochs=num_epochs, validation_data=test_data, callbacks=my_callbacks)

    # Compare model size
    print("[*] Size of gzipped baseline model: {} bytes".format(og_size))
    print("[*] Size of gzipped pruned model: {} bytes".format(get_gzipped_model_size(model.get_layer(index=2))))
    squeezenet.strip_model_prune()

    return model

def export_model(model, args):
    """
    :param model: SqueezeNet model
    :param args: Takes in filepath of network weights, filepath to save network
    weights, filepath to save pruned weights,filepath to save to pruned network,
    filepath to save zip model.
    
    Saves model weights and then zips it to compress using the filepaths
    inputted.

"""
    squeezenet = model.get_layer(index=2)
    squeezenet.strip_model_prune()
    squeezenet.summary()
    tf.keras.models.save_model(model, args['outpath'], include_optimizer=False)

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
    model = prune_model(model, train_data, test_data)
    export_model(model, args)
