# Remove most of the annoying wordy tensorflow prints
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

# pip install -q tensorflow-model-optimization
import tensorflow_model_optimization as tfmot

# Makes model quantization aware and retrains. Optional call before quantize.
# Inputs:
#   model - Pre-trained model to quantize.
#   optimizer, loss, metrics - Features used to train model.
#   train_data, train_labels - Retraining data. Does not need to be new.
#   subset_size - Size of subset of data to retrain on.
#   batch_size, epochs, validation_split - Features used to retrain model.
def pre_quantize(model,
                 optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'],
                 train_data=None, num_shards=5,
                 batch_size=500, epochs=1, validation_split=0.1):
    """
    :param model: Loaded SqueezeNet model
    :param optimizer: User option for optimizer function, default is adam
    :param loss: User option for loss function
    :param metrics: Array of metrics used for analysis of performance for model
    :param train_data: Train data for trianing the quantization aware model
    :param num_shards: Number of shards to split the training data on for train
    :param batch_size: User choice batch size
    :param epochs: User choice for number of epochs
    :param validation_split: Give some validation from the training data
    Uses the Keras quantize_model function on the optimizable layer of the 
    model, and then compiles it. Retrains the model on a subset of training
    data now that the model is quantization aware.
    
    """
    # Make model quantization aware
    #model.get_layer(index=2).quantize_submodels()
    quantize_model = tfmot.quantization.keras.quantize_model
    quantize_model(model.get_layer(index=2))
    q_aware_model = quantize_model(model)
    q_aware_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Retraining increases accuracy, but does not affect model size
    if train_data is not None:
        train_shard = train_data.shard(num_shards)
        q_aware_model.fit(train_shard,
                          batch_size=batch_size, epochs=epochs,
                          validation_split=validation_split)

    return q_aware_model

# Quantizes model
def quantize(model):
    """
    :param model: Loaded SqueezeNet model.
    Converts the model into a TFLite model for compression purposes.
    
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()

    return quantized_tflite_model

# Save quantized model to file
def save_model(quantized_tflite_model, file_path):
    """
    :param quantized_tflite_model: TFLite model type
    :param file_path: String file path to save the model at
    Saves model at inputted file_path
    
    """
    with open(file_path, 'wb') as f:
        f.write(quantized_tflite_model)

# Load quantized model as interpreter
def load_model(file_path):
    """
    :param file_path: File path where model is located
    Loads the model using a tf.lite.Interpreter

    """
    
    interpreter = tf.lite.Interpreter(model_path=file_path)
    interpreter.allocate_tensors()
    return interpreter

# Display model sizes before and after quantization
def display_diff(model, quantized_tflite_model):
    """
    :param model: SqueezeNet model
    :param quantized_tflite_mode: TFLite model that has been quantized
    Converts both models and stores them, and then uses an os function to
    determine the size of the files in storage.
    
    """
    import tempfile
    import os

    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_tflite_model = float_converter.convert()

    _, float_file = tempfile.mkstemp('.tflite')
    _, quant_file = tempfile.mkstemp('.tflite')

    with open(quant_file, 'wb') as f:
      f.write(quantized_tflite_model)

    with open(float_file, 'wb') as f:
      f.write(float_tflite_model)

    print("-" * 30)
    print("Float model in Mb:", os.path.getsize(float_file) / float(2**20))
    print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2**20))
    print("-" * 30)

def demo(save):
    """
    :param save: Save filepath for quantized model
    Demo function to display the usage of the quantized model on a simpler 
    model like MNIST. It can be run without GCP and displays how much compression
    in terms of storage quantization does for a model.

    """
    
    # Sample Model
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=1, validation_split=0.1)

    # Makes model quantization aware and retrains (OPTIONAL)
    q_aware_model = pre_quantize(model,
             optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'],
             train_data=train_images,
             train_labels=train_labels,
             subset_size=1000,
             batch_size=500,
             epochs=1,
             validation_split=0.1)

    # Quantizes model
    quantized_tflite_model = quantize(q_aware_model)

    # Display model sizes
    display_diff(model, quantized_tflite_model)

    # Save model
    if save:
        save_model(quantized_tflite_model, 'model.tflite')

def demo_interp():
    """
    Shows how the TfLite interpreter looks at the layers and shows which one is
    quantized.
    
    """
    # Load model
    interpreter = load_model('model.tflite')

    # See all the tensor layers
    for x in enumerate(interpreter.get_tensor_details()):
        print("[{}]: {}".format(x[0], x[1]), flush=True)

    # Most important tensors for MNIST model
    print(interpreter.get_tensor(0)) # Input Layer
    print(interpreter.get_tensor(6)) # Quantized Layer

# demo(True)
# demo_interp()
