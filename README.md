<h1 align="center">
  <br>
  🤏 DIYSqueezeNet
  <br>
</h1>

<h4 align="center">A small but powerful network.</h4>
<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#usage">Usage</a> •
  <a href="#usage">Details</a> •
  <a href="#usage">Known Bugs</a>
</p>

<p align="center">
<img src="ESP32_Demo_Classifier/assets/demo_vid.gif" width="375"/>
</p>


## Key Features

* SqueezeNet model with Fire layers
* Prunes model using tensorflow-model-optimization (tfmot)
* Quantizes model using tfmot
* Runs Huffman encoding using dahuffman Python module
* Deploys model to run on ESP32

## Usage

```
$ python run.py
``` 
This will run the model training and compression pipeline. To modify the hyperparameters, note that there are the following variables you can set in run.py:
```
IMAGE_DIM = 150
NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_EPOCHS = 10
IS_CALTECH = 0
``` 

## Details

### Overview
This project contains a successfully implemented base model of SqueezeNet 1.1 
using the Intel Image Classification dataset and Caltech 257 dataset as 
implemented preprocessing functions. It also contains functions for three 
added compression techniques: pruning, quantization, and huffman coding in 
three separate files that are manually made for our SqueezeNet implementation 
in many cases.

### Run
The run.py file instantiates a model based on set hyperparameters, which 
includes a choice for datasets from Intel to Caltech. Using a sequential
Keras model, it instantiates a SqueezeNet model and trains it for the user 
set number of epochs and saves checkpoints as well. After, it runs our three 
optimizations and ultimately, the compressed model can be saved locally.

### Squeezenet
The baseline model implementation is contained in the two files model.py, 
fire_mod.py. model.py contains the SqueezeNet class which has the general 
architecture of the model, composed of convolution, FireLayers, maxpooling,
and average pooling. It has the call function for training the model as well as
prune utility functions to wrap and unwrap the individual layers in the model.
The FireLayer is a tf.keras.Model that represents the FireLayer described in 
the paper, that has a squeeze layer and returns the output of the concatenation
of two expand layers with different sized kernels. We need to use tf.keras.Model
instead of as a layer because Keras can't examine layers within layers, so 
our implementation requires this sort of inheritance architecture. FireLayer 
also has pruning helper methods to prune its layers within, and is called by
the pruning functions in the SqueezeNet class.

### Optimizations
The three other files contain our compression implementations. pruning.py 
contains two helper methods, load_model and export_model to load in SqueezeNet
weights that have been saved into a SqueezeNet model, and exporting a model into
a zipped file locally. We then have a pruning method that prunes a SqueezeNet 
and creates a new SqueezeNet model with pruned weights.

quantization.py contains a pre_quantize method to make the model quantization 
aware, and retrain it, as well as a save_model and load_model utility function.
A display_diff method is provided to compare the size of an original SqueezeNet
model and our retrained quantized SqueezeNet model. There are also demo methods
to see more quickly on an MNIST model to see if it works.

huffman_opt has three methods using the da_huffman library for encoding and
decoding models. It provides an encoding size which doesn't include the codec 
size but shows how much compression can be used on this level and
this representation of the weights.

## Known Bugs
Huffman encoding doesn't describe the size of the codec as of yet and we can 
find the value manually but it doesn't currently reflect that in the code.
Also, full quantization doesn't work with our inheritance architecture of submodels
of FireLayers in models so we have an implementation of it that does represent 
the compression that it provides, but it may affect the accuracy of the model
after it has been quantized. Currently, our model does run on Caltech 257 but 
the dataset is unbalanced and has quite low training examples so training has been 
really unstable and not reliable enough to show as a final result. Thus, we are 
currently not supporting Caltech 257 and our results are based on Intel Image 
Classification.
