import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

class FireLayer(tf.keras.Model):
    def __init__(self, squeeze_filters, expand_filters):
        """
        :param squeeze_filters: num of filters after squeezing
        :param expand_filters: num of filter for each expansion

        Note that there are two types of expansion - 1x1 and 3x3 kernel,
        so we will concatenate the outputs of both types of expansion 
        depth-wise. This means that the number of output filters for eac
        fire module is expand_filters * 2. Inherits tf.keras.Model because
        
        """
        super(FireLayer, self).__init__()

        # Initialize learnable params
        self.squeeze = keras.layers.Convolution2D(squeeze_filters,
                                                  (1,1),
                                                  activation='relu',
                                                  kernel_initializer='glorot_uniform',
                                                  padding='same')

        self.expand_1x1 = keras.layers.Convolution2D(expand_filters,
                                                    (1,1),
                                                    activation='relu',
                                                    kernel_initializer='glorot_uniform',
                                                    padding='same')

        self.expand_3x3 = keras.layers.Convolution2D(expand_filters,
                                                    (3,3),
                                                    activation='relu',
                                                    kernel_initializer='glorot_uniform',
                                                    padding='same')

        self.concat = keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        """
        :param inputs: FireLayer inputs

        Squeeze inputs, then return concatenated outputs of the two expand
        layers.        

        """
        squeeze = self.squeeze(inputs)
        expand_1x1 = self.expand_1x1(squeeze)
        expand_3x3 = self.expand_3x3(squeeze)
        concat = self.concat([expand_1x1, expand_3x3])

        return concat
    
    def wrap_layer_pruning(self):
        """
        Manual pruning of individual sublayers of the FireLayer as a helper
        function to be called by the model call.
        
        """
        self.squeeze = tfmot.sparsity.keras.prune_low_magnitude(self.squeeze)
        self.expand_1x1 = tfmot.sparsity.keras.prune_low_magnitude(self.expand_1x1)
        self.expand_3x3 = tfmot.sparsity.keras.prune_low_magnitude(self.expand_3x3)
        
    def strip_pruning_wrapping(self, layer):
        """
        Helper function to remove the pruning wrapper to extract the pruned 
        layer to replace the unpruned layer.

        """
        if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
            if not hasattr(layer.layer, '_batch_input_shape') and hasattr(
                    layer, '_batch_input_shape'):
                layer.layer._batch_input_shape = self.squeeze._batch_input_shape
            return layer.layer
        else:
            print('layer not wrapped')
            return layer
    
    def strip_model_prune(self):
        """
        Strips layers in FireLayer for each layer in FireLayer using the above
        helper function.

        """
        self.squeeze = self.strip_pruning_wrapping(self.squeeze)
        self.expand_1x1 = self.strip_pruning_wrapping(self.expand_1x1)
        self.expand_3x3 = self.strip_pruning_wrapping(self.expand_3x3)
  
