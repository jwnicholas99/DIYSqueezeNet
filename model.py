import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

from fire_mod import FireLayer

class SqueezeNet(tf.keras.Model):
    def __init__(self, num_classes):
        """
        This is the function for initializing the hyperparameters and structure
        of the SqueezeNet model.
        :param num_classes: number of classes for images
        """
        super(SqueezeNet, self).__init__()

        # MOVING HYPERPARAMETERS INTO RUN.PY:
        # self.batch_size = 32
        # self.num_classes = num_classes
        # start_learning_rate = 0.04
        # end_learning_rate = 0.00001
        #num_steps = 10000
        #self.learning_rate_fn = keras.optimizers.schedules.PolynomialDecay(start_learning_rate,
                                                                      #10000,
                                                                      #end_learning_rate,
                                                                      #power=1.0)
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_fn)

        # Initialize learnable parameters
        self.conv1 = keras.layers.Convolution2D(64, (3,3), activation='relu',
                                                kernel_initializer='glorot_uniform',
                                                strides=(2,2), padding='same')
        self.maxpool1 = keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2))

        self.fire2 = FireLayer(16, 64)
        self.fire3 = FireLayer(16, 64)
        self.maxpool4 = keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2))

        self.fire5 = FireLayer(32, 128)
        self.fire6 = FireLayer(32, 128)
        self.maxpool7 = keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2))

        self.fire8 = FireLayer(48, 256)
        self.fire9 = FireLayer(48, 256)
        self.fire10 = FireLayer(64, 256)
        self.fire11 = FireLayer(64, 256)
        self.dropout12 = keras.layers.Dropout(0.5)

        self.conv13 = keras.layers.Convolution2D(num_classes, (1,1), activation='relu',
                                                 kernel_initializer='glorot_uniform',
                                                 padding='valid')
        self.global_avgpool14 = keras.layers.GlobalAveragePooling2D()
        self.softmax = keras.layers.Activation('softmax')

    def call(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        fire2 = self.fire2(maxpool1)
        fire3 = self.fire3(fire2)
        maxpool4 = self.maxpool4(fire3)

        fire5 = self.fire5(maxpool4)
        fire6 = self.fire6(fire5)
        maxpool7 = self.maxpool7(fire6)

        fire8 = self.fire8(maxpool7)
        fire9 = self.fire9(fire8)
        fire10 = self.fire10(fire9)
        fire11 = self.fire11(fire10)
        dropout12 = self.dropout12(fire11)
        conv13 = self.conv13(dropout12)
        global_avgpool14 = self.global_avgpool14(conv13)

        probs = self.softmax(global_avgpool14)
        return probs
    
    def wrap_layer_pruning(self):
        self.conv1 = tfmot.sparsity.keras.prune_low_magnitude(self.conv1)

        self.fire2.wrap_layer_pruning()
        self.fire3.wrap_layer_pruning()

        self.fire5.wrap_layer_pruning()
        self.fire6.wrap_layer_pruning()

        self.fire8.wrap_layer_pruning()
        self.fire9.wrap_layer_pruning()
        self.fire10.wrap_layer_pruning()
        self.fire11.wrap_layer_pruning()

        self.conv13 = tfmot.sparsity.keras.prune_low_magnitude(self.conv13)
        
    def strip_pruning_wrapping(self, layer):
        if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
            # The _batch_input_shape attribute in the first layer makes a Sequential
            # model to be built. This makes sure that when we remove the wrapper from
            # the first layer the model's built state preserves.
            if not hasattr(layer.layer, '_batch_input_shape') and hasattr(
                    layer, '_batch_input_shape'):
                layer.layer._batch_input_shape = layer._batch_input_shape
            return layer.layer
        else:
            return layer
        
    def strip_model_prune(self):
        self.conv1 = self.strip_pruning_wrapping(self.conv1)

        self.fire2.strip_model_prune()
        self.fire3.strip_model_prune()

        self.fire5.strip_model_prune()
        self.fire6.strip_model_prune()

        self.fire8.strip_model_prune()
        self.fire9.strip_model_prune()
        self.fire10.strip_model_prune()
        self.fire11.strip_model_prune()

        self.conv13 = self.strip_pruning_wrapping(self.conv13)


    # def loss(self, probs, labels):
    #     return tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(labels, probs))

    # def top1_accuracy(self, probs, labels):
    #     return keras.metrics.sparse_categorical_accuracy(labels, probs)

    # def top5_accuracy(self, probs, labels):
    #     return keras.metrics.sparse_top_k_categorical_accuracy(labels, probs, k=5)
