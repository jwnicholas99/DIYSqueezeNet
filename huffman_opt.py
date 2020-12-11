# Note: Don't name the file huffman.py. It will break the import.

import tensorflow as tf
import numpy as np

# pip install dahuffman
from dahuffman import HuffmanCodec

# Performance on tensors:
#   - Size: 500 * 500 , Time: ~ 8-10 seconds
#   - Size: 5000 * 5000 , Time: ~ 23 minutes

# Input: Tensor
# Output: Encoded tensor, Codec
def encode(weights):
    flattened_array = weights.flatten()
    codec = HuffmanCodec.from_data(flattened_array)
    return codec.encode(flattened_array), codec

# Input: Encoded tensor, Codec, Optional: original shape of tensor
# Output: Flattened array
def decode(code, codec, shape=None):
    assert(isinstance(code, bytes) and isinstance(codec, HuffmanCodec))
    if shape is not None:
        return np.reshape(codec.decode(code), shape)
    else:
        return codec.decode(code)
    
def compare_encoding_size(code, weights):
    weights_size = 32 
    for dim in weights.shape:
        weights_size *= dim 
    code_size = len(code) 
    print("Weights tensor: {} bits. Code: {} bits.".format(weights_size, code_size),
          flush=True) 

# Test Demo
def demo():
    print("-" * 30)

    # Test weights
    test_weights = tf.Variable(tf.random.truncated_normal([500, 500],
                                                          stddev=.1,
                                                          dtype=tf.float32))
    print("Testing on tensor of size {} x {}.".format(test_weights.shape[0],
                                                      test_weights.shape[1]),
                                                      flush=True)

    # Syntax for encoding
    print("Encoding...", flush=True)
    code, codec = encode(test_weights.numpy())
    print("Successfully encoded tensor to {} bytes.".format(len(code)), flush=True)

    # Syntax for decoding
    print("Decoding...", flush=True)
    output_weights = decode(code, codec, shape=test_weights.shape)
    print("Successfully decoded tensor into tensor of shape {} x {}.".format(
        output_weights.shape[0], output_weights.shape[1]), flush=True)

    # Making sure the encode and decode did not alter tensor
    equality_check = np.all(np.equal(test_weights, output_weights))
    print("Result of equality check: {}".format(equality_check), flush=True)

    print("-" * 30)

# demo()
