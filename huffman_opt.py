# Note: Don't name the file huffman.py. It will break the import.
# Alternative option with dahuffman library.

import tensorflow as tf
import numpy as np

# Option 1: huffman
# Results:
#   - Size: 500 * 500 , Time: ~ 10-15 seconds
#   - Size: 5000 * 5000 , Time: ~ 7.5 minutes
def huffman_opt1(dim_size):
    # pip install huffman
    import huffman
    import collections

    # Get some sample weights
    test_weights = tf.Variable(tf.random.truncated_normal([dim_size, dim_size], stddev=.1, dtype=tf.float32))

    # Preprocessing weights
    flattened_array = test_weights.numpy().flatten()
    huffman_input = collections.Counter(flattened_array).items()

    # Dictionary - { 'value' : bit-representation }
    result = huffman.codebook(huffman_input)

# Option 2: huffman
# Results:
#   - Size: 500 * 500 , Time: ~ 8-10 seconds
#   - Size: 5000 * 5000 , Time: ~ 23 minutes
def huffman_opt2():
    # pip install dahuffman
    from dahuffman import HuffmanCodec

    # Get some sample weights
    test_weights = tf.Variable(tf.random.truncated_normal([dim_size, dim_size], stddev=.1, dtype=tf.float32))

    # Preprocessing weights
    flattened_array = test_weights.numpy().flatten()

    # Get dahuffman encoded object
    codec = HuffmanCodec.from_data(flattened_array)

    # Get code table dictionary - { 'symbol' : (num_bits, value) }
    results = codec.get_code_table()

    # Print properties of object
    # codec.print_code_table()

huffman_opt1(500)
huffman_opt2(500)

# Sound to mark end of program
print('\a')
