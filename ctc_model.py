# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 00:27:59 2019

@author: amade
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def leaky_relu(features, alpha=0.2, name=None):
  with ops.name_scope(name, "LeakyRelu", [features, alpha]):
    features = ops.convert_to_tensor(features, name="features")
    alpha = ops.convert_to_tensor(alpha, name="alpha")
    return math_ops.maximum(alpha * features, features)

def default_model_params(img_height, vocabulary_size):
    params = dict()
    params['img_height'] = img_height
    params['img_width'] = None
    params['batch_size'] = 8
    params['img_channels'] = 1
    params['conv_blocks'] = 4
    params['conv_filter_n'] = [32, 64, 128, 256]
    params['conv_filter_size'] = [ [3,3], [3,3], [3,3], [3,3] ]
    params['conv_pooling_size'] = [ [2,2], [2,2], [2,2], [2,2] ]
    params['rnn_units'] = 256
    params['rnn_layers'] = 2
    params['vocabulary_size'] = vocabulary_size
    return params


def ctc_crnn(params):
    # TODO Assert parameters

    input = tf.placeholder(shape=(None,
                                   params['img_height'],
                                   params['img_width'],
                                   params['img_channels']),  # [batch, height, width, channels]
                            dtype=tf.float32,
                            name='model_input')

    input_shape = tf.shape(input)

    width_reduction = 1
    height_reduction = 1


    # Convolutional blocks
    x = input
    for i in range(params['conv_blocks']):

        x = tf.layers.conv2d(
            inputs=x,
            filters=params['conv_filter_n'][i],
            kernel_size=params['conv_filter_size'][i],
            padding="same",
            activation=None)

        x = tf.layers.batch_normalization(x)
        x = leaky_relu(x)

        x = tf.layers.max_pooling2d(inputs=x,
                                    pool_size=params['conv_pooling_size'][i],
                                    strides=params['conv_pooling_size'][i])

        width_reduction = width_reduction * params['conv_pooling_size'][i][1]
        height_reduction = height_reduction * params['conv_pooling_size'][i][0]


    # Prepare output of conv block for recurrent blocks
    features = tf.transpose(x, perm=[2, 0, 3, 1])  # -> [width, batch, height, channels] (time_major=True)
    feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / height_reduction)
    feature_width = input_shape[2] / width_reduction
    features = tf.reshape(features, tf.stack([tf.cast(feature_width,'int32'), input_shape[0], tf.cast(feature_dim,'int32')]))  # -> [width, batch, features]

    tf.constant(params['img_height'],name='input_height')
    tf.constant(width_reduction,name='width_reduction')

    # Recurrent block
    rnn_keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
    rnn_hidden_units = params['rnn_units']
    rnn_hidden_layers = params['rnn_layers']

    rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        tf.contrib.rnn.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(rnn_hidden_units), input_keep_prob=rnn_keep_prob)
             for _ in range(rnn_hidden_layers)]),
        tf.contrib.rnn.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(rnn_hidden_units), input_keep_prob=rnn_keep_prob)
             for _ in range(rnn_hidden_layers)]),
        features,
        dtype=tf.float32,
        time_major=True,
    )

    rnn_outputs = tf.concat(rnn_outputs, 2)

    logits = tf.contrib.layers.fully_connected(
        rnn_outputs,
        params['vocabulary_size'] + 1,  # BLANK
        activation_fn=None,
    )
    
    tf.add_to_collection("logits",logits) # for restoring purposes

    # CTC Loss computation
    seq_len = tf.placeholder(tf.int32, [None], name='seq_lengths')
    targets = tf.sparse_placeholder(dtype=tf.int32, name='target')
    ctc_loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len, time_major=True)
    loss = tf.reduce_mean(ctc_loss)

    # CTC decoding
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    # decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits,seq_len,beam_width=50,top_paths=1,merge_repeated=True)

    return input, seq_len, targets, decoded, loss, rnn_keep_prob

def ctc_crnn_custom(params):
    input = tf.placeholder(shape=(None,
                                   params['img_height'],
                                   params['img_width'],
                                   params['img_channels']),  # [batch, height, width, channels]
                            dtype=tf.float32,
                            name='model_input')

    input_shape = tf.shape(input)

    width_reduction = 1
    height_reduction = 1

    # Convolutional blocks
    x = input
    x1 = tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same")(x)
    x2 = tf.keras.layers.LeakyReLU()(x1)
    x2 = tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same")(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same")(x2)
    
    x = tf.keras.layers.Add()([x1, x2])
    
    for i in range(2):
        x1 = tf.keras.layers.BatchNormalization()(x)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same", use_bias = False)(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same", use_bias = False)(x1)
        x = tf.keras.layers.Add()([x, x1])        
    
    x1 = tf.keras.layers.BatchNormalization()(x)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.SeparableConv2D(64, (3, 3), strides = 2, padding="same", use_bias = False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same", use_bias = False)(x1)
    x2 = tf.keras.layers.SeparableConv2D(64, (1, 1), padding="same", use_bias = False)(x)
    x2 = tf.keras.layers.LeakyReLU()(x2)    
    x = tf.keras.layers.Add()([x1, x2])
    width_reduction = width_reduction * 2
    height_reduction = height_reduction * 2

    for i in range(2):
        x1 = tf.keras.layers.BatchNormalization()(x)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same", use_bias = False)(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same", use_bias = False)(x1)
        x = tf.keras.layers.Add()([x, x1])               
   
    x1 = tf.keras.layers.BatchNormalization()(x)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.SeparableConv2D(128, (3, 3), strides = 2, padding="same", use_bias = False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.SeparableConv2D(128, (3, 3), padding="same", use_bias = False)(x1)
    x2 = tf.keras.layers.SeparableConv2D(128, (1, 1), strides = 2, padding="same")(x)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x = tf.keras.layers.Add()([x1, x2])
    width_reduction = width_reduction * 2
    height_reduction = height_reduction * 2

    for i in range(2):
        x1 = tf.keras.layers.BatchNormalization()(x)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(128, (3, 3), padding="same", use_bias = False)(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(128, (3, 3), padding="same", use_bias = False)(x1)
        x = tf.keras.layers.Add()([x, x1])               
    
    x1 = tf.keras.layers.BatchNormalization()(x)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.SeparableConv2D(256, (3, 3), strides = 2, padding="same", use_bias = False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.SeparableConv2D(256, (3, 3), padding="same", use_bias = False)(x1)
    x2 = tf.keras.layers.SeparableConv2D(256, (1, 1), strides = 2,  padding="same")(x)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x = tf.keras.layers.Add()([x1, x2])
    width_reduction = width_reduction * 2
    height_reduction = height_reduction * 2

    for i in range(2):
        x1 = tf.keras.layers.BatchNormalization()(x)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(256, (3, 3), padding="same", use_bias = False)(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(256, (3, 3), padding="same", use_bias = False)(x1)
        x = tf.keras.layers.Add()([x, x1])               
    
    x = tf.keras.layers.SeparableConv2D(256, (1, 1), strides = 2,  padding="same")(x)
    x = tf.keras.layers.LeakyReLU()(x)
    width_reduction = width_reduction * 2
    height_reduction = height_reduction * 2

    # Prepare output of conv block for recurrent blocks
    #features = tf.transpose(x, perm=[0, 2, 1, 3])  # -> [width, batch, height, channels] (time_major=True)
    #features = x
    feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / height_reduction)
    feature_width = input_shape[2] / width_reduction
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(rate = 0.5)(x)
    features = tf.keras.layers.Dense(input_shape[0] * tf.cast(feature_width,'int32') * tf.cast(feature_dim,'int32'))(x)
    features = tf.reshape(features, tf.stack([input_shape[0], tf.cast(feature_width,'int32'), tf.cast(feature_dim,'int32')])) # -> [batch, width, features]

    tf.constant(params['img_height'],name='input_height')
    tf.constant(width_reduction,name='width_reduction')

    # Recurrent block
    rnn_keep_prob = tf.keras.backend.placeholder(dtype=tf.float32, name="keep_prob")
    rnn_hidden_units = params['rnn_units']
    #rnn_hidden_layers = params['rnn_layers']
    
    x1 = tf.keras.layers.GRU(rnn_hidden_units, return_sequences = True, activation = 'relu',
                             kernel_initializer = 'he_normal', dropout = 0.5)(features)
    x2 = tf.keras.layers.GRU(rnn_hidden_units, return_sequences = True, go_backwards = True, 
                             activation = 'relu', kernel_initializer = 'he_normal', dropout = 0.5)(features)
    x = tf.keras.layers.Add()([x1, x2])
    
    x1 = tf.keras.layers.GRU(rnn_hidden_units, return_sequences = True, activation = 'relu',
                             kernel_initializer = 'he_normal', dropout = 0.5)(x)
    x2 = tf.keras.layers.GRU(rnn_hidden_units, return_sequences = True, go_backwards = True, 
                             activation = 'relu', kernel_initializer = 'he_normal', dropout = 0.5)(x)

    rnn_outputs = tf.keras.layers.Concatenate()([x1, x2])
    logits = tf.keras.layers.Dense(params['vocabulary_size'] + 1)(rnn_outputs)    
    tf.add_to_collection("logits",logits) # for restoring purposes

    # CTC Loss computation
    seq_len = tf.placeholder(tf.int32, [None], name='seq_lengths')
    targets = tf.sparse_placeholder(dtype=tf.int32, name='target')
    ctc_loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len, time_major=False)
    loss = tf.reduce_mean(ctc_loss)

    # CTC decoding
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    # decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits,seq_len,beam_width=50,top_paths=1,merge_repeated=True)

    return input, seq_len, targets, decoded, loss, rnn_keep_prob

def ctc_ctcn_custom(params):
    input = tf.placeholder(shape=(None,
                                   params['img_height'],
                                   params['img_width'],
                                   params['img_channels']),  # [batch, height, width, channels]
                            dtype=tf.float32,
                            name='model_input')

    input_shape = tf.shape(input)

    width_reduction = 1
    height_reduction = 1

    # Convolutional blocks
    x = input
    x1 = tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same")(x)
    x2 = tf.keras.layers.LeakyReLU()(x1)
    x2 = tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same")(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same")(x2)
    
    x = tf.keras.layers.Add()([x1, x2])
    
    for i in range(2):
        x1 = tf.keras.layers.BatchNormalization()(x)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same", use_bias = False)(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same", use_bias = False)(x1)
        x = tf.keras.layers.Add()([x, x1])        
           
    x1 = tf.keras.layers.BatchNormalization()(x)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.SeparableConv2D(64, (3, 3), strides = 2, padding="same", use_bias = False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same", use_bias = False)(x1)
    x2 = tf.keras.layers.SeparableConv2D(64, (1, 1), strides = 2, padding="same")(x)
    x2 = tf.keras.layers.LeakyReLU()(x2)    
    x = tf.keras.layers.Add()([x1, x2])
    width_reduction = width_reduction * 2
    height_reduction = height_reduction * 2

    for i in range(2):
        x1 = tf.keras.layers.BatchNormalization()(x)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same", use_bias = False)(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same", use_bias = False)(x1)
        x = tf.keras.layers.Add()([x, x1])               
   
    x1 = tf.keras.layers.BatchNormalization()(x)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.SeparableConv2D(128, (3, 3), strides = 2, padding="same", use_bias = False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.SeparableConv2D(128, (3, 3), padding="same", use_bias = False)(x1)
    x2 = tf.keras.layers.SeparableConv2D(128, (1, 1), strides = 2, padding="same")(x)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x = tf.keras.layers.Add()([x1, x2])
    width_reduction = width_reduction * 2
    height_reduction = height_reduction * 2

    for i in range(2):
        x1 = tf.keras.layers.BatchNormalization()(x)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(128, (3, 3), padding="same", use_bias = False)(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(128, (3, 3), padding="same", use_bias = False)(x1)
        x = tf.keras.layers.Add()([x, x1])               
    
    x1 = tf.keras.layers.BatchNormalization()(x)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.SeparableConv2D(256, (3, 3), strides = 2, padding="same", use_bias = False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.SeparableConv2D(256, (3, 3), padding="same", use_bias = False)(x1)
    x2 = tf.keras.layers.SeparableConv2D(256, (1, 1), strides = 2,  padding="same")(x)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x = tf.keras.layers.Add()([x1, x2])
    width_reduction = width_reduction * 2
    height_reduction = height_reduction * 2

    for i in range(2):
        x1 = tf.keras.layers.BatchNormalization()(x)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(256, (3, 3), padding="same", use_bias = False)(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.SeparableConv2D(256, (3, 3), padding="same", use_bias = False)(x1)
        x = tf.keras.layers.Add()([x, x1])               
    
    x = tf.keras.layers.SeparableConv2D(256, (1, 1), strides = 2,  padding="valid")(x)
    x = tf.keras.layers.LeakyReLU()(x)
    width_reduction = width_reduction * 2
    height_reduction = height_reduction * 2

    # Prepare output of conv block for recurrent blocks
    #features = tf.transpose(x, perm=[0, 2, 1, 3])  # -> [width, batch, height, channels] (time_major=True)
    #features = x
    feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / height_reduction) #2048
    feature_width = input_shape[2] / width_reduction
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(rate = 0.5)(x)
    features = tf.keras.layers.Dense(input_shape[0] * tf.cast(feature_width,'int32') * tf.cast(feature_dim,'int32'))(x)
    features = tf.reshape(features, tf.stack([input_shape[0], tf.cast(feature_width,'int32'), tf.cast(feature_dim,'int32')])) # -> [batch, width, features]

    tf.constant(params['img_height'],name='input_height')
    tf.constant(width_reduction,name='width_reduction')

    # Recurrent block
    rnn_keep_prob = tf.keras.backend.placeholder(dtype=tf.float32, name="keep_prob")
    rnn_hidden_units = params['rnn_units']
    #rnn_hidden_layers = params['rnn_layers']
    
    x1 = tf.keras.layers.GRU(rnn_hidden_units, return_sequences = True, activation = 'relu',
                             kernel_initializer = 'he_normal', dropout = 0.5)(features)
    x2 = tf.keras.layers.GRU(rnn_hidden_units, return_sequences = True, go_backwards = True, 
                             activation = 'relu', kernel_initializer = 'he_normal', dropout = 0.5)(features)
    x = tf.keras.layers.Add()([x1, x2])
    
    x1 = tf.keras.layers.GRU(rnn_hidden_units, return_sequences = True, activation = 'relu',
                             kernel_initializer = 'he_normal', dropout = 0.5)(x)
    x2 = tf.keras.layers.GRU(rnn_hidden_units, return_sequences = True, go_backwards = True, 
                             activation = 'relu', kernel_initializer = 'he_normal', dropout = 0.5)(x)

    rnn_outputs = tf.keras.layers.Concatenate()([x1, x2])
    logits = tf.keras.layers.Dense(params['vocabulary_size'] + 1)(rnn_outputs)    
    tf.add_to_collection("logits",logits) # for restoring purposes

    # CTC Loss computation
    seq_len = tf.placeholder(tf.int32, [None], name='seq_lengths')
    targets = tf.sparse_placeholder(dtype=tf.int32, name='target')
    ctc_loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len, time_major=False)
    loss = tf.reduce_mean(ctc_loss)

    # CTC decoding
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    # decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits,seq_len,beam_width=50,top_paths=1,merge_repeated=True)

    return input, seq_len, targets, decoded, loss, rnn_keep_prob