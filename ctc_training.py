# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 00:21:17 2019

@author: amade
"""

import tensorflow as tf
from primus import CTC_PriMuS
import ctc_utils
import ctc_model
import argparse
from statistics import mean

import os

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.reset_default_graph()
sess = tf.InteractiveSession(config=config)

parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('-corpus', dest='corpus', type=str, required=True, help='Path to the corpus.')
parser.add_argument('-set',  dest='set', type=str, required=True, help='Path to the set file.')
parser.add_argument('-save_model', dest='save_model', type=str, required=True, help='Path to save the model.')
parser.add_argument('-vocabulary', dest='voc', type=str, required=True, help='Path to the vocabulary file.')
parser.add_argument('-semantic', dest='semantic', action="store_true", default=False)
parser.add_argument('-frequency', dest='frequency', type=int, default=1, help='Frequency of printing')
args = parser.parse_args()

# Load primus dataset

primus = CTC_PriMuS(args.corpus,args.set,args.voc, args.semantic, distortions = True, val_split = 0.1)

# Parameterization
img_height = 128
params = ctc_model.default_model_params(img_height, primus.vocabulary_size)
max_epochs = 100
dropout = 0.5

# Model
inputs, seq_len, targets, decoded, loss, rnn_keep_prob = ctc_model.ctc_crnn_custom(params)
train_opt = tf.train.AdamOptimizer().minimize(loss)

saver = tf.train.Saver(max_to_keep=None)
sess.run(tf.global_variables_initializer())

step_size = primus.getTrainSize // params['batch_size']

# Training loop
for epoch in range(max_epochs):
    loss = []
    for step in range(step_size):
        batch = primus.nextBatch(params, mode = 'Train')

        _, loss_value = sess.run([train_opt, loss],
                                 feed_dict={
                                    inputs: batch['inputs'],
                                    seq_len: batch['seq_lengths'],
                                    targets: ctc_utils.sparse_tuple_from(batch['targets']),
                                    rnn_keep_prob: dropout,
                                })
        loss.append(loss_value)

    if epoch % args.frequency == 0:
        # VALIDATION
        print ('Loss value at epoch ' + str(epoch + 1) + ':' + str(mean(loss_value)))
        print ('Validating...')

        val_step = primus.getValidationSize // params['batch_size']
        val_ed = 0
        val_len = 0
        val_count = 0
        
        for step in range(val_step):
            validation_batch = primus.nextBatch(params, mode = 'Validation')
            mini_batch_feed_dict = {
                inputs: validation_batch['inputs'],
                seq_len: validation_batch['seq_lengths'],
                rnn_keep_prob: 1.0            
                }
            
            prediction = sess.run(decoded,
                                  mini_batch_feed_dict)

            str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
            

            for i in range(len(str_predictions)):
                val_ed += ctc_utils.edit_distance(str_predictions[i], validation_batch['targets'])
                val_len = val_len + len(validation_batch['targets'][i])
                val_count = val_count + 1
            
    
        print ('[Epoch ' + str(epoch) + ']')
        print("Edit Distance: " + str(1. * val_ed / val_count))
        print(str(100. * val_ed / val_len) + ' SER from ' + str(val_count) + ' samples.')        
        print ('Saving the model...')
        saver.save(sess,args.save_model,global_step=epoch)
    print ('------------------------------')