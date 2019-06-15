import tensorflow as tf
from primus import CTC_PriMuS
import ctc_utils
import ctc_model
import argparse

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
args = parser.parse_args()

# Load primus

primus = CTC_PriMuS(args.corpus,args.set,args.voc, args.semantic, distortions = True, val_split = 0.1)

# Parameterization
img_height = 128
params = ctc_model.default_model_params(img_height,primus.vocabulary_size)
max_epochs = 64000
dropout = 0.5

# Model
inputs, seq_len, targets, decoded, loss, rnn_keep_prob = ctc_model.ctc_crnn(params)
train_opt = tf.train.AdamOptimizer().minimize(loss)


saver = tf.train.Saver(max_to_keep=None)
sess.run(tf.global_variables_initializer())

step_size = 
# Training loop
for epoch in range(max_epochs):
    batch = primus.nextBatch(params)

    _, loss_value = sess.run([train_opt, loss],
                             feed_dict={
                                inputs: batch['inputs'],
                                seq_len: batch['seq_lengths'],
                                targets: ctc_utils.sparse_tuple_from(batch['targets']),
                                rnn_keep_prob: dropout,
                            })

    if epoch % 1000 == 0:
        # VALIDATION
        print ('Loss value at epoch ' + str(epoch) + ':' + str(loss_value))
        print ('Validating...')
        print("Checking for Memory Error 1...")

        validation_batch, validation_size = primus.getValidation(params) #One Batch Only
        print("Checking for Memory Error 2...")
        
        val_idx = 0
        
        val_ed = 0
        val_len = 0
        val_count = 0
            
        
        mini_batch_feed_dict = {
            inputs: validation_batch['inputs'][val_idx:val_idx+params['batch_size']],
            seq_len: validation_batch['seq_lengths'][val_idx:val_idx+params['batch_size']],
            rnn_keep_prob: 1.0            
        }            
                        
            
        prediction = sess.run(decoded,
                              mini_batch_feed_dict)

        str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
        print("Checking for Memory Error 3...")


        for i in range(len(str_predictions)):
            ed = ctc_utils.edit_distance(str_predictions[i], validation_batch['targets'][val_idx+i])
            val_ed = val_ed + ed
            val_len = val_len + len(validation_batch['targets'][val_idx+i])
            val_count = val_count + 1

        val_idx = val_idx + params['batch_size']
    
        print ('[Epoch ' + str(epoch) + '])
        print('Edit Distance: " + str(1. * val_ed / val_count)
        print(str(100. * val_ed / val_len) + ' SER from ' + str(val_count) + ' samples.')        
        print ('Saving the model...')
        saver.save(sess,args.save_model,global_step=epoch)
        print ('------------------------------')
