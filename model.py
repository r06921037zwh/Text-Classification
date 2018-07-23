# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 09:45:06 2018

@author: zhewei
"""   
import os
import numpy as np
import pickle
import random
from keras.utils.np_utils import to_categorical  
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tqdm import tqdm

from attention import attention
from utils import batch_generator

NUM_WORDS = 5000
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 256
HIDDEN_SIZE = 300
ATTENTION_SIZE = 100
KEEP_PROB = 0.8
BATCH_SIZE = 100
NUM_EPOCHS = 4        # Model easily overfits without pre-trained words embeddings, that's why train for a few epochs
DELTA = 0.5
NUM_CLASSES = 23
MODEL_PATH = 'model'

        
dic = pickle.load(open(os.path.join('processed_data','dic.pkl'), 'rb'))
data = pickle.load(open(os.path.join('processed_data','data.pkl'), 'rb'))  
      
# Load the data set
X, X_len, y  = np.array(data['X']), np.array(data['X_len']), np.array(data['y'], dtype=np.float32)
y = to_categorical(y, num_classes=NUM_CLASSES)

random_index = np.arange(len(X))
random.shuffle(random_index)
train_num = int(0.8 * len(X))
X_train = X[random_index[:train_num]]
X_train_len = X_len[random_index[:train_num]]
y_train = y[random_index[:train_num]]

X_test = X[random_index[train_num:]]
X_test_len = X_len[random_index[train_num:]]
y_test = y[random_index[train_num:]]

# Different placeholders
with tf.name_scope('Inputs'):
    batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='batch_ph')
    target_ph = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='target_ph')
    seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

# Embedding layer
with tf.name_scope('Embedding_layer'):
    embeddings_var = tf.Variable(tf.random_uniform([NUM_WORDS, EMBEDDING_DIM], -1.0, 1.0), trainable=False)
    tf.summary.histogram('embeddings_var', embeddings_var)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
    # (Bi-)RNN layer(-s)
    rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
                            inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
    tf.summary.histogram('RNN_outputs', rnn_outputs)

# Attention layer
with tf.name_scope('Attention_layer'):
    attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
    tf.summary.histogram('alphas', alphas)

# Dropout
drop = tf.nn.dropout(attention_output, keep_prob_ph)
#rnn_out = tf.reshape(drop, [-1, 1])

# Fully connected layer
with tf.name_scope('Fully_connected_layer'):
    W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE * 2, NUM_CLASSES], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
    b = tf.Variable(tf.constant(0., shape=[NUM_CLASSES]))
    y_hat = tf.nn.xw_plus_b(drop, W, b)
    y_hat = tf.squeeze(y_hat)
    predictions = tf.argmax(y_hat, 1, name='predictions')
    tf.summary.histogram('W', W)

'''  
with tf.name_scope('Fully_connected_layer'):
    fc1 = fully_connected(drop, num_outputs=100)
    print(tf.shape(fc1))
    y_hat = fully_connected(fc1, num_outputs=23)
    print(tf.shape(y_hat))
    #y_hat = tf.squeeze(fc2)
    #print(tf.shape(y_hat))
    predictions = tf.argmax(y_hat, 1, name='predictions')
''' 
with tf.name_scope('Threshold'):
    threshold = tf.Variable(tf.random_normal([1]), trainable=True)
    
with tf.name_scope('Metrics'):
    # Cross-entropy loss and optimizer initialization
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=target_ph))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    # Accuracy metric
    correct_predictions = tf.equal(predictions, tf.argmax(target_ph, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
    tf.summary.scalar('accuracy', accuracy)
    

    
merged = tf.summary.merge_all()

# Batch generators
train_batch_generator = batch_generator(X_train, y_train, X_train_len, BATCH_SIZE)
test_batch_generator = batch_generator(X_test, y_test, X_test_len, BATCH_SIZE)

train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
test_writer = tf.summary.FileWriter('./logdir/test', accuracy.graph)

session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

saver = tf.train.Saver()

if __name__ == "__main__":
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, os.path.join(MODEL_PATH, 'model.ckpt'))
        print("Start learning...")
        for epoch in range(NUM_EPOCHS):
            loss_train = 0
            loss_test = 0
            accuracy_train = 0
            accuracy_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training
            num_batches = X_train.shape[0] // BATCH_SIZE
            for b in tqdm(range(num_batches)):
                x_batch, y_batch, seq_len = next(train_batch_generator)
                loss_tr, acc, _, summary = sess.run([loss, accuracy, optimizer, merged],
                                                    feed_dict={batch_ph: x_batch,
                                                               target_ph: y_batch,
                                                               seq_len_ph: seq_len,
                                                               keep_prob_ph: KEEP_PROB})
                accuracy_train += acc
                loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
                train_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_train /= num_batches

            # Testing
            num_batches = X_test.shape[0] // BATCH_SIZE
            for b in tqdm(range(num_batches)):
                x_batch, y_batch, seq_len = next(test_batch_generator)
                loss_test_batch, acc, summary = sess.run([loss, accuracy, merged],
                                                         feed_dict={batch_ph: x_batch,
                                                                    target_ph: y_batch,
                                                                    seq_len_ph: seq_len,
                                                                    keep_prob_ph: 1.0})
                accuracy_test += acc
                loss_test += loss_test_batch
                test_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_test /= num_batches
            loss_test /= num_batches

            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
                loss_train, loss_test, accuracy_train, accuracy_test
            ))
        train_writer.close()
        test_writer.close()
        saver.save(sess, os.path.join(MODEL_PATH, 'model.ckpt'))
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")