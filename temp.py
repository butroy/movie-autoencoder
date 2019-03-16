# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 23:07:59 2019

@author: Troy
"""

import tensorflow as tf
from data.dataset import _get_training_data, _get_test_data
from DAE import DAE
import numpy as np
import matplotlib.pyplot as plt

#tf.app.flags.DEFINE_string('tf_records_train_path', 
#                           '/Users/yifengbu/Desktop/autoencoder/train/',
#                           'Path of the training data.')

tf.app.flags.DEFINE_string('tf_records_train_path', 
                           'E://movie-autoencoder//train//',
                           'Path of the training data.')

#tf.app.flags.DEFINE_string('tf_records_test_path', 
#                           '/Users/yifengbu/Desktop/autoencoder/test/',
#                           'Path of the test data.')

tf.app.flags.DEFINE_string('tf_records_test_path', 
                           'E://movie-autoencoder//train//',
                           'Path of the test data.')

tf.app.flags.DEFINE_integer('num_epoch', 100,
                            'Number of training epochs.')

tf.app.flags.DEFINE_integer('batch_size', 8,
                            'Size of the training batch.')

tf.app.flags.DEFINE_float('learning_rate',0.001,
                          'Learning_Rate')

tf.app.flags.DEFINE_boolean('l2_reg', False,
                            'L2 regularization.'
                            )
tf.app.flags.DEFINE_float('lambda_',0.01,
                          'Wight decay factor.')

tf.app.flags.DEFINE_integer('num_v', 3952,
                            'Number of visible neurons (Number of movies the users rated.)')

tf.app.flags.DEFINE_integer('num_h', 128,
                            'Number of hidden neurons.)')

tf.app.flags.DEFINE_integer('num_samples', 5953,
                            'Number of training samples (Number of users, who gave a rating).')
FLAGS = tf.app.flags.FLAGS




num_batches=int(FLAGS.num_samples/FLAGS.batch_size)

with tf.Graph().as_default():
    train_loss_summary = []
    test_loss_summary =[]
    train_data, train_data_infer=_get_training_data(FLAGS)
    test_data=_get_test_data(FLAGS)
    
    iter_train = train_data.make_initializable_iterator()
    iter_train_infer=train_data_infer.make_initializable_iterator()
    iter_test=test_data.make_initializable_iterator()
    
    x_train= iter_train.get_next()
    x_train_infer=iter_train_infer.get_next()
    x_test=iter_test.get_next()
 
    model=DAE(FLAGS)

    train_op, train_loss_op=model._optimizer(x_train)
    pred_op, test_loss_op=model._validation_loss(x_train_infer, x_test)
   
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        train_loss=0
        test_loss=0

        for epoch in range(FLAGS.num_epoch):
            
            sess.run(iter_train.initializer)
            
            for batch_nr in range(num_batches):
                
                _, loss_=sess.run((train_op, train_loss_op))
                train_loss+=loss_
          
            sess.run(iter_train_infer.initializer)
            sess.run(iter_test.initializer)

            for i in range(FLAGS.num_samples):
                pred, loss_=sess.run((pred_op, test_loss_op))
                test_loss+=loss_
                
            print('epoch_nr: %i, train_loss: %.3f, test_loss: %.3f'%(epoch,(train_loss/num_batches),(test_loss/FLAGS.num_samples)))
            train_loss_summary.append(train_loss/num_batches)
            test_loss_summary.append(test_loss/FLAGS.num_samples)
            train_loss=0
            test_loss=0
            
'''            
fig,ax = plt.subplots()
for key in train_loss_all.keys():
    ax.plot(np.arange(FLAGS.num_epoch),train_loss_all[key],label='hSize:' + str(key))
    ax.set_xlabel('num of epoch')
    ax.set_ylabel('RMSE loss')
    ax.set_title('training loss')
    ax.legend(loc='upper right')
    ax.set_ylim((0,5))
plt.plot()
fig.savefig('hiddenSize_train.png',dpi=500)

fig,ax = plt.subplots()
for key in test_loss_all.keys():
    ax.plot(np.arange(FLAGS.num_epoch),test_loss_all[key],label='hSize:' + str(key))
    ax.set_xlabel('num of epoch')
    ax.set_ylabel('RMSE loss')
    ax.set_title('testing loss')
    ax.legend(loc='upper right')
    ax.set_ylim((0,2))
plt.plot()
fig.savefig('hiddenSize_test.png',dpi=500)




fig,ax = plt.subplots()
ax.plot(np.arange(FLAGS.num_epoch),train_loss_summary,label='train_loss')
ax.plot(np.arange(FLAGS.num_epoch),test_loss_summary,label='test_loss')
ax.set_xlabel('num of epoch')
ax.set_ylabel('RMSE loss')
ax.set_ylim((0,1.4))
ax.legend()
plt.show()
fig.savefig('conclusion.png',dpi=500)
'''