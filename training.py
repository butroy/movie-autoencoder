import tensorflow as tf
from data.dataset_TB import _get_training_data, _get_test_data
from DAE_TB import DAE
import numpy as np

def set_arguments():
    arguments={}
    arguments['tf_records_train_path'] = 'E://movie-autoencoder//train//'
    arguments['tf_records_test_path'] = 'E//movie-autoencoder//test//'
    arguments['num_epoch'] = 100 # num of training epoch
    arguments['batch_size'] = 8  #size of training batch  
    arguments['learning_rate'] = 0.001 #learning rate
    arguments['l2_reg'] = False  #l2 regularization switch
    arguments['lambda_'] = 0.01  #weight decay factor
    arguments['num_v'] = 3952    #num of visible neurons (Number of movies the users rated)
    arguments['num_h'] = 128     #num of hidden neurons
    arguments['num_samples'] = 5953 #num of training samples (Number of users, who gave a rating)
    return arguments


arguments = set_arguments()
num_batches=int(arguments['num_samples']/arguments['batch_size'])
print(arguments)

with tf.Graph().as_default():

    train_data, train_data_infer=_get_training_data(arguments)
    test_data=_get_test_data(arguments)
    
    iter_train = train_data.make_initializable_iterator()
    iter_train_infer=train_data_infer.make_initializable_iterator()
    iter_test=test_data.make_initializable_iterator()
    
    x_train= iter_train.get_next()
    x_train_infer=iter_train_infer.get_next()
    x_test=iter_test.get_next()

    model=DAE(arguments)

    train_op, train_loss_op=model._optimizer(x_train)
    pred_op, test_loss_op=model._validation_loss(x_train_infer, x_test)
   
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        train_loss=0
        test_loss=0

        for epoch in range(arguments['num_epoch']):
            
            sess.run(iter_train.initializer)
            
            for batch_nr in range(num_batches):
                
                _, loss_=sess.run((train_op, train_loss_op))
                train_loss+=loss_
          
            sess.run(iter_train_infer.initializer)
            sess.run(iter_test.initializer)

            for i in range(arguments['num_samples']):
                pred, loss_=sess.run((pred_op, test_loss_op))
                test_loss+=loss_
                
            print('epoch_nr: %i, train_loss: %.3f, test_loss: %.3f'%(epoch,(train_loss/num_batches),(test_loss/arguments['num_samples'])))
            train_loss=0
            test_loss=0

