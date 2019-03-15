# Deep Autoencoder For Collaborative Filtering

In this project, I will explore what autoencoder is and how it works and then apply it in python scripts. I will also validate various properties mentioned in Oleksii and Boris' [paper](https://arxiv.org/pdf/1708.01715.pdf). 

## Introduction
An autoencoder is a deep learning neural network architecture that achieves a recommendation system in the area of collaborative filtering. Let's first explore how autoencoder is structured. 

## Deep autoencoder
****
**Autoencoder:**
Architecturally, the form of an Autoencoder is a feedforward neural network having an input layer, one hidden layer and an output layer. Unlike conventional neural network structure, the output layer has the same number of neurons as the input layer in autoencoder for the purpose of reconstructing it's own inputs. This makes an autoencoder a form of unsupervised learning, which means we only need a set of input data instead of input-output pairs

<p align="center">
  <img width="300" height="300" src="https://github.com/butroy/movie-autoencoder/blob/master/plots/Fig1.png">
</p>

The transition from the input to the hidden layer is called the encoding step and the transition from the hidden to the output layer is called the decoding step. 

**Deep Autoencoder:**
A deep autoencoder will have more hidden layers. The additional hidden layers enable the autoencoder to learn mathematically more complex underlying patterns in the data. 

The training process of an autoencoder is pretty much similar with conventional multi-layer perceptron training. We could randomly choose num of hidden layers and size of them, optimizer, activation functions and various hyperparameters. The detailed optimization will be shown in the following part.

## Implementation
In this project, I will apply a deep autoencoder that learns to predict the rating auers would give a movie. I will use the famouse [MovieLens dataset](https://grouplens.org/datasets/movielens/). MovieLens is web based recommender system and online community that recommends for users to watch.

For simplicity, I will use *ml-1m.zip* that contains 1,000,209 ratings of 3952 movies made by 5953 users. The *rating.dat* file contains 1,000,209 lines having the format of: user_id::movie_id::rating:time_stamp.

For example, 1::595::5::978824268 means user 1 gave movie 595 a 5 star rating, the time_stamp information is not useful for training and we drop it.

I first split the *rating.dat* file into training set and test set with a 80/20 ratio. 

I use an online [source](https://github.com/mikelaidata/autoencoder) as my base structure. My major work focuses on tuning hyperparameters and validate Oleksii and Boris' [paper](https://arxiv.org/pdf/1708.01715.pdf)

## Train and Optimize 
### Base Model Structure:
Mike's model has 3 hidden layers and each layer has 128 perceptrons. The default hyperparameters as following:

| Hyperparameters| Value         | Explanation  |
| -------------  |:-------------:| -----:|
| batch_size     | 16            | size of the training batch|
| learning_rate  | 0.0005        | learning rate |
| num of hidden perceptron  | 128    | Number of hidden neurons |
| l2_regularization | False | l2_regularization switch|

The model chooses MSE as the loss function and Adam as the optimizer.

The training result is
<p align="center">
  <img width="400" height="300" src="https://github.com/butroy/movie-autoencoder/blob/master/plots/p1_original_loss.png">
</p>

Training the base model in 50 epochs give train loss of 0.824 and test loss 0.795. This is a not bad result and let's see if we can optimize it. 

### HyperParameter tuning

* **batch_size**
<p align="center">
  <img src ="https://github.com/butroy/movie-autoencoder/blob/master/plots/p1_batch_size_train.png" width="300" />
  <img src="https://github.com/butroy/movie-autoencoder/blob/master/plots/p1_batch_size_test.png" width="300" /> 
 </p>



we could see from the above images that the most optimized batch size is 8.

* **learning_rate**
<p align="center">
  <img src ="https://github.com/butroy/movie-autoencoder/blob/master/plots/p1_learning_rate_training.png" width="300" />
  <img src="https://github.com/butroy/movie-autoencoder/blob/master/plots/p1_learning_rate_testing.png" width="300" /> 
 </p>

As our intuition, if we set the learning rate too big, e.g. 0.01, it's hard for model to get approximate to the best loss, and in the contrary, if we set the the learning rate too small, e.g. 1e-6, the step is too small so that the model will be hard to converge as well. Thus, the learning rate I will choose in the following work is 0.001

* **activation function**

I chose 5 activation functions to compare: Sigmoid, relu, tanh, elu, selu.

<p align="center">
  <img src ="https://github.com/butroy/movie-autoencoder/blob/master/plots/act_func_train.png" width="300" />
  <img src="https://github.com/butroy/movie-autoencoder/blob/master/plots/act_func_test.png" width="300" /> 
 </p>
 
As we see, elu and selu perform better then other activation functions and this result consistent with the paper's

* **wider network**

I also explored a hypothesis that would a wider network give a better result ?
I chose the base model with hidden neurons of 128, 256, 512 and 1024 in all layers, and the results are

<p align="center">
  <img src ="https://github.com/butroy/movie-autoencoder/blob/master/plots/hiddenSize_train.png" width="300" />
  <img src="https://github.com/butroy/movie-autoencoder/blob/master/plots/hiddenSize_test.png" width="300" /> 
 </p>

From the result, we could conclude that with more neurons in each layer doesn't necessarily increase the performance. In fact, wider network can make the model perform worse. The neuron sizes of 128 and 256 don't have big difference. 

* **l2 regularization**

L2 regularization is a loss term added in loss function to avoid the model overfitting. Below is the result if we include the l2 regularization term.

<p align="center">
  <img src ="https://github.com/butroy/movie-autoencoder/blob/master/plots/p4_l2reg.png" width="300" />
 </p>
 
 Compare with the base model which give train loss of ---- and test loss --- in 50 epochs, adding l2 regularization term in the loss function doesn't optimize the model. I think this is because our model never overfit the training set and adding regularization will make the neurons away from the correct direction when computing the gradient.

* **Batch Normalization**

Let's see if we add batch normalization after each layer could optimize

<p align="center">
  <img src ="https://github.com/butroy/movie-autoencoder/blob/master/plots/p4_BN.png" width="300" />
 </p>
 
Obviously, the model overfits the training dataset and even the train loss is not as good as our base model. 



* More hidden layers

In Mike's autoencoder model, it has 3 hidden layers with 128 perceptron in each. And if I choose batch_size of 8 and learning rate of 0.001, the training result is: 

<p align="center">
  <img width="400" height="300" src="https://github.com/butroy/movie-autoencoder/blob/master/plots/p1_original_layer.png">
</p>

I add 3 more hidden layers and the results are


128-128-128-128-128-128-3952            |  128-64-32-64-128-128-3952  
:-------------------------:|:-------------------------:
![](https://github.com/butroy/movie-autoencoder/blob/master/plots/p2_128_128_128_layer.png)  |  ![](https://github.com/butroy/movie-autoencoder/blob/master/plots/p2_128_64_32_layer.png)





