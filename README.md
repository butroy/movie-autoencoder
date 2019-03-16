# Deep Autoencoder For Movie Recommendation System
In this project, I will explore what autoencoder is and how it works and then apply it in python. I will also test several properties mentioned in Oleksii and Boris' [paper](https://arxiv.org/pdf/1708.01715.pdf). 

## Introduction
An autoencoder is a deep learning neural network architecture that achieves a recommendation system in the area of collaborative filtering. Let's first explore how autoencoder is structured. 

## Deep autoencoder
**Autoencoder:**
Architecturally, the form of an Autoencoder is a feedforward neural network having an input layer, one hidden layer and an output layer. Unlike conventional neural network structure, the output layer has the same number of neurons as the input layer in autoencoder for the purpose of reconstructing it's own inputs. This makes an autoencoder a form of unsupervised learning, which means we only need a set of input data instead of input-output pairs

<p align="center">
  <img width="400" height="400" src="https://github.com/butroy/movie-autoencoder/blob/master/plots/Fig1.png">
</p>

The transition from the input to the hidden layer is called the encoding step and the transition from the hidden to the output layer is called the decoding step. 

**Deep Autoencoder:**
A deep autoencoder will have more hidden layers. The additional hidden layers enable the autoencoder to learn mathematically more complex underlying patterns in the data. 

The training process of an autoencoder is very similar with a conventional multi-layer perceptron training. We could randomly choose num of hidden layers, size of each layer, optimizer, activation functions and various hyperparameters. The detailed optimization will be shown in the following part.

## Implementation
In this project, I applied a deep autoencoder that learns to predict the rating a user would give a movie. I used the famouse [MovieLens dataset](https://grouplens.org/datasets/movielens/). MovieLens is web based recommender system and online community that recommends for users to watch.

For simplicity, I used *ml-1m.zip* that contains 1,000,209 ratings of 3952 movies made by 5953 users. The *ratings.dat* file contains 1,000,209 lines having the format of: user_id::movie_id::rating:time_stamp.

For example, 1::595::5::978824268 means user 1 gave movie 595 a 5 star rating, the time_stamp information is not useful for training and we drop it.

I first split the *ratings.dat* file into training set and test set with a 80/20 ratio. 

I use [Mike's Model](https://github.com/mikelaidata/autoencoder) as my base model structure. My major work focuses on tuning hyperparameters and validate Oleksii and Boris' [paper](https://arxiv.org/pdf/1708.01715.pdf)

## Train and Optimize 
### Base Model Structure:
[Mike's model](https://github.com/mikelaidata/autoencoder) has 3 hidden layers and each layer has 128 neurons. The default hyperparameters are as follows:

| Hyperparameters| Value         | Explanation  |
| -------------  |:-------------:| -----:|
| batch_size     | 16            | size of the training batch|
| learning_rate  | 0.0005        | learning rate |
| num of hidden neurons  | 128    | Number of neurons in each layer|
| l2_regularization | False | l2_regularization switch|

The model chooses MSE as the loss function and Adam as the optimizer. To estimate performance, I use RMSE loss score.

The training result is
<p align="center">
  <img width="400" height="300" src="https://github.com/butroy/movie-autoencoder/blob/master/plots/base_model.png">
</p>

Training the base model in 50 epochs gives a train loss of 0.819 and a test loss 0.790. This is a not bad result and let's see if we could optimize it. 

### HyperParameter tuning

* **batch_size**

Sticking with the base model, let's sweep over different batch sizes.

<p align="center">
  <img src ="https://github.com/butroy/movie-autoencoder/blob/master/plots/p1_batch_size_train.png" width="400" />
  <img src="https://github.com/butroy/movie-autoencoder/blob/master/plots/p1_batch_size_test.png" width="400" /> 
 </p>

We could see from the above images that the best batch size is 8.

* **learning_rate**

Next, let's check different learning rates.

<p align="center">
  <img src ="https://github.com/butroy/movie-autoencoder/blob/master/plots/p1_learning_rate_training.png" width="400" />
  <img src="https://github.com/butroy/movie-autoencoder/blob/master/plots/p1_learning_rate_testing.png" width="400" /> 
 </p>

As our intuition, if we set the learning rate too big, e.g. 0.01, it's hard for model to get closing to the best loss, and in the contrary, if we set the the learning rate too small, e.g. 1e-6, the step is too small so that the model will be hard to converge as well. Thus, the learning rate I will choose in the following work is 0.001

* **activation function**

I chose 5 activation functions to compare: Sigmoid, relu, tanh, elu, selu.

<p align="center">
  <img src ="https://github.com/butroy/movie-autoencoder/blob/master/plots/act_func_train.png" width="400" />
  <img src="https://github.com/butroy/movie-autoencoder/blob/master/plots/act_func_test.png" width="400" /> 
 </p>
 
As we see, elu and selu perform better then other activation functions and this result consistent with the paper's result.

* **wider network**

I also explored the hypothesis that would a wider network give a better result ?
I chose the base model with hidden neurons of 128, 256, 512 and 1024 in all layers, and the results are

<p align="center">
  <img src ="https://github.com/butroy/movie-autoencoder/blob/master/plots/hiddenSize_train.png" width="400" />
  <img src="https://github.com/butroy/movie-autoencoder/blob/master/plots/hiddenSize_test.png" width="400" /> 
 </p>

From the result, we could conclude that with more neurons in each layer doesn't necessarily increase the performance. In fact, wider network can make the model perform worse. The neuron sizes of 128 and 256 don't have big difference. 

* **l2 regularization**

L2 regularization is a loss term added in loss function to avoid overfitting. Below is the result of including l2 regularization term.

<p align="center">
  <img src ="https://github.com/butroy/movie-autoencoder/blob/master/plots/p4_l2reg.png" width="400" />
 </p>
 
 Compare with the base model which give train loss of 0.819 and test loss 0.790 in 50 epochs, adding l2 regularization term in the loss function doesn't optimize the model. I think this is because our model never overfit the training set and adding regularization will make the neurons away from the correct direction when computing the gradient.

* **Batch Normalization**

Let's see if we add batch normalization after each layer could optimize

<p align="center">
  <img src ="https://github.com/butroy/movie-autoencoder/blob/master/plots/p4_BN.png" width="400" />
 </p>
 
Obviously, the model overfits the training dataset and even the train loss is not as good as our base model. 



* **More hidden layers**

In Mike's autoencoder model, it has 3 hidden layers with 128 perceptron in each. And when I choose batch_size as 8, learning rate as 0.001 and elu as the activation function, the training result is: 

<p align="center">
  <img width="400" height="300" src="https://github.com/butroy/movie-autoencoder/blob/master/plots/original_layer_elu.png">
</p>

To verify more hidden layer's property, I add 3 hidden layers in the above model, one has 128 neruons in each layer and the other has a 64-32-64 neurons to see which one could beats our original model. 

128-128-128-128-128-128            |  128-64-32-64-128-128  
:-------------------------:|:-------------------------:
![](https://github.com/butroy/movie-autoencoder/blob/master/plots/P2_128_128_128_elu.png)  |  ![](https://github.com/butroy/movie-autoencoder/blob/master/plots/P2_64_32_64_elu.png)

The test loss of our original model could reach 0.584 and neither of the two structures with more hidden layers could beat it. However, this result contradicts with the [paper](https://arxiv.org/pdf/1708.01715.pdf)'s result, which says that "there is a positive correlation between the number of layers and the evaluation accuracy." I think the reason for the contradiction is the paper users Netflix dataset which is about 100 times larger than mine and thus it would need more parameters to build an efficient network. However, this is just my hypothesis and needs to be further verified. 

## Conclusion
<p align="center">
  <img width="400" height="300" src="https://github.com/butroy/movie-autoencoder/blob/master/plots/conclusion.png">
</p>

In this project, I learnt to use autoencoder to predict user's rating on movies. I applied [Mike](https://github.com/mikelaidata/autoencoder)'s autoencoder and tuned hyperparameters by taking Oleksii and Boris' [paper](https://arxiv.org/pdf/1708.01715.pdf) as a reference. I test and concluded that with a batch size of 8, learning rate of 0.001, elu as the activation function, 128 neurons in each layear, not including l2 regularization and batch nomalization could give the best RMSE loss result with a train loss of 0.593 and a test loss of 0.488.

My test based on Oleksii and Boris' [paper](https://arxiv.org/pdf/1708.01715.pdf) validates most of their hypothesis except for the impacts of a larger network. My result shows that adding more layers actually decreases performance while the paper thinks that the number of layers and the performance have a postive correlation. I guess the contradiction comes from the different sizes of our datasets: their dataset is about 100 times larger than mine. Intuitively, they will need more parameters to get an optimized result. However, this guess needs to be further verified. 

### Reference

https://github.com/mikelaidata

https://grouplens.org/datasets/movielens/

https://arxiv.org/pdf/1708.01715.pdf

https://towardsdatascience.com/deep-autoencoders-for-collaborative-filtering-6cf8d25bbf1d
 




