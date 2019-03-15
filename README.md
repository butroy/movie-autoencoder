## Deep Autoencoder For Collaborative Filtering

In this project, I will explore what autoencoder is and how it works and then apply it in python scripts. I will also validate various properties mentioned in Oleksii and Boris' [paper](https://arxiv.org/pdf/1708.01715.pdf). 

### Introduction
An autoencoder is a deep learning neural network architecture that achieves a recommendation system in the area of collaborative filtering. Let's first explore how autoencoder is structured. 

### Deep autoencoder
****
**Autoencoder**
Architecturally, the for of an Autoencoder is a feedforward neural network having an input layer, one hidden layer and an output layer. Unlike conventional neural network structure, the output layer has the same number of neurons as the input layer in autoencoder for the purpose of reconstructing it's own inputs. This makes an autoencoder a form of unsupervised learning, which means we only need a set of input data instead of input-output pairs

![](https://github.com/butroy/movie-autoencoder/blob/master/plots/Fig1.png)



