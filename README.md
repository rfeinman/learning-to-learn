# Learning the Shape Bias with Neural Networks

In this project, we train a neural network on artificial toy data for the task
of object recognition. Each sample has a specific shape, color and texture
value. Objects with the same shape are of the same category.

## Requirements & Setup
This code repository requires Keras and TensorFlow. Keras must be
configured to use TensorFlow backend. A full list of requirements can be found
in `requirements.txt`. After cloning this repository, it is recommended that
you add the path to the repository to your `PYTHONPATH` environment variable
to enable imports from any folder:

    export PYTHONPATH="/path/to/learning-to-learn:$PYTHONPATH"


## Repository Structure
The repository contains 5 folders:

#### 1. learning2learn
This folder contains the core reusable source code for the project.

#### 2. scripts
This folder contains short Python scripts for running some experiments. Here
you will find scripts for training a neural network model and evaluating its
performance. There are also scripts for generating and saving datasets.

#### 3. notebooks
This folder contains a collection of Jupyter Notebooks for various small tasks,
such as synthesizing the artificial data sets and wrangling the data once
synthesized.

#### 4. data
This is where the artificial toy data sets will be saved to and loaded from.

#### 5. data
This is where experiment results will be saved to and loaded from.

## Results
Our ultimate goal is to model the infant learning tasks described in Smith
et al. 2002 using simple neural network (NN) models. In order to do so, we use
artificial toy data that is designed to mimic the data described in the paper.
Each sample in the dataset is assigned a shape, texture and color value.

### Canonical Bit-vector Features
Since these are categorical feature values, we encode the values using unique bit
vectors that are randomly assigned at the beginning of the experiment. A given
training set has a certain number of categories and a certain number of exemplars
per category; these quantities are varied. The shape values are perfectly
correlated with the categories, and the texture & color values are selected at
random from sets of 200 possible values. In Smith et al., there are 2 evaluation
metrics used: the first-order generalization and the second-order
generalization. Below, we describe experiments for each case. In both, we use a
simple feed-forward NN with one hidden layer of 30 units, and the
ReLU activation function.

#### 1. First-order Generalization
For the first-order generalization test, infants are asked to evaluate novel
instances of familiar objects. To simulate this test, we trained our NN model
to classify objects, ensuring that objects of the same category were assigned
the same shape in the training set. Then, we built a test set by creating one
novel instance of each category that was presented in the training set. Results
are shown below for a variety of different (# category) and (# exemplar/category)
values. With each dataset, the NN model was trained for 100 epochs. Keep in mind
that as the # category value increases, the classification task becomes more
challenging (more possible classes).

Set 1             |  Set 2
:----------------:|:----------------:
<img src="https://github.com/rfeinman/toy-neuralnet/blob/master/results/mlp_plot_firstOrder1.png" width="400" height="500"> | <img src="https://github.com/rfeinman/toy-neuralnet/blob/master/results/mlp_plot_firstOrder2.png" width="400" height="300">

#### 2. Second-order Generalization
For the second-order generalization test, infants are presented with an exemplar
of a novel object category as a baseline. Then, they are shown 3 comparison objects:
one which has the same shape as the baseline, one with the same color, and one
with the same texture. In each case, the other 2 features are different from
the baseline. The infants are asked to select which of the 3 comparison objects
are of the same category as the baseline object. We simulated this test by
creating an evaluation set containing groupings of 4 samples: the baseline,
the shape constant, the color constant, and the texture constant. Each grouping
serves as one test example. We find which of the 3 samples the NN thinks to be
most similar by evaluating the cosine similarity using the hidden layer features
of the model. The accuracy metric used is the % of groupings for which the
model chose the correct (shape-similar) object.

Set 1             |  Set 2
:----------------:|:----------------:
<img src="https://github.com/rfeinman/toy-neuralnet/blob/master/results/mlp_plot_secondOrder1.png" width="400" height="500"> | <img src="https://github.com/rfeinman/toy-neuralnet/blob/master/results/mlp_plot_secondOrder2.png" width="400" height="300">


### Image Data

Some examples of the images:

Exemplar #1                |  Exemplar #2
:-------------------------:|:-------------------------:
![image 0](https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_generated/img0000.png) | ![image 1](https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_generated/img0001.png)
Shape 0, Color 7, Texture 8 | Shape 0, Color 5, Texture 0
![image 2](https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_generated/img0002.png) | ![image 3](https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_generated/img0003.png)
Shape 1, Color 5, Texture 7 | Shape 1, Color 4, Texture 4
![image 12](https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_generated/img0004.png) | ![image 13](https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_generated/img0005.png)
Shape 2, Color 8, Texture 3 | Shape 2, Color 2, Texture 0

Results:
* Small toy dataset with 10 categories and 2 exemplars per category
* % time correct match was max: 100%
* Average cosine similarity for correct matches: 0.830
* Average cosine similarity for incorrect matches: 0.256