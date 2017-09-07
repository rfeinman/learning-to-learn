# Neural network learning of object categories from canonicalized feature representations

In this project, we train a neural network on artificial toy data for the task
of object recognition. Each sample is represented by a shape,
color and texture value.

## Requirements & Setup
This code repository requires Keras and TensorFlow. Keras must be
configured to use TensorFlow backend. A full list of requirements can be found
in `requirements.txt`. After cloning this repository, it is recommended that
you add the path to the repository to your `PYTHONPATH` environment variable
to enable imports from any folder:

    export PYTHONPATH="/path/to/toy-neuralnet:$PYTHONPATH"


## Repository Structure
The repository contains 3 folders:

### 1. Data
This is where the artificial toy data sets will be saved to and loaded from.

### 2. Notebooks
This folder contains a collection of Jupyter Notebooks for various small tasks,
such as synthesizing the artificial data sets and wrangling the data once
synthesized.

### 3. Toy-neuralnet
This folder contains core reusable source code for the experiments.

### 3. Scripts
This folder contains short Python scripts for running some experiments. Here
you will find scripts for training a neural network model and evaluating its
performance.

## Demo

To run train_nn.py from the scripts folder, you must indicate the data file and
the labels file to use for training. The command might look as follows:

    python train_nn.py -d ../data/objects.csv -l ../data/labels.csv


## Results (in progress)
Our ultimate goal is to model the infant learning tasks described in Smith
et al. 2002 using simple neural network (NN) models. In order to do so, we use
artificial toy data that is designed to mimic the data described in the paper.
Each sample in the dataset is assigned a shape, texture and color value. Since
these are categorical feature values, we encode the values using unique bit
vectors that are randomly assigned at the beginning of the experiment. A given
training set has a certain number of categories and a certain number of exemplars
per category; these quantities are varied. The shape values are perfectly
correlated with the categories, and the texture & color values are selected at
random from a set of 200 possible values. In Smith et al., there are 2 evaluation
metrics used: the first-order generalization and the second-order
generalization. Below, we describe experiments for each case. In both, we use a
simple feed-forward NN with one hidden layer of 30 units, and the
ReLU activation function.

### First-order Generalization
For the first-order generalization test, infants are asked to evaluate novel
instances of familiar objects. To simulate this test, we trained our NN model
to classify objects, ensuring that objects of the same category were assigned
the same shape in the training set. Then, we built a test set by creating one
novel instance of each category that was presented in the training set. Results
are shown below for a variety of different (# category) and (# exemplar/category)
values. With each dataset, the NN model was trained for 100 epochs. Keep in mind
that as the # category value increases, the classification task becomes more
challenging (more possible classes).

| # categories    | # exemplars   | test accuracy\
| 100             | 3             | 86.0%\
| 100             | 5             | 100.0%\
100             | 10            | 100.0%
500             | 3             | 99.4%
500             | 5             | 100.0%
500             | 10            | 100.0%
1000            | 3             | 97.8%
1000            | 5             | 100.0%
1000            | 10            | 100.0%
5000            | 3             | 96.2%
5000            | 5             | 99.3%
5000            | 10            | 99.7%


### Second-order Generalization
TODO