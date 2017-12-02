# Results
Our ultimate goal is to model the infant learning tasks described in
[Smith et al. 2002](https://www.ncbi.nlm.nih.gov/pubmed/11892773)
using simple neural network models. In order to do so, we use
artificial toy data that is designed to mimic the data described in the paper.
Each object sample is assigned a shape, texture and color value. There are two
types of model evaluations performed, both drawn from Smith et al. 2002:

### 1. First-order Generalization Test
For the first-order generalization test, infants are asked to evaluate novel
instances of familiar objects. To simulate this test, we trained our neural network models
to classify objects, ensuring that objects of the same category were assigned
the same shape. Then, we built a test set by creating one novel exemplar of each
category that was presented in the training set. The novel
exemplar has the same shape as the training exemplars of that category, but a
new color and texture combination. This test was repeated for different training
set sizes, i.e. different combinations of (# categories, # exemplars).

### 2. Second-order Generalization Test
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
model chose the correct (shape-similar) object. This test was repeated for different training
set sizes, i.e. different combinations of (# categories, # exemplars).

## Simple MLP with Bit-vector Data
To begin with, we use a simple Multi-layer Perceptron (MLP) that operates on categorical
data. Since *shape, color, & texture* have categorical feature values, we encode
the values using unique bit
vectors that are randomly assigned at the beginning of the experiment. We use a
simple feed-forward NN with one hidden layer of 30 units, and the ReLU activation
function. The number of units in the softmax layer depends on the *# categories* parameter
for the particular dataset.

### 1. First-order Generalization
Results are shown below for a variety of different (# categories, # exemplars)
training set sizes. With each dataset, the NN model was trained for 100 epochs. Keep in mind
that as the # category value increases, the classification task becomes more
challenging (more possible classes).

Set 1             |  Set 2
:----------------:|:----------------:
<img src="https://github.com/rfeinman/toy-neuralnet/blob/master/results/mlp_plot_firstOrder1.png" width="400" height="500"> | <img src="https://github.com/rfeinman/toy-neuralnet/blob/master/results/mlp_plot_firstOrder2.png" width="300" height="350">

### 2. Second-order Generalization
Results are shown below for a variety of different (# categories, # exemplars)
training set sizes.

Set 1             |  Set 2
:----------------:|:----------------:
<img src="https://github.com/rfeinman/toy-neuralnet/blob/master/results/mlp_plot_secondOrder1.png" width="400" height="500"> | <img src="https://github.com/rfeinman/toy-neuralnet/blob/master/results/mlp_plot_secondOrder2.png" width="300" height="350">



## Simple CNN with Image Data
As a second type of model, we generated images of artificial 2-D objects. Some
examples of these images can be found in the "Datasets" section of the main
repository README. TODO...

### 1. First-order Generalization
Results are shown below for a variety of different (# categories, # exemplars)
training set sizes. With each dataset, the NN model was trained for 100 epochs. Keep in mind
that as the # category value increases, the classification task becomes more
challenging (more possible classes).

<img src="https://github.com/rfeinman/toy-neuralnet/blob/master/results/cnn_plot_firstOrder.png" width="400" height="500">

### 2. Second-order Generalization
Results are shown below for a variety of different (# categories, # exemplars)
training set sizes.

<img src="https://github.com/rfeinman/toy-neuralnet/blob/master/results/cnn_plot_secondOrder.png" width="400" height="500">