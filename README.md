# Learning Inductive Biases with Simple Neural Networks

Code for the paper "Learning Inductive
Biases with Simple Neural Networks" (Feinman & Lake 2018).

## Requirements & Setup
This code repository requires Keras and TensorFlow. Keras must be
configured to use TensorFlow backend. A full list of requirements can be found
in `requirements.txt`. After cloning this repository, it is recommended that
you add the path to the repository to your `PYTHONPATH` environment variable
to enable imports from any folder:

    export PYTHONPATH="/path/to/learning-to-learn:$PYTHONPATH"


## Repository Structure
The repository contains 5 subfolders:

#### 1. learning2learn
This folder contains the core reusable source code for the project.

#### 2. scripts
This folder contains short Python scripts for running some experiments. Here
you will find scripts for training a neural network model and evaluating its
performance.

#### 3. notebooks
This folder contains a collection of Jupyter Notebooks for various small tasks,
such as plotting results and performing parametric sensitivity tests.

#### 4. data
This is a placeholder folder for image and model data. The Brodatz texture
dataset is stored here.

#### 5. results
This is where experiment results will be saved to and loaded from.

## Running the Experiments

### Experiment 1

To train the MLP of Experiment 1 on all dataset sizes, i.e. all pairs of
{# categories, # examples}, run the following command using `mlp_loop.py` from
the scripts folder:

    python mlp_loop.py -ep=200 -r=10 -b=32 -s=</path/to/save/folder>

Where `/path/to/save/folder` is a string containing the folder name you'd like
to use for the results (folder should not yet exist, or it will be overwritten).
The parameter `ep=200` indicates that you'd like to train for 200 epochs,
parameter `r=10` indicates that you'd like to train 10 model runs for each
dataset size, and parameter `b=32` indicates that you'd like to use a batch
size of 32 (although, for a dataset with N samples, the batch size will be
min(N/5, 32) to ensure that we use at least 5 batches in SGD).

### Experiment 2

TODO