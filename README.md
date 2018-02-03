# Learning Inductive Biases with Simple Neural Networks

This repository contains the code for "Learning Inductive
Biases with Simple Neural Networks" (Feinman & Lake 2018).

## Requirements & Setup
This code repository requires Keras and TensorFlow. Keras must be
configured to use TensorFlow backend. A full list of requirements can be found
in `requirements.txt`. After cloning this repository, it is recommended that
you add the path to the repository to your `PYTHONPATH` environment variable
to enable imports from any folder:

    export PYTHONPATH="/path/to/learning-to-learn:$PYTHONPATH"


## Repository Structure
The repository contains 6 subfolders:

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

#### 5. results
This is where experiment results will be saved to and loaded from.

#### 6. paper_latex
This is where the latex for the CogSci paper is kept.

## Datasets

### Bitvector categorical features

Exemplar #1                |  Exemplar #2
:-------------------------:|:-------------------------:
10101-00000-00000 | 10101-00000-00000
Shape 0, Color 7, Texture 8 | Shape 0, Color 5, Texture 0
01010-00000-00000 | 01010-00000-00000
Shape 1, Color 5, Texture 7 | Shape 1, Color 4, Texture 4
10110-00000-00000 | 10110-00000-00000
Shape 2, Color 8, Texture 3 | Shape 2, Color 2, Texture 0


### Machine-generated 2-D object images

<img src="https://github.com/rfeinman/toy-neuralnet/blob/master/data/generated_images.pdf" width="400" height="400">
