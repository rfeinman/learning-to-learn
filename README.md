# Learning the Shape Bias with Neural Networks

In this project, we train simple neural network models on artificial toy data for the task
of object recognition. Each object sample has a specific shape, color and texture
value. Objects with the same shape are of the same category. The data is
designed to mimic the objects that were used to teach children the shape bias
from [Smith et al. 2002](https://www.ncbi.nlm.nih.gov/pubmed/11892773). Ultimately,
we hope to analyze the development of the shape bias during object category
learning and the influence that this bias has on future learning ("learning-to-learn").

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

#### 5. results
This is where experiment results will be saved to and loaded from.

## Results
For a detailed description of results please see results/README.md.
