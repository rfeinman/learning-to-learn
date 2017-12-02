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

As a preliminary type of training data, we used... TODO

Exemplar #1                |  Exemplar #2
:-------------------------:|:-------------------------:
10101-00000-00000 | 10101-00000-00000
Shape 0, Color 7, Texture 8 | Shape 0, Color 5, Texture 0
01010-00000-00000 | 01010-00000-00000
Shape 1, Color 5, Texture 7 | Shape 1, Color 4, Texture 4
10110-00000-00000 | 10110-00000-00000
Shape 2, Color 8, Texture 3 | Shape 2, Color 2, Texture 0


### Machine-generated 2-D object images

As a second type of training data, we used... TODO. Some examples of the images:

Exemplar #1                |  Exemplar #2
:-------------------------:|:-------------------------:
<img src="https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_generated/img0000.png" width="200" height="200"> | <img src="https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_generated/img0001.png" width="200" height="200">
Shape 0, Color 7, Texture 8 | Shape 0, Color 5, Texture 0
<img src="https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_generated/img0002.png" width="200" height="200"> | <img src="https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_generated/img0003.png" width="200" height="200">
Shape 1, Color 5, Texture 7 | Shape 1, Color 4, Texture 4
<img src="https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_generated/img0004.png" width="200" height="200"> | <img src="https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_generated/img0005.png" width="200" height="200">
Shape 2, Color 8, Texture 3 | Shape 2, Color 2, Texture 0


### Artist-designed 3-D object images

Exemplar #1                |  Exemplar #2
:-------------------------:|:-------------------------:
<img src="https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_artist/fake1_carpet_red.jpg" width="200" height="200"> | <img src="https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_artist/fake1_sponge_yellow.jpg" width="200" height="200">
Shape 0, Color 5, Texture 3 | Shape 0, Color 6, Texture 0
<img src="https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_artist/fake5_wood_pink.jpg" width="200" height="200"> | <img src="https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_artist/fake5_carpet_purple.jpg" width="200" height="200">
Shape 1, Color 4, Texture 7 | Shape 1, Color 3, Texture 3
<img src="https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_artist/fake4_sponge_orange.jpg" width="200" height="200"> | <img src="https://github.com/rfeinman/toy-neuralnet/blob/master/data/images_artist/fake4_wood_green.jpg" width="200" height="200">
Shape 2, Color 8, Texture 0 | Shape 2, Color 2, Texture 7

## Results
For a detailed description of results please see results/README.md.
