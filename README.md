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

where `</path/to/save/folder>` is a string containing the folder name you'd like
to use for the results (folder should not yet exist, or it will be overwritten).
This will default to `../results/mlp_results` if left unspecified. Results
of the 1st-order and 2nd-order generalization tests will be recorded for all 10
trials of each dataset size.
The parameter `ep=200` indicates that you'd like to train for 200 epochs,
parameter `r=10` indicates that you'd like to train 10 model runs for each
dataset size, and parameter `b=32` indicates that you'd like to use a batch
size of 32 (although, for a training set with N samples, the batch size will be
min(N/5, 32) to ensure that we use at least 5 batches in SGD). This model will
be trained on CPU, as it is too small to benefit from GPU.

Once training is complete, you can plot heatmaps & contours of the results using
`notebooks/plot_results_experiments1and2.ipynb`.

To perform the parametric sensitivity tests with the MLP, see
`notebooks/parametric_tests_mlp.ipynb` for a walk-through.

### Experiment 2

To train the CNN of Experiment 2 on all dataset sizes, i.e. all pairs of
{# categories, # examples}, run the following command using `cnn_loop.py` from
the scripts folder:

    python cnn_loop.py -ep=400 -r=10 -b=32 -s=</path/to/save/folder> -g=0

where `</path/to/save/folder>` is again a string containing the folder name
you'd like to use for the results. This will default to
`../results/cnn_results` if left unspecified. Note that we are using 400
epochs as opposed to the 200 from Experiment 1. The additional parameter `-g=0`
indicates which GPU you would like to use for training, as this experiment will
benefit significantly from GPU speedup. Defaults to the system default if left
unspecified.

A bottleneck of this experiment is the building of the image datasets used for
the generalization tests. These datasets each contain 1000 trials
(1000x4 = 4000 images). I have parallelized the code using multiprocessing,
selecting a # of processes based on the available resources. You will see a
significant speedup with a larger CPU count machine.

Once training is complete, you can plot heatmaps & contours of the results using
`notebooks/plot_results_experiments1and2.ipynb`.

To perform the parametric sensitivity tests with the CNN, see
`notebooks/parametric_tests_cnn.ipynb` for a walk-through.

### Experiment 3

To train the 20 models of Experiment 3, run the following command using
`vocabulary_acceleration.py` from the scripts folder:

    python vocabulary_acceleration_multi.py -ep=70 -sf=0.6 -cf=0.2 -ca=60 -ex=10 -b=10 -r=20 -t=500 -g=0 -sp=</path/to/save/folder>

where `</path/to/save/folder>` is again a string containing the folder name
you'd like to use for the results. For each model, the cumulative vocabulary
size and the 2nd-order generalization test results at each epoch will be stored
in a file called `run%i.csv` where `%i` is the index of the particular model.

Once training is complete, you can analyze the results using
`notebooks/analyze_vocabulary_acceleration.ipynb`.