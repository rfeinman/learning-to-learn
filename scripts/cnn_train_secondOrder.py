from __future__ import division, print_function
import os
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
import matplotlib as mpl
mpl.use('Agg')

from learning2learn.models import simple_cnn
from learning2learn.wrangle import synthesize_data, get_secondOrder_data
from learning2learn.util import evaluate_secondOrder, train_model

def run_experiment(nb_categories, nb_exemplars, params):
    # Create custom TF session if requested
    if params['gpu_options'] is not None:
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=params['gpu_options'])
        )
        K.set_session(sess)
    print('Training CNN model...')
    scores = []
    for i in range(params['nb_trials']):
        print('Round #%i' % (i + 1))
        # Get the dataset (random subset is selected)
        df_train, labels = synthesize_data(nb_categories, nb_exemplars)
        X_train, X_test = get_secondOrder_data(df_train, nb_test_trials=1000)
        ohe = OneHotEncoder(sparse=False)
        Y_train = ohe.fit_transform(labels.reshape(-1, 1))
        # Build a neural network model and train it with the training set
        model = simple_cnn(input_shape=X_train.shape[1:],
                           nb_classes=Y_train.shape[-1])
        # We're going to keep track of the best model throughout training,
        # monitoring the training loss
        weights_file = '../data/cnn_secondOrder.h5'
        if os.path.isfile(weights_file):
            os.remove(weights_file)
        checkpoint = ModelCheckpoint(
            weights_file,
            monitor='loss',
            save_best_only=True,
            save_weights_only=True
        )
        train_model(
            model, X_train, Y_train, epochs=params['nb_epochs'],
            validation_data=None, batch_size=params['batch_size'],
            checkpoint=checkpoint
        )
        # Now that we've completed all training epochs, let's go ahead and
        # load the best model
        model.load_weights(weights_file)
        # Now evaluate the model on the test data
        score = evaluate_secondOrder(
            model, X_test, layer_num=-4,
            batch_size=128
        )
        scores.append(score)
    avg_score = np.mean(scores)
    print('\nScore: %0.4f' % avg_score)
    if params['gpu_options'] is not None:
        K.clear_session()
        sess.close()

    return avg_score

def main():
    # GPU settings
    if args.gpu_num is not None:
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    visible_device_list=args.gpu_num)
    else:
        gpu_options = None
    # Create the experiment parameter dictionary
    params = {
        'nb_epochs': args.nb_epochs,
        'batch_size': args.batch_size,
        'nb_trials': 1,
        'img_size': (200, 200),
        'gpu_options': gpu_options
    }
    # Run the experiment
    acc = run_experiment(args.nb_categories, args.nb_exemplars, params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    parser.add_argument('-ca', '--nb_categories',
                        help='The number of categories.',
                        required=False, type=int)
    parser.add_argument('-ex', '--nb_exemplars',
                        help='The number of exemplars.',
                        required=False, type=int)
    parser.add_argument('-g', '--gpu_num',
                        help='Int indicating which GPU to use',
                        required=False, type=str)
    parser.add_argument('-b', '--batch_size',
                        help='Int indicating the batch size to use',
                        required=False, type=int)
    parser.set_defaults(nb_epochs=100)
    parser.set_defaults(nb_categories=10)
    parser.set_defaults(nb_exemplars=5)
    parser.set_defaults(gpu_num=None)
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    # Set random seeds
    np.random.seed(0)
    tf.set_random_seed(0)
    main()