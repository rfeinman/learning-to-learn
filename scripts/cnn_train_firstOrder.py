from __future__ import division, print_function
import os
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import matplotlib as mpl
mpl.use('Agg')

from learning2learn.models import simple_cnn
from learning2learn.util import load_image_dataset

def run_experiment(nb_categories, nb_exemplars, params):
    # Set random seeds
    np.random.seed(0)
    tf.set_random_seed(0)
    # Create custom TF session if requested
    if params['gpu_options'] is not None:
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=params['gpu_options'])
        )
        K.set_session(sess)
    #data_folder = os.path.realpath('../data/images_generated_old/images_ca0050_ex0014')
    data_folder = os.path.realpath('../data/images_generated')
    X, shapes = load_image_dataset(data_folder, nb_categories, nb_exemplars,
                                    target_size=params['img_size'])
    ohe = OneHotEncoder(sparse=False)
    Y = ohe.fit_transform(shapes.reshape(-1, 1))
    # Now, we separate the train and test sets
    test_inds = [i*(nb_exemplars+1) for i in range(nb_categories)]
    # The train inds are the set difference of all inds and test inds
    train_inds = list(set(range(len(shapes))).difference(test_inds))
    # Build a neural network model and train it with the training set
    print('Training CNN model...')
    scores = []
    for i in range(params['nb_trials']):
        print('Round #%i' % (i+1))
        model = simple_cnn(input_shape=X.shape[1:], nb_classes=Y.shape[-1])
        # We'll provide the test set as 'validation data' merely so we can
        # monitor the trajectory... the network won't be using this data.
        model.fit(X[train_inds], Y[train_inds], epochs=params['nb_epochs'],
                  shuffle=True, validation_data=(X[test_inds], Y[test_inds]),
                  verbose=1, batch_size=params['batch_size'],
                  callbacks=[EarlyStopping(monitor='loss', patience=5)])
        loss, acc = model.evaluate(X[test_inds], Y[test_inds], verbose=0,
                                   batch_size=params['batch_size'])
        scores.append(acc)
    avg_score = np.mean(scores)
    print('\nScore: %0.4f' % avg_score)
    #model.save('../data/model.h5')
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
    main()