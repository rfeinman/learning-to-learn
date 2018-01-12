import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_pickle('cog_bias.pkl')

def make_plot(measure):
  for name in df.name.unique():
    x = df[df.name==name].time.values
    y = df[df.name==name][measure].values

    if measure == 'euclidean_linda':
      y = y/50

    upto = 200000
    y = y[np.where(x < upto)]
    x = x[np.where(x < upto)]

    plt.ion()

    plt.plot(x, y, c='b')
  plt.show()

plt.figure(1)
make_plot('euclidean_linda')
plt.figure(2)
make_plot('training_cost')

import pdb; pdb.set_trace()

# todo: 
#  figure out why inet-sl_subj13_50 accuracy plot looks weird
#  rename training cost to test accuracy (somehow that got mislabelled)
#  truncate to 200k entries
