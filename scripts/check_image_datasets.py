from __future__ import division, print_function
import os
import pandas as pd

def main():
    data_folder = '../data'
    folders = [file for file in os.listdir(data_folder)
             if file.startswith('images')]
    folders = sorted(folders)
    for folder in folders:
        print('Checking folder %s' % folder)
        assert len(folder) == 20
        nb_categories = int(folder[9:13])
        nb_exemplars = int(folder[16:])
        nb_images = nb_categories*(nb_exemplars+1)
        folder_path = os.path.join(data_folder, folder)
        df_path = os.path.join(folder_path, 'data.csv')
        assert os.path.isfile(df_path)
        df = pd.read_csv(df_path, index_col=0)
        assert df.shape == (nb_images, 3)
        for i in range(nb_images):
            img_path = os.path.join(folder_path, 'img%0.4i.png' % i)
            assert os.path.isfile(img_path)
    print('Success! All tests passed')

if __name__ == '__main__':
    main()