import pathlib
import os
import shutil

import numpy as np

from matplotlib import image


def load_data(data_folder, split):

    desktop = pathlib.Path(data_folder)
    data_filenames = [str(item) for item in desktop.rglob('*.png') if item.is_file()]

    np.random.seed(1)
    rand_order = np.random.permutation(len(data_filenames))
    data_filenames_rand = [data_filenames[i] for i in rand_order]

    train_len = int(split[0]*len(data_filenames_rand))
    train_files, rest_files = data_filenames_rand[:train_len], data_filenames_rand[train_len:]

    valid_len = int(split[1]*len(data_filenames_rand))
    valid_files, test_files = rest_files[:valid_len], rest_files[valid_len:]

    return train_files, valid_files, test_files


def reorder_data_for_image_folder(source, dest):

    os.makedirs(dest, exist_ok=True)
    os.makedirs(dest + '/0/', exist_ok=True)
    os.makedirs(dest + '/1/', exist_ok=True)
    for ii, fname in enumerate(source):
        if image.imread(fname).shape == (50, 50, 3):
            if fname[-5] == '1':
                shutil.copy2(fname, dest + '/1/')
            else:
                shutil.copy2(fname, dest + '/0/')
