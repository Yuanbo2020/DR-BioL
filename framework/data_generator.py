
import numpy as np
import os, sys

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

import time
import os
import pickle
from sklearn.utils import shuffle, class_weight
import numbers
from numpy.lib.stride_tricks import as_strided

def hyb_view_as_windows(arr_in, window_shape, step=1):
    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")
    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides
    win_indices_shape = (((np.array(arr_in.shape) - np.array(window_shape))// np.array(step)) + 1)
    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))
    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)

    return arr_out


import json
def save_json(filename, load_dict):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(load_dict, f, ensure_ascii=False)

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
    return load_dict


class DataGenerator_4domains_8species(object):

    def __init__(self, batch_size=64, random_seed=42):

        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)
        self.validate_random_state = np.random.RandomState(random_seed)
        self.test_random_state = np.random.RandomState(random_seed)

        load_time = time.time()

        all_Tacotron_Mel_feature_pickle = 'Mosquito_sound_8species_4domains.pickle'
        all_Tacotron_Mel_feature_pickle_file = os.path.join(os.getcwd(), 'Dataset', all_Tacotron_Mel_feature_pickle)

        print('Loading: ', all_Tacotron_Mel_feature_pickle_file)
        data_dict = self.load_pickle_only_dict(all_Tacotron_Mel_feature_pickle_file)

        self.X_train = data_dict['all_train_x']
        self.y_train = np.array(data_dict['all_train_y_species']).astype(np.int16)
        self.y_train_domain = np.array(data_dict['all_train_y_domain']).astype(np.int16)

        self.X_val = data_dict['all_val_x']
        self.y_val = np.array(data_dict['all_val_y_species']).astype(np.int16)
        self.y_val_domain = np.array(data_dict['all_val_y_domain']).astype(np.int16)

        self.X_test = data_dict['all_test_x']
        self.y_test = np.array(data_dict['all_test_y_species']).astype(np.int16)
        self.y_test_domain = np.array(data_dict['all_test_y_domain']).astype(np.int16)

        print(self.X_train.shape, self.y_train.shape, self.y_train_domain.shape)
        print(self.X_val.shape, self.y_val.shape, self.y_val_domain.shape)
        print(self.X_test.shape, self.y_test.shape, self.y_test_domain.shape)
        print(set(self.y_train), set(self.y_val), set(self.y_test))
        # {0, 1, 2, 3, 4, 5, 6, 7} {0, 1, 2, 3, 4, 5, 6, 7} {0, 1, 2, 3, 4, 5, 6, 7}

        # (42216, 1, 200, 64) (42216,) (42216,)
        # (7652, 1, 200, 64) (7652,) (7652,)
        # (10256, 1, 200, 64) (10256,) (10256,)

        self.class_weights = class_weight.compute_class_weight('balanced',
                                                               classes=np.unique(np.array(self.y_train)),
                                                               y=np.array(self.y_train))
        print('self.class_weights; ', self.class_weights)
        # self.class_weights;  [ 0.37436639  0.82749864  0.3581953   0.92478499  3.43462869  7.97180205
        #   4.63658086 10.02795396  1.94076196]
        ################################################################################
        print('Loading data time: {:.3f} s'.format(time.time() - load_time))

    def load_pickle_only_dict(self, Tacotron_mel_file_path):
        with open(Tacotron_mel_file_path, 'rb') as input_file:
            log_mel_feat = pickle.load(input_file)
        return log_mel_feat

    def save_output_pickle(self, output_file, feat_train):
        with open(output_file, 'wb') as f:
            pickle.dump(feat_train, f, protocol=4)
            print('Saved features to:', output_file)

    def load_pickle(self, file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data

    def generate_training(self):
        audios_num = len(self.X_train)

        audio_indexes = [i for i in range(audios_num)]
        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_x = self.X_train[batch_audio_indexes]

            random_p = np.random.rand()
            if random_p < 0.5:
                batch_x = np.flip(batch_x, axis=2)

            batch_y = self.y_train[batch_audio_indexes]
            batch_y_domain = self.y_train_domain[batch_audio_indexes]

            if len(batch_x) % 2 == 1:
                pointer = 0
                self.random_state.shuffle(audio_indexes)
            else:
                yield batch_x, batch_y, batch_y_domain

    def generate_validate(self, max_iteration=None, data_type=''):
        audios_num = len(self.X_val)
        audio_indexes = [i for i in range(audios_num)]

        self.validate_random_state.shuffle(audio_indexes)
        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0
        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_x = self.X_val[batch_audio_indexes]
            batch_y = self.y_val[batch_audio_indexes]
            batch_y_domain = self.y_val_domain[batch_audio_indexes]

            yield batch_x, batch_y, batch_y_domain


    def generate_testing(self, batch_size=None, max_iteration=None):
        # try:
        #     if self.using_mel:
        #         self.val_all_feature_data
        # except NameError:
        #     var_exists = False
        # else:
        #     var_exists = True
        # print('\n\nvar_exists: ', var_exists)
        #
        # if delete_val and var_exists:
        #     if self.using_mel:
        #         del self.val_all_feature_data
        #         del self.val_x
        #     if self.using_loudness:
        #         del self.val_all_feature_data_loudness
        #         del self.val_x_loudness
        #     gc.collect()
        #     torch.cuda.empty_cache()

        if batch_size is not None:
            self.batch_size = batch_size

        audios_num = len(self.X_test)

        audio_indexes = [i for i in range(audios_num)]
        print('Number of {} audios in {}'.format(len(audio_indexes), ' testing'))

        print(set(self.y_train), set(self.y_val), set(self.y_test))

        iteration = 0
        pointer = 0
        while True:
            if iteration == max_iteration:
                break
            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_x = self.X_test[batch_audio_indexes]
            batch_y = self.y_test[batch_audio_indexes]
            batch_y_domain = self.y_test_domain[batch_audio_indexes]

            yield batch_x, batch_y, batch_y_domain



