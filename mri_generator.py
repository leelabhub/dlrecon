from __future__ import print_function, division

import numpy as np
import os
import h5py
from os import listdir
from os.path import splitext
from random import shuffle


def calc_generator_info(data_path, batch_size):
    files = next(os.walk(data_path))[2]
    nfiles = len(files)
    batches_per_epoch = int(np.ceil(float(nfiles) / batch_size))

    return (files, batches_per_epoch)

def img_generator(data_x_path, data_y_path, batch_size, img_size, shuffle_files=True, verbose=False):
    files, batches_per_epoch = calc_generator_info(data_x_path, batch_size)
    nfiles = len(files)
    
    #x = np.zeros((batch_size,)+img_size+(1,))
    #y = np.zeros((batch_size,)+img_size+(1,))

    while True:
        if shuffle_files:
            shuffle(files)
        if verbose:
            print(files)

        for batch_cnt in xrange(batches_per_epoch):

            # last batch may be smaller
            if batch_cnt == batches_per_epoch-1:
                final_batch_size = nfiles - batch_cnt*batch_size
                x = np.zeros((final_batch_size,)+img_size+(1,))
                y = np.zeros((final_batch_size,)+img_size+(1,))
            else:
                x = np.zeros((batch_size,)+img_size+(1,))
                y = np.zeros((batch_size,)+img_size+(1,))


            for file_cnt in xrange(batch_size):

                file_ind = batch_cnt*batch_size+file_cnt

                if file_ind < nfiles:
                    zf_path = '%s/%s'%(data_x_path,files[file_ind])
                    with h5py.File(zf_path,'r') as h5f:
                        zf = np.array(h5f[h5f.keys()[0]])

                    orig_path = '%s/%s'%(data_y_path,files[file_ind])
                    with h5py.File(orig_path,'r') as h5f:
                        orig = np.array(h5f[h5f.keys()[0]])                    

                    x[file_cnt,...,0] = zf
                    y[file_cnt,...,0] = orig
            yield(x,y)
