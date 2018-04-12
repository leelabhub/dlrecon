from __future__ import print_function, division

import numpy as np
import os
import h5py
from os import listdir
from os.path import splitext
from random import shuffle

# from pyimshow3d import imshow3d
# def main():

#     train_path = '/home/zhongnan/Development/keras_unet3d/data/pool/train'
#     test_path = '/home/zhongnan/Development/keras_unet3d/data/pool/test'

#     for x,y in img_generator(test_path,1,(128,192,96)):
#         print(x.shape)
#         imshow3d(x[0,:,:,:,0])


def calc_generator_info(data_path, batch_size):
    files = next(os.walk(data_path))[2]
    nfiles = len(files)
    batches_per_epoch = int(np.ceil(float(nfiles) / batch_size))

    return (files, batches_per_epoch)

def img_generator(data_x_path, data_y_path, batch_size, ksp_size, img_size, shuffle_files=True, verbose=False):
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
                x = np.zeros((final_batch_size,)+ksp_size)
                y = np.zeros((final_batch_size,)+img_size+(1,))
            else:
                x = np.zeros((batch_size,)+ksp_size)
                y = np.zeros((batch_size,)+img_size+(1,))


            for file_cnt in xrange(batch_size):

                file_ind = batch_cnt*batch_size+file_cnt

                if file_ind < nfiles:
                    ksp_path = '%s/%s'%(data_x_path,files[file_ind])
                    with h5py.File(ksp_path,'r') as h5f:
                        ksp = np.array(h5f[h5f.keys()[0]])

                    orig_path = '%s/%s'%(data_y_path,files[file_ind])
                    with h5py.File(orig_path,'r') as h5f:
                        orig = np.array(h5f[h5f.keys()[0]])                    

                    x[file_cnt,...] = ksp.reshape(ksp_size,order='F') # data is flattened in matlab before saving so it was stored as fortran style in memory
                    y[file_cnt,...,0] = orig
            yield(x,y)

# # find unique data regardless of the file prefix
# def calc_generator_info(data_path, batch_size):
#     files = listdir(data_path)
#     unique_filename = {}

#     for file in files:
#         file,_ = splitext(file)
#         if not file in unique_filename:
#             unique_filename[file] = file

#     files = unique_filename.keys()

#     nfiles = len(files)
#     batches_per_epoch = nfiles // batch_size

#     return (files,batches_per_epoch)

# def img_generator(data_path, batch_size, img_size,verbose=False):

#     files, batches_per_epoch = calc_generator_info(data_path, batch_size)
    
#     x = np.zeros((batch_size,)+img_size+(1,))
#     y = np.zeros((batch_size,)+img_size+(1,))

#     while True:
#         shuffle(files)
#         if verbose:
#             print(files)

#         for batch_cnt in xrange(batches_per_epoch):
#             for file_cnt in xrange(batch_size):

#                 file_ind = batch_cnt*batch_size+file_cnt
#                 lt2_path = '%s/%s.lt2'%(data_path,files[file_ind])
#                 with open(lt2_path,'r') as fid:
#                     data = np.fromfile(fid,dtype=np.float32)
#                     ndim = data[0].astype(int)
#                     data_size = data[1:1+ndim].astype(int)
#                     lt2 = np.reshape(data[1+ndim:],data_size,order='F')

#                 ht2_path = '%s/%s.ht2'%(data_path,files[file_ind])
#                 with open(ht2_path,'r') as fid:
#                     data = np.fromfile(fid,dtype=np.float32)
#                     ndim = data[0].astype(int)
#                     data_size = data[1:1+ndim].astype(int)
#                     ht2 = np.reshape(data[1+ndim:],data_size,order='F')
#                 x[file_cnt,...,0] = lt2
#                 y[file_cnt,...,0] = ht2
#             yield(x,y)

# if __name__ == '__main__':
#     main()
