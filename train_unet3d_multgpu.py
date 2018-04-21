from __future__ import print_function, division

import os
import numpy as np
import pickle

from keras.models import Model
#from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, add
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, add, BatchNormalization, Activation, Conv3DTranspose, Cropping3D
#from keras_contrib.layers.convolutional import Deconvolution3D
from keras import losses
from keras.optimizers import Adam
from keras.utils import plot_model, multi_gpu_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard as tfb

from model_utils import BatchLoss
from csdnn_utils import get_available_gpus
from mri_generator import img_generator, calc_generator_info
from models import unet_3d_model

import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train():
    # set image format to be (N, dim1, dim2, dim3, ch)
    K.set_image_data_format('channels_last')

    # number of gpus
    numGPUs = len(get_available_gpus())
    print('Training on %d GPUs' % numGPUs)

    # training and validation image size
    img_size = (140,140,32,1)
    input_size = (5992*2,32)#(2*5992*32,)

    # training data
    train_path_ksp = '%s/calkan/LeeLabRatImageDatabase/2d_small_training_new/kspace' % os.environ['OAK']
    train_path_orig = '%s/calkan/LeeLabRatImageDatabase/2d_small_training_new/original' % os.environ['OAK']
    train_batch_size = 16
    train_files,train_nbatches = calc_generator_info(train_path_ksp,
                                                     train_batch_size)
    print('[INFO] Training data size: %d, batch size: %d, batches per epochs: %d' % (len(train_files), train_batch_size, train_nbatches))


    # validation data
    valid_path_ksp = '%s/calkan/LeeLabRatImageDatabase/2d_small_validation_new/kspace' % os.environ['OAK']
    valid_path_orig = '%s/calkan/LeeLabRatImageDatabase/2d_small_validation_new/original' % os.environ['OAK']
    valid_batch_size = 16
    valid_files,valid_nbatches = calc_generator_info(valid_path_ksp,
                                                     valid_batch_size)
    print('[INFO] Validation data size: %d, batch size: %d, batches per epochs: %d' % (len(valid_files), valid_batch_size, valid_nbatches))

    # create the unet model (on cpu)
    with tf.device('/cpu:0'):
        model = unet_3d_model(input_size, img_size, batch_norm_order=None)

    # replicate the model on multiple GPUs
    parallel_model = multi_gpu_model(model, gpus=numGPUs)
    parallel_model.compile(optimizer=Adam(lr=1e-5),loss=losses.mean_squared_error)
    parallel_model.callback_model = model
    
    # create folders first
    os.system('mkdir -p ./checkpoints')
    os.system('mkdir -p ./tf_log')

    # model callbacks
    # save model every epoch
    cp_callback = ModelCheckpoint('./checkpoints/weights.{epoch:03d}-{val_loss:.4f}.h5')

    # tensorboard
    #tfb_callback = tfb('./tf_log',histogram_freq=1)

    # train
    bl_callback = BatchLoss()
    hist = parallel_model.fit_generator(img_generator(train_path_ksp, train_path_orig, train_batch_size,
                                             input_size, img_size[0:3]),
                train_nbatches,epochs=60,
                validation_data=img_generator(valid_path_ksp, valid_path_orig, valid_batch_size,
                                                input_size, img_size[0:3]),
                validation_steps=valid_nbatches,
                callbacks=[bl_callback,cp_callback])

    # save history
    with open('./checkpoints/history.pk','w') as fid:
        pickle.dump(hist.history,fid)
    # save batchloss
    with open('./checkpoints/batchloss.pk','w') as fid:
        pickle.dump(bl_callback.losses,fid)

    # plot loss curves
    plt.figure()
    plt.plot(np.arange(1,61), hist.history['loss'], label='train')
    plt.plot(np.arange(1,61), hist.history['val_loss'], label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./checkpoints/epochloss.png')
    plt.savefig('./checkpoints/epochloss.eps')
    plt.clf()


if __name__ == '__main__':
    train()
