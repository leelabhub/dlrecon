from __future__ import print_function, division

import os
import numpy as np
import h5py

from keras.models import Model
from keras.models import load_model
#from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, add
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, add, BatchNormalization, Activation, Conv3DTranspose, Cropping3D
#from keras_contrib.layers.convolutional import Deconvolution3D
from keras import losses
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard as tfb

from mri_generator import img_generator, calc_generator_info
from csdnn_utils import montage
#from pyimshow3d import imshow3d

import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def test():

    #img_size = (128,192,96,1)
    #img_size = (140,140,32,1)
    img_size = (140,140,32,1)
    input_size = (5992*2,32)
    # find the model checkpoint path corresponding to epoch number 29
    chckpnts = next(os.walk('./checkpoints'))[2]
    chckpnt_path = [item for item in chckpnts if item.startswith('weights.029')][0]

    model = load_model('./checkpoints/%s'%(chckpnt_path))

    # test if the model weights are right
    # layer = model.get_layer('conv3d_1')
    # weight = layer.get_weights()
    # print(np.squeeze(weight[0])[:,0,0,0])

    test_path_zf = '%s/calkan/LeeLabRatImageDatabase/2d_small_test_new/kspace' % os.environ['OAK']
    test_path_orig = '%s/calkan/LeeLabRatImageDatabase/2d_small_test_new/original' % os.environ['OAK']
    test_result_path = './test_results'
    test_result_images_path = './test_results/images'
    test_batch_size = 1


    # create necessary directories
    os.system('mkdir -p %s'%test_result_images_path)

    test_files,ntest = calc_generator_info(test_path_zf, test_batch_size)

    prediction_times = []
    img_cnt = 0
    for x_test, y_test in img_generator(test_path_zf, test_path_orig, test_batch_size,
                                        input_size, img_size[0:3],shuffle_files=False, verbose=True):
        # remove the .h5 part from the image name
        image_name = test_files[img_cnt][:-3]
        print('predicting image %s'%image_name)

        # tmp = np.transpose(np.squeeze(x_test),(2,0,1))
        # imshow3d(tmp)
        start_time = time.time()
        recon = model.predict(x_test,batch_size = 1)
        prediction_time = time.time()-start_time
        print('\t%s seconds'%prediction_time)
        prediction_times.append(prediction_time)
        recon = np.squeeze(recon)
        # imshow3d(np.transpose(recon,(2,0,1)))
        save_name = '%s/%s.h5' %(test_result_path,image_name)
        with h5py.File(save_name,'w') as h5f:
            h5f.create_dataset('recon',data=recon)

        # save the visualization of the result on file
        recon_M = montage(recon)
        
        plt.imsave(fname='%s/%s.png'%(test_result_images_path, image_name), arr=recon_M, cmap='gray', vmin=0, vmax=1, format='png')

        img_cnt += 1
        if img_cnt == ntest:
            print('average forward pass time: %.4f seconds'%np.mean(prediction_times))
            break

if __name__ == '__main__':
    test()
