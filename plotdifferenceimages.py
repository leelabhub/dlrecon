from __future__ import print_function, division

import numpy as np
import os
import h5py
from os import listdir
from os.path import splitext
from random import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def montage(X):    
    m, n, count = X.shape 
    mm = int(np.ceil(np.sqrt(count)))
    nn = mm
    M = np.zeros((mm * m, nn * n))

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count: 
                break
            sliceM, sliceN = j * m, k * n
            M[sliceN:sliceN + n, sliceM:sliceM + m] = X[:, :, image_id]
            image_id += 1
    return M


def generate_difference_images():
    orig_path = '%s/calkan/LeeLabRatImageDatabase/small_test_new/original' % os.environ['OAK']
    regridded_path = '%s/calkan/LeeLabRatImageDatabase/small_test_new/regridded' % os.environ['OAK']
    test_result_path = './test_results'

    os.system('mkdir -p %s/difference_images'%test_result_path)

    files = next(os.walk(orig_path))[2]
    for currfile in files:
        origfile = '%s/%s'%(orig_path,currfile)
        reconfile = '%s/%s'%(test_result_path,currfile)

        with h5py.File(origfile,'r') as h5f:
            origimg = np.array(h5f[h5f.keys()[0]])
        with h5py.File(reconfile,'r') as h5f:
            reconimg = np.array(h5f[h5f.keys()[0]])

        difference = np.abs(origimg-reconimg)
        difference_M = montage(difference)

        output_path = '%s/difference_images/%s'%(test_result_path,currfile[:-3])

        plt.imsave(fname='%s.png'%(output_path), arr=difference_M, cmap='gray', vmin=0, vmax=1, format='png')
        plt.imsave(fname='%s_10x.png'%(output_path), arr=difference_M*10, cmap='gray', vmin=0, vmax=1, format='png')
        plt.imsave(fname='%s_50x.png'%(output_path), arr=difference_M*50, cmap='gray', vmin=0, vmax=1, format='png')
        plt.imsave(fname='%s_100x.png'%(output_path), arr=difference_M*100, cmap='gray', vmin=0, vmax=1, format='png')

        #plt.imsave(fname='%s.png'%(output_path), arr=difference_M, cmap='gray', format='png')
        #plt.imsave(fname='%s_10x.png'%(output_path), arr=difference_M*10, cmap='gray', format='png')
        #plt.imsave(fname='%s_100x.png'%(output_path), arr=difference_M*100, cmap='gray', format='png')



if __name__ == '__main__':
    generate_difference_images()