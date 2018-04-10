from __future__ import print_function, division
import numpy as np
from tensorflow.python.client import device_lib

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
            #M[sliceN:sliceN + n, sliceM:sliceM + m] = X[:, :, image_id]
            M[sliceM:sliceM + m, sliceN:sliceN + n] = X[:, :, image_id]
            image_id += 1
    return M


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']