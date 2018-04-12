from __future__ import print_function, division

import numpy as np
import pickle

from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, add, BatchNormalization, Activation, Conv3DTranspose, Cropping3D, Reshape, Dense, Concatenate, Lambda
#from keras_contrib.layers.convolutional import Deconvolution3D
from keras import losses
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard as tfb

def unet_3d_model(input_size, image_size, batch_norm_order=None):

    # input size is a tuple of the size of the image
    # assuming channel last
    # input_size = (2*n_enc, dim3) stacked real and imaginary parts for fourier domain data
    # image_size = (dim1, dim2, dim3, ch)
    # batch_norm_order : None (for no batch normalization), 'bn-act' or 'act-bn'.

    nfeatures = [32,64,128,256,512]
    depth = len(nfeatures)

    conv_ptr = []

    # input layer
    inputs = Input(input_size)

    # fully connected layer
    fclayer_shared = Dense(np.prod(image_size[:-2]))
    
    fc_list=[]
    for dim3_idx in xrange(image_size[2]):
        sliced_input = Lambda(lambda x: x[:,dim3_idx])(inputs)
        # apply batch norm
        sliced_input_bn = BatchNormalization()(sliced_input)
        # apply activation
        sliced_input_bn_act = Activation(activation='relu')(sliced_input_bn)

        fc_list.append(fclayer_shared(sliced_input_bn_act))
    
    # concatenate dim3 many tensors of shape (dim1*dim2)
    fc = Concatenate()(fc_list)

    # reshape as an image
    fc_reshaped = Reshape(image_size)(fc)

    # step down convolutional layers
    #pool = inputs
    pool = fc_reshaped
    for depth_cnt in xrange(depth):
    	conv = conv3Dlayer(input_tensor=pool, filters=nfeatures[depth_cnt], kernel_size=(3,3,3),
    						 padding='same', activation='relu', batch_norm_order=None)
        conv = conv3Dlayer(input_tensor=conv, filters=nfeatures[depth_cnt], kernel_size=(3,3,3),
        					 padding='same', activation='relu', batch_norm_order=None)
        conv_ptr.append(conv)

        if depth_cnt < depth-1:
            #pool = conv3Dlayer(input_tensor=conv, filters=nfeatures[depth_cnt], kernel_size=(3,3,3),
            #                 strides=(2,2,2), padding='same', activation='relu', batch_norm_order=None)
            pool = MaxPooling3D(pool_size=(2,2,2),padding='same')(conv)

    # step up convolutional layers
    for depth_cnt in xrange(depth-2,-1,-1):

        #deconv_shape = conv_ptr[depth_cnt].shape.as_list()
        #deconv_shape[0] = None
        #print(deconv_shape)
        #up = concatenate([Deconvolution3D(nfeatures[depth_cnt],(3,3,3),
        #                  activation='relu',
        #                  padding='same',
        #                  strides=(2,2,2),
        #                  output_shape=deconv_shape)(conv),
        #                  conv_ptr[depth_cnt]], 
        #                  axis=4)
        convT = Conv3DTranspose(nfeatures[depth_cnt],(3,3,3),
                          activation='relu',
                          padding='same',
                          strides=(2,2,2))(conv)

        deconv_shape = K.int_shape(conv_ptr[depth_cnt])
        convT_shape = K.int_shape(convT)
        #print(deconv_shape)
        #print(convT_shape)

        # print(convT_shape)
        # print(deconv_shape)
        # if convT_shape[1] != deconv_shape[1]:
        #     convT = Cropping3D(cropping=(1,1,0))(convT)

        convT = Cropping3D(cropping=((0,convT_shape[1]-deconv_shape[1]),(0,convT_shape[2]-deconv_shape[2]),(0,0)))(convT)
        up = concatenate([convT,conv_ptr[depth_cnt]], axis=4)
        #up = concatenate([Conv3DTranspose(nfeatures[depth_cnt],(3,3,3),
        #                  activation='relu',
        #                  padding='same',
        #                  strides=(2,2,2))(conv),
        #                  conv_ptr[depth_cnt]], 
        #                  axis=4)
        

        conv = conv3Dlayer(input_tensor=up, filters=nfeatures[depth_cnt], kernel_size=(3,3,3),
        					 padding='same', activation='relu', batch_norm_order=None)
        conv = conv3Dlayer(input_tensor=conv, filters=nfeatures[depth_cnt], kernel_size=(3,3,3),
        					 padding='same', activation='relu', batch_norm_order=None)

    # combine features
    conv = Conv3D(1, (1,1,1), padding='same', activation=None)(conv)

    # add fc output to the computed residual
    recon = add([fc_reshaped,conv])

    model = Model(inputs=[inputs], outputs=[recon])
    #plot_model(model, to_file='unet3d.png',show_shapes=True)
    
    return model


def conv3Dlayer(input_tensor, filters, kernel_size, strides=(1,1,1), padding='valid', activation=None, batch_norm_order=None):
    # batch_norm_order : None (for no batch normalization), 'bn-act' or 'act-bn'.
    if batch_norm_order == None:
        return Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)(input_tensor)
    elif batch_norm_order == 'act-bn':
        temp = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)(input_tensor)
        return BatchNormalization()(temp)
    elif batch_norm_order == 'bn-act':
        temp = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(input_tensor)
        temp = BatchNormalization()(temp)
        return Activation(activation=activation)(temp)
    else:
        raise ValueError('\'batch_norm_order\' must be either None, \'act-bn\' or \'bn-act\'')