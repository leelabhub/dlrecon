# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 13:08:05 2016

@author: msi
"""

import numpy as np
import scipy as sp
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import seaborn as sns

#sns.set_style("whitegrid", {'axes.grid' : False})
#sns.set_style("dark")


N = 60
x = np.arange(0,N*1,1)

mat_contents = sio.loadmat('phantom_layered_amp_original_hrfamp_4x')
hrf1mean = mat_contents['hrf1mean']
hrf1err = mat_contents['hrf1err']
hrf2mean = mat_contents['hrf2mean']
hrf2err = mat_contents['hrf2err']

fig, ax = plt.subplots()
ax.errorbar(x,100*hrf1mean,yerr=100*hrf1err,color='b',linewidth=2,elinewidth=0.5,label='Layer 1',capsize=5)
ax.errorbar(x,100*hrf2mean,yerr=100*hrf2err,color='r',linewidth=2,elinewidth=0.5,label='Layer 2',capsize=5)
plt.xlabel('Time (s)',fontweight='bold')
plt.ylabel('HRF Amplitude (%)',fontweight='bold')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.title('Ground Truth with Noise',fontweight='bold')
ax.legend(numpoints=1)
plt.xlim((0,60))
plt.ylim((-2,5))
plt.savefig('amp_orig.png',format='png')
plt.savefig('amp_orig.eps',format='eps')



mat_contents = sio.loadmat('phantom_layered_amp_lr_hrfamp_4x')
hrf1mean = mat_contents['hrf1mean']
hrf1err = mat_contents['hrf1err']
hrf2mean = mat_contents['hrf2mean']
hrf2err = mat_contents['hrf2err']
idx1 = mat_contents['idx1']
idx2 = mat_contents['idx2']

fig, ax = plt.subplots()
ax.errorbar(x,100*hrf1mean,yerr=100*hrf1err/np.sqrt(idx1.size),color='b',linewidth=2,elinewidth=0.5,label='Layer 1',capsize=5)
ax.errorbar(x,100*hrf2mean,yerr=100*hrf2err/np.sqrt(idx2.size),color='r',linewidth=2,elinewidth=0.5,label='Layer 2',capsize=5)
plt.xlabel('Time (s)',fontweight='bold')
plt.ylabel('HRF Amplitude (%)',fontweight='bold')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.title('Nyquist Acquisition',fontweight='bold')
ax.legend(numpoints=1)
plt.xlim((0,60))
plt.ylim((-2,5))
plt.savefig('amp_nyquist.png',format='png')
plt.savefig('amp_nyquist.eps',format='eps')



mat_contents = sio.loadmat('phantom_layered_amp_fistarec_hrfamp_4x')
hrf1mean = mat_contents['hrf1mean']
hrf1err = mat_contents['hrf1err']
hrf2mean = mat_contents['hrf2mean']
hrf2err = mat_contents['hrf2err']

fig, ax = plt.subplots()
ax.errorbar(x,100*hrf1mean,yerr=100*hrf1err,color='b',linewidth=2,elinewidth=0.5,label='Layer 1',capsize=5)
ax.errorbar(x,100*hrf2mean,yerr=100*hrf2err,color='r',linewidth=2,elinewidth=0.5,label='Layer 2',capsize=5)
plt.xlabel('Time (s)',fontweight='bold')
plt.ylabel('HRF Amplitude (%)',fontweight='bold')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.title('CS',fontweight='bold')
ax.legend(numpoints=1)
plt.xlim((0,60))
plt.ylim((-2,5))
plt.savefig('amp_CS.png',format='png')
plt.savefig('amp_CS.eps',format='eps')


mat_contents = sio.loadmat('phantom_layered_amp_cnn_hrfamp_4x')
hrf1mean = mat_contents['hrf1mean']
hrf1err = mat_contents['hrf1err']
hrf2mean = mat_contents['hrf2mean']
hrf2err = mat_contents['hrf2err']

fig, ax = plt.subplots()
ax.errorbar(x,100*hrf1mean,yerr=100*hrf1err,color='b',linewidth=2,elinewidth=0.5,label='Layer 1',capsize=5)
ax.errorbar(x,100*hrf2mean,yerr=100*hrf2err,color='r',linewidth=2,elinewidth=0.5,label='Layer 2',capsize=5)
plt.xlabel('Time (s)',fontweight='bold')
plt.ylabel('HRF Amplitude (%)',fontweight='bold')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.title('CNN',fontweight='bold')
ax.legend(numpoints=1)
plt.xlim((0,60))
plt.ylim((-2,5))
plt.savefig('amp_CNN.png',format='png')
plt.savefig('amp_CNN.eps',format='eps')