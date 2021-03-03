# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 05:18:38 2016

@author: llab
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

mat_contents = sio.loadmat('phantom_layered_delay_original_hrfamp_4x')
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
plt.ylim((-1,3))
plt.savefig('delay_orig.png',format='png')
plt.savefig('delay_orig.eps',format='eps')


mat_contents = sio.loadmat('phantom_layered_delay_lr_hrfamp_4x')
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
plt.ylim((-1,3))
plt.savefig('delay_nyquist.png',format='png')
plt.savefig('delay_nyquist.eps',format='eps')



mat_contents = sio.loadmat('phantom_layered_delay_fistarec_hrfamp_4x')
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
plt.ylim((-1,3))
plt.savefig('delay_CS.png',format='png')
plt.savefig('delay_CS.eps',format='eps')




mat_contents = sio.loadmat('phantom_layered_delay_cnn_hrfamp_4x')
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
plt.ylim((-1,3))
plt.savefig('delay_CNN.png',format='png')
plt.savefig('delay_CNN.eps',format='eps')

#
#rects = ax.bar(np.arange(N),meanF, align='center',tick_label=labels,yerr=meanFerr)
#plt.title('Mean F-Value',fontweight='bold')
#sns.despine()
#
#
#
#mat_contents = sio.loadmat('sens_fpr')
#sens = mat_contents['sens']
#fpr = mat_contents['fpr']
#
#mat_contents = sio.loadmat('orig_analysis')
#cnr_org = mat_contents['cnr_org']
#cnrerr_org = mat_contents['cnrerr_org']
#meanF_org = mat_contents['meanF_org']
#meanFerr_org = mat_contents['meanFerr_org']
#peakhrf_org = mat_contents['peakhrf_org']
#sens_org = mat_contents['sens_org']
#fpr_org = mat_contents['fpr_org']
#
#
#fpr = np.concatenate((fpr_org[0],fpr[correctinds,0]))
#sens = np.concatenate((sens_org[0],sens[correctinds,0]))
#peakhrf = np.concatenate((peakhrf_org[0],peakhrf[correctinds,0]))
#meanF = np.concatenate((meanF_org[0],meanF[correctinds,0]))
#meanFerr = np.concatenate((meanFerr_org[0],meanFerr[correctinds,0]))
#cnr = np.concatenate((cnr_org[0],cnr[correctinds,0]))
#cnrerr = np.concatenate((cnrerr_org[0],cnrerr[correctinds,0]))
#
#N = 7
#
#fig, ax = plt.subplots()
#rects = ax.bar(np.arange(N),meanF, align='center',tick_label=labels,yerr=meanFerr)
#plt.title('Mean F-Value',fontweight='bold')
#sns.despine()
#
#fig, ax = plt.subplots()
#rects = ax.bar(np.arange(N),cnr, align='center',tick_label=labels,yerr=cnrerr)
#plt.title('CNR',fontweight='bold')
#sns.despine()
#
#fig, ax = plt.subplots()
#rects = ax.bar(np.arange(N),100*peakhrf, align='center',tick_label=labels)
#plt.title('Peak HRF Amplitude (%)',fontweight='bold')
#sns.despine()
#
#fig, ax = plt.subplots()
#rects = ax.bar(np.arange(N),sens, align='center',tick_label=labels)
#plt.title('Sensitivity',fontweight='bold')
#sns.despine()
#
#fig, ax = plt.subplots()
#rects = ax.bar(np.arange(N),fpr, align='center',tick_label=labels)
#plt.title('False Positive Rate',fontweight='bold')
#sns.despine()
#
#mat_contents = sio.loadmat('nrmse')
#nrmse = mat_contents['nrmse']
#
#fig, ax = plt.subplots()
#rects = ax.bar(np.arange(N-1),nrmse, align='center',tick_label=labels[1:])
#plt.title('NRMSE',fontweight='bold')
#sns.despine()