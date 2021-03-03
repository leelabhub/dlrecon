import numpy as np
import scipy as sp
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

labels = ['GT with Noise','Gridding','CS','CNN']
colors = ['b','k','g','r']
numrealizations = 5 # total number of realizations

phantomname = 'phantom_sh'
subpathnames = ['original_hrfcomp.mat',
		'regridded_hrfcomp.mat',
		'fistarec_hrfcomp.mat',
		'cnn_hrfcomp.mat']

#subfoldername = 'hrfcomp'

numpoints = 60 # number of points

x = np.arange(0,numpoints)


N = len(subpathnames)



for perc in [1,2,3,4]:

	hrfmean = np.zeros((N, numrealizations, numpoints))
	hrferr = np.zeros((N, numrealizations, numpoints))

	fig,ax = plt.subplots()
	

	for subpathidx in range(N):
		subpathname = subpathnames[subpathidx]
		for realizationidx in range(numrealizations):
				currfile = '%s_%dperc_r%d_%s'%(phantomname, perc, realizationidx+1, subpathname)
				print(currfile)
		
				mat_contents = sio.loadmat(currfile)

				hrfmean[subpathidx, realizationidx, :] = mat_contents['hrfmean'].flatten()
				hrferr[subpathidx, realizationidx, :] = mat_contents['hrferr'].flatten()

		
		meanhrf = np.mean(hrfmean[subpathidx], axis=0)
		
		meanerr = np.mean(hrferr[subpathidx], axis=0)

		#meanerr2 = np.zeros(meanerr.shape)
		#meanerr2[4::5] = meanerr[4::5]

		ax.errorbar(x, 100*meanhrf, yerr=100*meanerr, color=colors[subpathidx], label=labels[subpathidx],linewidth=1,elinewidth=0.5,capsize=1.5)


	plt.xlabel('Time (s)',fontweight='bold')
	plt.ylabel('HRF Amplitude (%)',fontweight='bold')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	plt.title('%d%% HRF'%perc,fontweight='bold')
	ax.legend(loc="upper right")
	plt.xlim((0,60))
	plt.ylim((-1,4))
	plt.savefig('hrf_%dperc.png'%perc,format='png')
	plt.savefig('hrf_%dperc.eps'%perc,format='eps')


# fig, ax = plt.subplots()
# ax.errorbar(x,100*hrfmeans_orig[:,0],yerr=100*hrferrs_orig[:,0],color='b',linewidth=2,elinewidth=0.5,label='Original',capsize=5)
# ax.errorbar(x,100*hrfmeans[:,0],yerr=100*hrferrs[:,0],color='r',linewidth=2,elinewidth=0.5,label='CNN',capsize=5)
# plt.xlabel('Time (s)',fontweight='bold')
# plt.ylabel('HRF Amplitude (%)',fontweight='bold')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# plt.title('4% HRF',fontweight='bold')
# ax.legend(numpoints=1)
# plt.xlim((0,60))
# plt.ylim((-4,10))
# plt.savefig('%s/hrf_4perc.png'%subfoldername,format='png')
# plt.savefig('%s/hrf_4perc.eps'%subfoldername,format='eps')


# fig, ax = plt.subplots()
# ax.errorbar(x,100*hrfmeans_orig[:,1],yerr=100*hrferrs_orig[:,0],color='b',linewidth=2,elinewidth=0.5,label='Original',capsize=5)
# ax.errorbar(x,100*hrfmeans[:,1],yerr=100*hrferrs[:,0],color='r',linewidth=2,elinewidth=0.5,label='CNN',capsize=5)
# plt.xlabel('Time (s)',fontweight='bold')
# plt.ylabel('HRF Amplitude (%)',fontweight='bold')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# plt.title('6% HRF',fontweight='bold')
# ax.legend(numpoints=1)
# plt.xlim((0,60))
# plt.ylim((-4,10))
# plt.savefig('%s/hrf_6perc.png'%subfoldername,format='png')
# plt.savefig('%s/hrf_6perc.eps'%subfoldername,format='eps')


# fig, ax = plt.subplots()
# ax.errorbar(x,100*hrfmeans_orig[:,2],yerr=100*hrferrs_orig[:,0],color='b',linewidth=2,elinewidth=0.5,label='Original',capsize=5)
# ax.errorbar(x,100*hrfmeans[:,2],yerr=100*hrferrs[:,0],color='r',linewidth=2,elinewidth=0.5,label='CNN',capsize=5)
# plt.xlabel('Time (s)',fontweight='bold')
# plt.ylabel('HRF Amplitude (%)',fontweight='bold')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# plt.title('8% HRF',fontweight='bold')
# ax.legend(numpoints=1)
# plt.xlim((0,60))
# plt.ylim((-4,10))
# plt.savefig('%s/hrf_8perc.png'%subfoldername,format='png')
# plt.savefig('%s/hrf_8perc.eps'%subfoldername,format='eps')


# fig, ax = plt.subplots()
# ax.errorbar(x,100*hrfmeans_orig[:,3],yerr=100*hrferrs_orig[:,0],color='b',linewidth=2,elinewidth=0.5,label='Original',capsize=5)
# ax.errorbar(x,100*hrfmeans[:,3],yerr=100*hrferrs[:,0],color='r',linewidth=2,elinewidth=0.5,label='CNN',capsize=5)
# plt.xlabel('Time (s)',fontweight='bold')
# plt.ylabel('HRF Amplitude (%)',fontweight='bold')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# plt.title('10% HRF',fontweight='bold')
# ax.legend(numpoints=1)
# plt.xlim((0,60))
# plt.ylim((-4,10))
# plt.savefig('%s/hrf_10perc.png'%subfoldername,format='png')
# plt.savefig('%s/hrf_10perc.eps'%subfoldername,format='eps')