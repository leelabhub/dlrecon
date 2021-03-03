import numpy as np
import scipy as sp
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## AMPLITUDE

# mat_contents = sio.loadmat('phantom_layered_amp_original_imghrfamp_4x')
# baseimg = mat_contents['baseimg']
# overlay = mat_contents['overlay']

# fig,ax = plt.subplots()
# plt.imshow(baseimg, 'gray', interpolation='none')
# overlay[overlay==0] = np.nan;
# plt.imshow(overlay, 'jet', interpolation='none', vmin=0, vmax=0.04, alpha=1)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# cbar = plt.colorbar(ticks=[0,0.04])
# cbar.ax.set_yticklabels([0,4], fontweight='bold')
# cbar.set_label('Peak HRF Amplitude (%)', fontweight='bold', rotation=270)
# plt.savefig('map_amp_orig.png', format='png')
# plt.savefig('map_amp_orig.eps', format='eps')



# mat_contents = sio.loadmat('phantom_layered_amp_lr_imghrfamp_4x')
# baseimg = mat_contents['baseimg']
# overlay = mat_contents['overlay']

# fig,ax = plt.subplots()
# plt.imshow(baseimg, 'gray', interpolation='none')
# overlay[overlay==0] = np.nan
# plt.imshow(overlay, 'jet', interpolation='none', vmin=0, vmax=0.04, alpha=1)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# cbar = plt.colorbar(ticks=[0,0.04])
# cbar.ax.set_yticklabels([0,4], fontweight='bold')
# cbar.set_label('Peak HRF Amplitude (%)', fontweight='bold', rotation=270)
# plt.savefig('map_amp_nyquist.png', format='png')
# plt.savefig('map_amp_nyquist.eps', format='eps')


# mat_contents = sio.loadmat('phantom_layered_amp_fistarec_imghrfamp_4x')
# baseimg = mat_contents['baseimg']
# overlay = mat_contents['overlay']

# fig,ax = plt.subplots()
# plt.imshow(baseimg, 'gray', interpolation='none')
# overlay[overlay==0] = np.nan
# plt.imshow(overlay, 'jet', interpolation='none', vmin=0, vmax=0.04, alpha=1)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# cbar = plt.colorbar(ticks=[0,0.04])
# cbar.ax.set_yticklabels([0,4], fontweight='bold')
# cbar.set_label('Peak HRF Amplitude (%)', fontweight='bold', rotation=270)
# plt.savefig('map_amp_CS.png', format='png')
# plt.savefig('map_amp_CS.eps', format='eps')


# mat_contents = sio.loadmat('phantom_layered_amp_cnn_imghrfamp_4x')
# baseimg = mat_contents['baseimg']
# overlay = mat_contents['overlay']

# fig,ax = plt.subplots()
# plt.imshow(baseimg, 'gray', interpolation='none')
# overlay[overlay==0] = np.nan
# plt.imshow(overlay, 'jet', interpolation='none', vmin=0, vmax=0.04, alpha=1)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# cbar = plt.colorbar(ticks=[0,0.04])
# cbar.ax.set_yticklabels([0,4], fontweight='bold')
# cbar.set_label('Peak HRF Amplitude (%)', fontweight='bold', rotation=270)
# plt.savefig('map_amp_CNN.png', format='png')
# plt.savefig('map_amp_CNN.eps', format='eps')


# DELAY

mat_contents = sio.loadmat('phantom_layered_delay_original_imghrfdelay_4x')
baseimg = mat_contents['baseimg']
overlay = mat_contents['overlay'].astype(np.float32)

fig,ax = plt.subplots()
plt.imshow(baseimg, 'gray', interpolation='none')
overlay[overlay==0] = np.nan
plt.imshow(overlay, 'jet', interpolation='none', vmin=0, vmax=17, alpha=1)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
cbar = plt.colorbar(ticks=[0,17])
cbar.ax.set_yticklabels([0,17], fontweight='bold')
cbar.set_label('Time to Half Peak Value (s)', fontweight='bold', rotation=270)
plt.savefig('map_delay_orig.png', format='png')
plt.savefig('map_delay_orig.eps', format='eps')

mat_contents = sio.loadmat('phantom_layered_delay_lr_imghrfdelay_4x')
baseimg = mat_contents['baseimg']
overlay = mat_contents['overlay'].astype(np.float32)

fig,ax = plt.subplots()
plt.imshow(baseimg, 'gray', interpolation='none')
overlay[overlay==0] = np.nan
plt.imshow(overlay, 'jet', interpolation='none', vmin=0, vmax=17, alpha=1)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
cbar = plt.colorbar(ticks=[0,17])
cbar.ax.set_yticklabels([0,17], fontweight='bold')
cbar.set_label('Time to Half Peak Value (s)', fontweight='bold', rotation=270)
plt.savefig('map_delay_nyquist.png', format='png')
plt.savefig('map_delay_nyquist.eps', format='eps')

mat_contents = sio.loadmat('phantom_layered_delay_fistarec_imghrfdelay_4x')
baseimg = mat_contents['baseimg']
overlay = mat_contents['overlay'].astype(np.float32)

fig,ax = plt.subplots()
plt.imshow(baseimg, 'gray', interpolation='none')
overlay[overlay==0] = np.nan
plt.imshow(overlay, 'jet', interpolation='none', vmin=0, vmax=17, alpha=1)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
cbar = plt.colorbar(ticks=[0,17])
cbar.ax.set_yticklabels([0,17], fontweight='bold')
cbar.set_label('Time to Half Peak Value (s)', fontweight='bold', rotation=270)
plt.savefig('map_delay_CS.png', format='png')
plt.savefig('map_delay_CS.eps', format='eps')

mat_contents = sio.loadmat('phantom_layered_delay_cnn_imghrfdelay_4x')
baseimg = mat_contents['baseimg']
overlay = mat_contents['overlay'].astype(np.float32)

fig,ax = plt.subplots()
plt.imshow(baseimg, 'gray', interpolation='none')
overlay[overlay==0] = np.nan
plt.imshow(overlay, 'jet', interpolation='none', vmin=0, vmax=17, alpha=1)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
cbar = plt.colorbar(ticks=[0,17])
cbar.ax.set_yticklabels([0,17], fontweight='bold')
cbar.set_label('Time to Half Peak Value (s)', fontweight='bold', rotation=270)
plt.savefig('map_delay_CNN.png', format='png')
plt.savefig('map_delay_CNN.eps', format='eps')