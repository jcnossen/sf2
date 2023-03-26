#%%
from smlmtorch.simflux import SFLocalizer
from smlmtorch.ui.array_view import array_view
import numpy as np

# Configuration for spot detection and 2D Gaussian fitting
cfg = {
    'psf_calib':[1.8,1.8],
   'roisize':10,
   'detection_threshold':2,
   'pattern_frames': [[0,2,4],[1,3,5]],
   'gain': 0.45,
   'offset': 100,
   'pixelsize' : 65 # nm/pixel
}

path = 'C:/data/simflux/sim4_1_MMStack_Pos0.ome.tif'
cfg['psf_calib']=[1.3,1.3]; cfg['pixelsize']=97.5; cfg['detection_threshold']=1.5

sfloc = SFLocalizer(path, **cfg, device='cuda:0')

sfloc.detect_spots(ignore_cache=False, moving_window=True)

ds = sfloc.fit_smlm()
mp = sfloc.estimate_angles(pitch_minmax_nm=[100,1000])
#%%
mp = sfloc.estimate_phases(mp, frame_binsize=4000, accept_percentile=30, iterations=10, verbose=False)
#%%
# Filter by modulation error
ds_filtered = ds[mp.mod_error(ds) < 0.1]
sf_ds = sfloc.fit_simflux(mp, ds_filtered, iterations=50, lambda_=500, normalizeWeights=True, ignore_cache=True)

#%%
sfloc.drift_correct(framesPerBin=20)
#em=sfloc.excitation_matrix(sfloc.sum_ds.frame, sfloc.sum_ds.pos)
#array_view(sfloc.all_rois[1])4
#%%

    
