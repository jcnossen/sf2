#%%
# -*- coding: utf-8 -*-
"""
Simulate spots + modulated intensities, add localization errors
Find back modulation pattern
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import smlmtorch.simflux.pattern_estimator as pe
import numpy as np
from smlmtorch.simflux.simulate import angles_to_mod
import matplotlib.pyplot as plt
from smlmtorch.ui.array_view import array_view
from smlmtorch.multipart_tiff  import MultipartTiffSaver
from smlmtorch import Dataset

from smlmtorch.simflux.simulate import simulate
from smlmtorch.simflux import SFLocalizer

import tqdm
from fastpsf import Context, GaussianPSFMethods, CSplineMethods, CSplineCalibration

from generate_tubules import generate_microtubule_points
from smlmtorch.simflux.localizer_report import LocalizationReporter
from smlmtorch.simflux.pattern_estimator import ModulationPattern
from smlmtorch import struct


sim_roisize = 10  # simulation roisize
roisize = 10 # localization roisize

psf_calib = [1.8,1.8]
psf_label= 'gauss1.8px'

np.random.seed(0)

W = 100
Zrange = [0.1, 0.7]
pts = generate_microtubule_points(W, depth=Zrange[1]-Zrange[0], 
                                        numtubules = 10, 
                                        linedensity=20, margin=0)

on_fraction = 0.001
avg_on_time = 6
nframes = 5000
intensity = 1000

pts[:,2] += Zrange[0]

background = 4 # approximately dna paint background on our dmd setup
path = f'simulated/{psf_label}/sim_gauss2D_tubules_bg{background}_I{intensity}.tif'


pixelsize = 100
pattern_frames = np.array([[0,2,4], [1,3,5]])
mod = angles_to_mod([200, 200], pixelsize, [1, 91], 0.95, pattern_frames)

mp_gt = ModulationPattern(pattern_frames, mod)

cfg = struct(
   psf_calib= psf_calib,
   roisize=roisize,
   detection_threshold=2,
   pattern_frames= pattern_frames,
   gain= 1,
   offset= 0,
   pixelsize= pixelsize,
   zrange= Zrange, # [0.2, 0.6]# [-0.3, 1.3]
   debug_mode=False,
   psf_sigma_binsize = None,
   result_dir='results2'
)


#%% Simulation

psf_calib_sim = psf_calib
with Context() as ctx:
    psf = GaussianPSFMethods(ctx).CreatePSF_XYIBg(sim_roisize, psf_calib_sim, cuda=True)
    simulate(path, mp_gt, psf, 
         pts[:,:2], numframes=nframes, intensity=intensity, width=W, 
         bg=background, avg_on_time=avg_on_time, on_fraction=on_fraction)

#%%
cfg['psf_calib'] = psf_calib
#cfg['psf_calib'] = [2,2]
cfg['roisize'] = roisize
sfloc = SFLocalizer(path, **cfg, device='cuda:0')

sfloc.detect_spots(ignore_cache=False, moving_window=True)
smlm_ds = sfloc.fit_smlm(max_crlb_xy=None, ignore_cache=False)
print(f"numrois: {sfloc.numrois}. #summed_fits: {sfloc.summed_fits.shape[0]}")


#%%
""" Estimate modulation pattern """
if nframes > 600 and W>40:
    mp_est = sfloc.estimate_angles(pitch_minmax_nm=[100,240])
    mp_est = sfloc.estimate_phases(mp_est, frame_binsize=10000, 
                                   accept_percentile=40, iterations=10, verbose=False)
    
else:
    mp_est=mp_gt

#%%
mp_r =pe.estimate_phase_drift(smlm_ds, mp_est, me_threshold=0.1, frame_binsize=50, 
                              initial_step=10, max_iterations=2000, max_step=10000,
                              loss_max=0.05, device='cuda:0', print_step=1)

npat = 2
fig,ax=plt.subplots(npat,1,sharex=True)
for i in range(npat):
    ax[i].plot(np.rad2deg(mp_est.phase_offset_per_frame[i::pattern_frames.size]), label='Estimated (SMLM)')
    #ax[i].plot(np.rad2deg(mp_d.phase_offset_per_frame[i::pattern_frames.size]), label='Estimated (DECODE)')
    ax[i].plot(np.rad2deg(mp_r.phase_offset_per_frame[i::pattern_frames.size]), label='Estimated (Refined)')
    #ax[i].plot(mp_dc.phase_offset_per_frame[i::pattern_frames.size], label='Refined')
    #ax[i].plot(np.rad2deg(mp_gt.phase_offset_per_frame[i::pattern_frames.size]), label='GT')
    ax[i].set_xlabel('Frame')
    ax[i].set_ylabel('Phase [deg]')
plt.legend()
plt.title('Phase drift plotted in phase space')
plt.tight_layout()

#%%
mp_est.mod['depth'] = 0.9

lr = LocalizationReporter(sfloc.sum_ds, sfloc.result_dir, mp_est)
lr.draw_patterns(2, me_threshold=0.1)



#%%

me = mp_est.mod_error(smlm_ds)
me_sel = me < 0.1
#    sf_psf.SetLevMarParams(1e-20, iterations=100)

sf_ds = sfloc.fit_simflux(mp_est, smlm_ds[me_sel], iterations=50, lambda_=500, normalizeWeights=True, ignore_cache=True)
lr.scatterplot([ sfloc.sum_ds, sf_ds ], connected=False, labels=['SMLM', 'SF'], limits=None, s=2)
#simple_ds = sfloc.fit_simple(mp_est , ds[me_sel])


#%%
#edgedist = 40
roi=(43,45),(63,65)
#roi=(46,46),(60,60)
figsize=(8,8)
zoom = 20

datasets=[
    ('smlm', 'SMLM'),
    ('sf', 'SIMFLUX'),
    ]

frc_curves = []
frc_val = []

for fn,label in datasets:
    ds = Dataset.load(sfloc.result_dir + f'{fn}.hdf5')
    ds['pixelsize'] = pixelsize
    
    ds.crlb_filter(0.2)

    zoom=20    
    ds_frc, frc_curve = ds.frc(display=True, zoom=zoom, smooth=10)
    fig,ax=plt.subplots(figsize=figsize)
    ds.crop(*roi, silent=True).renderFigure(axes=ax,title=f'{label} (FRC = {ds_frc:.1f} nm)', 
                                           zoom=zoom, clip_percentile=98, scalebar_nm=500,
                                           scalebar_fontsize=30, cmap='gray_r')
    
    frc_curves.append(frc_curve)
    frc_val.append(ds_frc)
    
    plt.savefig(sfloc.result_dir + f"render-{fn}.png")
    plt.savefig(sfloc.result_dir + f"render-{fn}.svg")
#%%
freq = np.fft.fftfreq(zoom*W)
freq = freq[:W*zoom//2]

fig,ax=plt.subplots(figsize=(7,6))
for i in [0,1]:
    frc = frc_curves[i]
    frc = frc[5:W*zoom//2]
    plt.plot(freq[5:],  frc , label=f'{datasets[i][1]} (FRC={frc_val[i]:.1f} nm)')
plt.legend(fontsize=14)
plt.ylabel('FRC')
plt.xlabel('Spatial freq. [px^-1]')
plt.savefig(sfloc.result_dir+'frc-compare.svg')

#%%
figsize=(6,6)
zoom = 6
render_args = dict(clip_percentile=97, 
                   pixelsize=pixelsize, 
                   scalebar_nm=2000, 
                   scalebar_fontsize=20)

# render ground truth

ds_gt = Dataset(len(pts), 2,(W,W), pixelsize=pixelsize)
ds_gt.pos[:,:2] = pts[:,:2]
ds_gt.photons[:] = 1

fig,ax=plt.subplots(figsize=figsize)
ds_gt.renderFigure(zoom=zoom,axes=ax, **render_args);
ax.set_title('Ground-truth')
plt.savefig(sfloc.result_dir+'render-gt.svg')


# render full view

fig,ax=plt.subplots(figsize=figsize)
ds = Dataset.load(sfloc.result_dir + f'smlm.hdf5')
ds.renderFigure(zoom=zoom, axes=ax, **render_args)
ax.set_title('Overview')
plt.savefig(sfloc.result_dir+'render-full-smlm.svg')


fig,ax=plt.subplots(figsize=figsize)
ds = Dataset.load(sfloc.result_dir + f'sf.hdf5')
ds.renderFigure(zoom=zoom, axes=ax, **render_args)
ax.set_title('Overview')
plt.savefig(sfloc.result_dir+'render-full-sf.svg')


# %%
