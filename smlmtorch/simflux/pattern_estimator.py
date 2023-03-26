# -*- coding: utf-8 -*-
"""
Find 2D SIM patterns in localization data
@author: jelmer
"""
import numpy as np
import matplotlib.pyplot as plt
import torch.fft
import torch
import os
from scipy.interpolate import InterpolatedUnivariateSpline
from .dataset import SFDataset
from torch import Tensor
import tqdm
import pickle
from smlmtorch.util.locs_util import array_split_minimum_binsize
from smlmtorch.splines import CatmullRomSpline1D
from smlmtorch import struct

ModulationType = np.dtype([('k', '<f4', (3,)), 
                            ('depth','<f4'),
                            ('phase','<f4'),
                            ('relint','<f4')
                            ])



class ModulationPattern:
    def __init__(self, pattern_frames, mod=None, moving_window = True, 
                 phase_offset_per_frame =None):
        """ 
        N = number of patterns
        nax = number of axes
        mod: [ N, [kx,ky,kz,depth,phase,relint]  ]
        """
        assert mod is None or len(mod.shape) == 1
        #assert phase_per_frame is None or len(phase_per_frame.shape) == 1

        self.pattern_frames = pattern_frames
        self.mod = mod.copy() if mod is not None else None
        self.phase_offset_per_frame = phase_offset_per_frame.copy() if phase_offset_per_frame is not None else None
        self.moving_window = moving_window 
        
    def save(self, fn):
        with open(fn, "wb") as s:
            pickle.dump(self.__dict__, s)
            
    @staticmethod
    def load(fn):
        with open(fn, "rb") as s:
            d = pickle.load(s)
        return ModulationPattern(**d)
        
    def clone(self):
        return ModulationPattern(self.pattern_frames,
                                 self.mod,
                                 self.moving_window,
                                 self.phase_offset_per_frame)
       
    def mod_at_frame(self, start_frame_ix, frame_window):
        
        frame_ix = start_frame_ix[:, None] + np.arange(frame_window)[None]
        mod = self.mod[frame_ix % len(self.mod)].copy()
        
        # modify phase if interpolation data is available
        if self.phase_offset_per_frame  is not None:
            mod['phase'] += self.phase_offset_per_frame [frame_ix]

        return mod
    
    def pos_phase(self, pos, frame):
        """
        Calculate the phase at given position and frame
        """
        pos = np.array(pos)
        mod = self.mod_at_frame(frame,1)[:,0]
        return ((mod['k'] * pos).sum(1) - mod['phase']) % (np.pi*2)
    
    def add_drift(self, drift_px):
        if self.phase_offset_per_frame is None:
            self.phase_offset_per_frame = np.zeros((len(drift_px)))
        
        for i in range(len(self.mod)):
            k = self.mod[i]['k'][:drift_px.shape[1]]
            self.phase_offset_per_frame[i:len(drift_px):len(self.mod)] += (drift_px[i::len(self.mod)]*k[None]).sum(1)
            
        self._center_phase_offsets()
        
    def pattern_phase_offsets(self, pat_ix):
        return self.phase_offset_per_frame[pat_ix::len(self.mod)]
        
    def _center_phase_offsets(self):
        """ Make sure mean of phase offsets is zero """
        
        for i in range(len(self.mod)):
            ofs = self.phase_offset_per_frame[i::len(self.mod)].mean()
            self.mod['phase'][i] += ofs
            self.phase_offset_per_frame[i::len(self.mod)] -= ofs
            
    def drift_trace_px(self):
        """
        Returns a XY drift trace in pixel units
        """
        
        nframes = len(self.phase_offset_per_frame)
        k = self.mod['k'][np.arange(nframes)%len(self.mod)]
        # phase = k * pos
        # pos = phase / k
        
        # normalized direction
        k_len = np.sqrt( (k**2).sum(1,keepdims=True) )
        k_norm = k / k_len
        return (self.phase_offset_per_frame[:,None] * k_norm / k_len)[:,:2] 
    
    def pitch(self):
        k = self.mod['k']
        k_len = np.sqrt((k**2).sum(1))
        pitch = 2*np.pi/k_len
        
        return pitch    
    
    @property
    def k(self):
        return self.mod['k']
    
    def __str__(self):
        pf = ""
        if self.phase_offset_per_frame  is not None:
            pf = f"{len(self.phase_offset_per_frame )} frames of phase data."
            
        axinfo = ', '.join([f'{np.rad2deg(ang):.1f} deg' for ang in self.angles])
        return f"{len(self.mod)} patterns, axes: [{axinfo}]. {pf}"

    def __repr__(self):
        return self.__str__()

    def compute_excitation(self, start_frame_ix, frame_window, pos):
        """
        
        """
        dims = pos.shape[1]
        mod = self.mod_at_frame(np.array(start_frame_ix), frame_window)
        k = mod['k'][:,:,:dims]
        spot_phase = (k * pos[:,None]).sum(-1) - mod['phase']
        exc = mod['relint']*(1+mod['depth']*np.sin(spot_phase))
        return exc
    
    def mod_error(self, ds):
        exc = self.compute_excitation(ds.frame, len(self.mod), ds.pos)
        sumI = np.sum(ds.ibg[:, :, 0],1)

        normI = ds.ibg[:, :, 0] / sumI[:,None]
        moderr = normI-exc
        return moderr.max(1)
    
    @property
    def angles(self):
        k = self.mod['k']
        return np.arctan2(k[:,1],k[:,0])
    
    @property
    def depths(self):
        return self.mod['depth']
    
    @depths.setter
    def depths(self, v):
        self.mod['depth'] = v
    
    @property
    def angles_deg(self):
        return np.rad2deg(self.angles)
    
    def print_info(self, pixelsize=None, reportfn=print):
        k = self.mod['k']
        phase = self.mod['phase']
        depth = self.mod['depth']
        ri = self.mod['relint']
        
        for i in range(len(self.mod)):
            reportfn(f"Pattern {i}: kx={k[i,0]:.4f} ky={k[i,1]:.4f} Phase {phase[i]*180/np.pi:8.2f} Depth={depth[i]:5.2f} "+
                   f"Rel.Int={ri[i]:5.3f} ")
    
        for ang in range(len(self.pattern_frames)):
            pat=self.pattern_frames[ang]
            d = np.mean(depth[pat])
            phases = phase[pat]
            shifts = (np.diff(phases[-1::-1]) % (2*np.pi)) * 180/np.pi
            shifts[shifts > 180] = 360 - shifts[shifts>180]
            
            with np.printoptions(precision=3, suppress=True):
                reportfn(f"Angle {ang} shifts: {shifts} (deg) (patterns: {pat}). Depth={d:.3f}")


# Curve-fit y around the peak to find subpixel peak x
def quadraticpeak(y, x=None, npts=7, plotTitle=None):
    if x is None:
        x = np.arange(len(y))
    xmax = np.argmax(y)
    W = int((npts + 1) / 2)
    window = np.arange(xmax - W + 1, xmax + W)
    window = np.clip(window, 0, len(x) - 1)
    coeff = np.polyfit(x[window], y[window], 2)

    if plotTitle:
        plt.figure()
        plt.plot(x[window], y[window], label="data")
        sx = np.linspace(x[xmax - W], x[xmax + W], 100)
        plt.plot(sx, np.polyval(coeff, sx), label="fit")
        plt.legend()
        plt.title(plotTitle)

    return -coeff[1] / (2 * coeff[0])




@torch.jit.script
def _dft(xyI, kx,ky):
    p = xyI[:,0,None,None] * kx[None, None, :] + xyI[:,1,None,None] * ky[None, :, None]
    r = ( torch.cos(p) * xyI[:,2,None,None] ).sum(0)
    i = ( torch.sin(p) * xyI[:,2,None,None] ).sum(0)

    return torch.complex(r,i)    

def torch_dft(xyI, kx,ky, device, batch_size=None):
    if batch_size is None:
        batch_size = 20000

    with torch.no_grad():
        xyI = torch.tensor(xyI)
        kx = torch.tensor(kx, device=device)
        ky = torch.tensor(ky, device=device)

        return torch.stack([ _dft(s.clone(), kx, ky) for s in 
                                torch.split(xyI.to(device), batch_size)]).sum(0)


def cuda_dft(xyI, kx,ky, useCuda=True):
    from fastpsf import Context
    from fastpsf.simflux import SIMFLUX
    import numpy as np
    
    KX,KY = np.meshgrid(kx,ky)
    klist = np.zeros((len(kx)*len(ky),2),dtype=np.float32)
    klist[:,0] = KX.flatten()
    klist[:,1] = KY.flatten()
    with Context() as ctx:
        return SIMFLUX(ctx).DFT2D(xyI, klist, useCuda=useCuda).reshape((len(ky),len(kx)))
    


def render_sr_image_per_pattern(xy, I, img_shape, sr_factor):
    h,w=img_shape
    #img = np.zeros((h*sr_factor,w*sr_factor), dtype=np.float32)
    
    H,W = h*sr_factor, w*sr_factor
    img, xedges, yedges = np.histogram2d(xy[:,1]*sr_factor, xy[:,0]*sr_factor, 
                                         bins=[H,W], range=[[0, H], [0, W]], weights=I )
    
    return img
    

def estimate_angle_and_pitch_dft(xy, I, frame_ix, freq_minmax, imgshape,
                                  dft_peak_search_range, file_ix=0, fft_timebins=1,
                                  debug_images=False, sr_zoom=6, device=None, 
                                  results_dir=None, dft_batch_size=50000, fft_img_cb=None):
    """
    xy: [N, 2]
    I: [N, num phase steps]
    frame_ix: [N]
    """
    h,w=imgshape
    
    from smlmtorch.util.locs_util import indices_per_frame
    ipf = indices_per_frame(frame_ix)
    framebins = np.array_split(np.arange(len(ipf)), fft_timebins)
    
    ft_sum = torch.zeros((h*sr_zoom,w*sr_zoom),device=device)
    ft_smpimg  = ft_sum * 0
    for i in range(len(framebins)):
        ix = np.concatenate( [ipf[f] for f in framebins[i]] )
        # render all patterns
        smpimg = render_sr_image_per_pattern(xy[ix], I[ix].sum(1), imgshape, sr_zoom)
        ft_smpimg += torch.abs(torch.fft.fft2(torch.from_numpy(smpimg).to(device)))
        
        for ep in range(I.shape[1]):
            img = render_sr_image_per_pattern(xy[ix], I[ix][:,ep], imgshape, sr_zoom)
            
            #if results_dir is not None:
            #    plt.imsave(f"{results_dir}/ep{ep}_sr_render.png", img/np.max(img))
    
            ft_img = torch.fft.fft2(torch.from_numpy(img).to(device))
            ft_sum += torch.abs(ft_img)
         
    ft_smpimg /= torch.sum(ft_smpimg)
    ft_smpimg = torch.fft.fftshift(ft_smpimg)
    ft_sum = torch.fft.fftshift(ft_sum)
    
    freq = torch.fft.fftshift( torch.fft.fftfreq(h*sr_zoom) )*sr_zoom*2*np.pi
    XFreq, YFreq = torch.meshgrid(freq,freq, indexing='xy')
    Freq = torch.sqrt(XFreq**2+YFreq**2)

    mask = (Freq>freq_minmax[0]) & (Freq<freq_minmax[1])
    ft_smpimg[~mask] = 0
    ft_sum[~mask] = 0
        
    ft_sum /= ft_sum.sum()
    ft_sum = ft_sum - ft_smpimg

    #print(f"Max pixel frequency: {freq[0]:.2f}")
    
    ft_sum = ft_sum.cpu().numpy()

    if debug_images:
        saved_img = ft_sum*1
        plt.imsave(f"imgs/{file_ix}pattern-FFT.png", saved_img)

    #ft_sum = ft_sum / np.sum(ft_sum) - ft_smpimg
    
#        plt.imsave(self.outdir + f"pattern-{pattern_indices}-FFT-mask.png", mask)
    
    if results_dir:
        plt.imsave(f"{results_dir}/pattern-{file_ix}-FFT-norm.png", ft_sum)
        
    if fft_img_cb is not None:
        fft_img_cb(ft_sum, file_ix)

    max_index = np.argmax(ft_sum)
    max_indices = np.unravel_index(max_index, ft_sum.shape)
    
    W=10
    if results_dir:
        plt.imsave(f'{results_dir}/pattern-{file_ix}-FFT-peak.png', 
               ft_sum[max_indices[0]-W:max_indices[0]+W,
                      max_indices[1]-W:max_indices[1]+W])
    
    #print(f'Freq peak value:{ft_sum[max_indices]}')
            
    peak_yx = freq[list(max_indices)]
    peak_xy = _find_dft2_peak(peak_yx [[1,0]], xy, I, dft_peak_search_range, 
                              file_ix, device, results_dir, batch_size=dft_batch_size)

    return peak_xy    
    

def _find_dft2_peak(xy, loc_xy, I, dft_peak_search_range=0.02, file_ix=0, device=0, 
                    results_dir=None, batch_size=None):
    def compute_peak_img(x,y,S):
        kxrange = np.linspace(x-S, x+S, 50)
        kyrange = np.linspace(y-S, y+S, 50)

        """
        img = torch.zeros((len(kyrange),len(kxrange)))
        for ep in range(I.shape[1]):
            xyI = np.concatenate((loc_xy, I[:,[ep]]), 1)

            sig = torch_dft(xyI, kxrange, kyrange, device=device,batch_size=batch_size).cpu()
            img += torch.abs(sig**2)
        img = img.numpy()
         """
        img = np.zeros((len(kyrange),len(kxrange)))
        for ep in range(I.shape[1]):
            xyI = np.concatenate((loc_xy, I[:,[ep]]), 1)

            sig = cuda_dft(xyI, kxrange, kyrange, useCuda=True)
            img += np.abs(sig**2)
         
        peak = np.argmax(img)
        peak = np.unravel_index(peak, img.shape)
        kx_peak = quadraticpeak(img[peak[0], :], kxrange, npts=11, plotTitle=None)#='X peak')
        ky_peak = quadraticpeak(img[:, peak[1]], kyrange, npts=11, plotTitle=None)#='Y peak')

        return img, kx_peak, ky_peak
    
    peakimg, kxpeak, kypeak= compute_peak_img(*xy, dft_peak_search_range)
    if results_dir:
        plt.imsave(f"{results_dir}/pattern-{file_ix}-DFT-peak1.png", peakimg)
    
    peakimg2, kxpeak2, kypeak2 = compute_peak_img(kxpeak, kypeak, dft_peak_search_range)
    if results_dir:
        plt.imsave(f"{results_dir}/pattern-{file_ix}-DFT-peak2.png", peakimg2)

    #print(f"KXPeak={kxpeak:.2f},{kypeak:.2f},{kxpeak2:.2f},{kypeak2:.2f}")
    return kxpeak2, kypeak2



def _estimate_phase_and_depth_bin(pos, I, frame_ix, k, iterations, accept_percentile, verbose=True):
    nsteps = I.shape[1]

    def process_step(ep):
        sumI = I.sum(1)

        intensity = sumI
        spotPhaseField = (k[None] * pos[:,:len(k)]).sum(1)
        
        basefreq = torch.tensor([-1, 0, 1],device=pos.device)
        weights = torch.ones(len(intensity), device=pos.device)

        depth_trace=[]
        phase_trace=[]
        for it in range(iterations):
            # DFT on modulated and unmodulated data
            f = [ torch.sum(weights * I[:,ep] * torch.exp(-1j * spotPhaseField * k)) for k in basefreq ]
            B = [ torch.sum(weights * sumI * torch.exp(-1j * spotPhaseField * k)) for k in basefreq - 1]
            A = [ torch.sum(weights * sumI * torch.exp(-1j * spotPhaseField * k)) for k in basefreq ]
            C = [ torch.sum(weights * sumI * torch.exp(-1j * spotPhaseField * k)) for k in basefreq + 1]
            
            # Solve Ax = b
            nrows = len(basefreq)
            M = torch.zeros((nrows,3), dtype=torch.cfloat)
            b = torch.zeros(nrows, dtype=torch.cfloat)
            for i in range(nrows):
                M[i] = torch.tensor(  [B[i], C[i], A[i] ] )
                b[i] = f[i]
                            
            # Actually x[1] is just the complex conjugate of x[0], 
            # so it seems a solution with just 2 degrees of freedom is also possible
#            x, residual, rank, s = np.linalg.lstsq(M,b,rcond=None)
            x = torch.linalg.solve(M,b)
            b,c,a = x
           
            depth = torch.real(2*torch.abs(b)/a)
            phase = -torch.angle(b*1j)
            relint = torch.real(2*a)/2  

            q = relint * (1+depth*torch.sin( (k[None]*pos[:,:len(k)]).sum(1) - phase))
            
            normI = I[:,ep] / sumI
            errs = (normI-q)**2
            moderr = errs.mean().cpu().numpy()
            
            errs_ = errs.cpu().numpy()
            median_err = np.percentile(errs_, accept_percentile)# median(errs)
            weights = errs < median_err
            
            depth_trace.append(depth)
            phase_trace.append(phase)
            
        if verbose:
            f_min = int(frame_ix.min().numpy())
            f_max = int(frame_ix.max().numpy())
            print(f"Frame {f_min, f_max}, EP={ep}. Depth estimation per iteration: ", ','.join([f"{d:.2f}" for d in depth_trace]))
            #phase_trace = np.diff(np.array(phase_trace))
            #print("Phase estimation per iteration: ", ','.join([f"{d:.2f}" for d in phase_trace]))

#        print(f"ep{ep}. depth={depth}, phase={phase}")
        return phase, depth, relint, moderr

    with torch.no_grad():
        return np.array([process_step(ep) for ep in range(nsteps)])


def indices_per_frame(frame_indices):
    if len(frame_indices) == 0: 
        numFrames = 0
    else:
        numFrames = torch.max(frame_indices)+1
    frames = [[] for i in range(numFrames)]
    for k in range(len(frame_indices)):
        frames[frame_indices[k]].append(k)
    for f in range(numFrames):
        frames[f] = torch.tensor(frames[f], dtype=torch.int64)
    return frames
    
        
def _estimate_angles(xy, intensities, frame_ix, imgshape, pattern_frames, pitch_minmax_nm,
                      pixelsize, moving_window, **kwargs):

    freq_minmax = 2*np.pi / (torch.tensor(pitch_minmax_nm[::-1].copy()) / pixelsize)
    npat = pattern_frames.size
    
    assert intensities.shape[1] == npat
    
    k = np.zeros((npat, 2))
    ri = np.zeros(npat)
    
    for i, pf in enumerate(pattern_frames):
        #I = torch.stack( [ intensities[torch.arange(len(xy)), ( pf[j] + npat//2 - frame_ix) % npat] for j in range(len(pf)) ], -1 )
        if moving_window:
            I = np.stack( [ intensities[np.arange(len(xy)), ( pf[j] - frame_ix) % npat] for j in range(len(pf)) ], -1 )
        else:
            I = intensities[:,pf]
        
        k_i = estimate_angle_and_pitch_dft(xy = xy, 
                                     I = I,
                                     frame_ix = frame_ix,
                                     freq_minmax = freq_minmax,
                                     imgshape = imgshape, 
                                     dft_peak_search_range = 0.03,
                                     file_ix = i,
                                     **kwargs)

        if k_i[np.abs(k_i).argmax()] < 0:
            k_i = -np.array(k_i)
        kx,ky = k_i
        
        angle_rad = np.arctan2(ky,kx)
        freq_px= np.sqrt(kx**2+ky**2)
        pitch_px = 2*np.pi/freq_px
        k[pf] = [kx,ky]
        ri[pf] = I.sum()
            
        print(f"Angle: {np.rad2deg(angle_rad):.2f}, Pitch: {pitch_px * pixelsize:.2f} nm")

    ri /= ri.sum()

    return k,ri


def _estimate_phases(xyz:Tensor, intensities:Tensor, k, frame_ix:Tensor, numframes, pattern_frames, frame_binsize, 
                    device, moving_window, fig_callback = None, **kwargs):

    
    ix_per_bin = array_split_minimum_binsize(np.array(frame_ix), binsize = frame_binsize, use_tqdm=True)
    
    if len(ix_per_bin) == 2:
        raise ValueError(f'frame_binsize should be set to result in either 1 or more than 2 frame bins (now {len(ix_per_bin)}, for {len(frame_ix)} spots.)')
            
    npat = np.array(pattern_frames).size
    nframes = int(frame_ix.max()) + npat
    #print(f"frame ix max: {frame_ix.max()}. nframes={nframes}")
    #frame_bin_ix = (frame_ix / nframes * frame_bins).long()
    #ix_per_bin = indices_per_frame(frame_bin_ix)
    
    num_angles = pattern_frames.shape[0]
    num_steps = pattern_frames.shape[1]
    npat = pattern_frames.size
    
    pdre = np.zeros((num_angles, num_steps, len(ix_per_bin), 4))

    with tqdm.tqdm(total=len(pattern_frames)*len(ix_per_bin)) as pb:
        for i, pf in enumerate(pattern_frames):
            
            # TODO: After stack the loc_ix dont point to the right spots
            if moving_window:
                I = torch.stack( [ intensities[torch.arange(len(xyz)), ( pf[j] #+ npat//2 
                                                                    - frame_ix.long()) % npat] for j in range(len(pf)) ], -1 )
            else:
                I = intensities[:,pf]
                
            for j in range(len(ix_per_bin)):
                loc_ix = ix_per_bin[j]
    
                pdre[i, :, j] = _estimate_phase_and_depth_bin(xyz[loc_ix], I[loc_ix], frame_ix[loc_ix], k[i], **kwargs)
                pb.update(1)
                
    pdre[:,:,:,0] = np.unwrap(pdre[:,:,:,0],axis=-1)
    phases = pdre[:,:,:,0]
    depths = pdre[:,:,:,1]

    # store interpolated phase for every frame
    frame_bin_t = [frame_ix[ipb].float().mean() for ipb in ix_per_bin]
    phase_interp = np.zeros((num_angles, num_steps, numframes))
    for i in range(len(pattern_frames)):
        for j,p_ix in enumerate(pattern_frames[i]):
            if len(ix_per_bin)>1:
                spl = InterpolatedUnivariateSpline(frame_bin_t, phases[i,j], k=2)
                phase_interp[i,j] = spl(np.arange(numframes))
            else:
                phase_interp[i,j] = phases[i,j]
                
    if fig_callback is not None:        
        fig,axes=plt.subplots(len(pattern_frames),squeeze=False)
        for i, pf in enumerate(pattern_frames):
            for j in range(len(pf)):
                l=axes[i][0].plot(frame_bin_t, phases[i,j], '.', label=f'step {j}')
                axes[i][0].plot(phase_interp[i,j],'--', color=l[0].get_color())
            axes[i][0].set_title(f"angle {i} phases")
            axes[i][0].legend()
        plt.tight_layout()
        fig_callback('phases')

        fig,axes=plt.subplots(len(pattern_frames),squeeze=False)
        for i, pf in enumerate(pattern_frames):
            for j in range(len(pf)):
                axes[i][0].plot(depths[i,j],'.-', label=f'step {j}')
            axes[i][0].set_title(f"angle {i} depths")
            axes[i][0].legend()
        plt.tight_layout()
        fig_callback('depths')
        
    return depths, phases, phase_interp

def estimate_kz(xyz: Tensor, intensities: Tensor, kxy, z_pitch_range, frame_ix, frame_bins, 
                    fig_callback = None):
        
    nframes = frame_ix.max()+1
    frame_bin_ix = (frame_ix / nframes * frame_bins).long()
    ix_per_bin = indices_per_frame(frame_bin_ix)
    
    numsteps = intensities.shape[1]
    
    pdre = np.zeros((len(z_pitch_range), numsteps, frame_bins, 4))
    
    for i,z_pitch in tqdm.tqdm(enumerate(z_pitch_range),total=len(z_pitch_range)):
                        
        for j in range(frame_bins):
            loc_ix = ix_per_bin[j]
            
            kz = 2*np.pi/z_pitch
            k = torch.tensor([kxy[0], kxy[1], kz], device=xyz.device)

            pdre[i, :, j] = _estimate_phase_and_depth_bin(xyz[loc_ix], intensities[loc_ix], frame_ix[loc_ix], k, 
                                                iterations=1, 
                                                accept_percentile=100, # redundant if iterations=1 
                                                verbose=False)
            
            #moderr[i, j] = _moderror(xyz[loc_ix], intensities[loc_ix], frame_ix[loc_ix])
    
    if fig_callback is not None:
        fig,ax=plt.subplots(2,sharex=True)

        kz_vs_depth = pdre[:,:,:,1].mean((1,2))        
        kz_peak = quadraticpeak(kz_vs_depth, z_pitch_range)
        if kz_peak > z_pitch_range[-1]: 
            kz_peak = None
        else:          
            print(f"depth based kz estimate: {kz_peak:.4f}")

        
        kz_vs_moderr = pdre[:,:,:,3].mean((1,2))
        kz_peak_moderr = quadraticpeak(-kz_vs_moderr, z_pitch_range)
        if kz_peak_moderr  > z_pitch_range[-1]: 
            kz_peak_moderr = None
        else:        
            print(f"mod error based kz estimate: {kz_peak_moderr:.4f}")
        
        ax[0].plot(z_pitch_range, kz_vs_depth, label=f'Peak {kz_peak:.2f} um' if kz_peak is not None else None)
        ax[1].plot(z_pitch_range, kz_vs_moderr, label=f'Peak {kz_peak_moderr:.2f} um' if kz_peak_moderr is not None else None)
        ax[1].set_xlabel('Z pitch [um]')
        ax[0].set_ylabel('Depth')
        ax[1].set_ylabel('Moderr')
        ax[0].legend()
        ax[1].legend()
        
        fig_callback('kz')
    
    return pdre

def simple_sine_fit(I : Tensor):
    """
    Sine fits as done in the SIMPLE paper, resulting in modulation depth per spot and per axis.
    Pattern is defined as I = A/2 * (1+cos(2pi * (x+phase))) + b
    """
    
    _x = torch.sqrt(
        (I[:,0]-I[:,1])**2 + 
        (I[:,0]-I[:,2])**2 +
        (I[:,1]-I[:,2])**2)
    
    ampl = 2*np.sqrt(2)/3 * _x
    
    phase = -1/torch.pi*torch.arctan((-2*I[:,0]+I[:,1]+I[:,2]+np.sqrt(2)*_x)/
                                     (np.sqrt(3)*(I[:,1]-I[:,2])))
    
    bg = 1/3*(I[:,0]+I[:,1]+I[:,2]-np.sqrt(2)*_x)
    
    return ampl, phase/(2*torch.pi), bg


    
    
def estimate_angles(pitch_minmax_nm, ds: SFDataset, pattern_frames, result_dir, 
                    moving_window=True, **kwargs):

    k, ri = _estimate_angles(ds.pos[:,:2], ds.ibg[:,:,0], 
                            ds.frame, ds.imgshape,
                            pitch_minmax_nm=pitch_minmax_nm,
                    pattern_frames = pattern_frames, 
                    results_dir = result_dir, 
                    moving_window=moving_window,
                    pixelsize = ds['pixelsize'], **kwargs)
    
    mod = np.zeros(pattern_frames.size, dtype=ModulationType)
    mod['k'][:,:2] = k
    mod['relint'] = ri

    #print(k, ri)
    return ModulationPattern(pattern_frames, mod = mod, moving_window = moving_window)
    
        
def estimate_phases(ds: SFDataset, mp: ModulationPattern, frame_binsize, numframes=None,
                    accept_percentile = 50, iterations = 10, verbose=True, device='cpu',
                    fig_callback = None):
    """
    estimate phase, depth, and a coarse estimate of phase drift
    frame_binsize: minimum number of localizations in each phase estimation bin
    """
    
    dims = ds.pos.shape[1]
    
    if numframes is None:
        numframes = ds.numFrames + mp.pattern_frames.size - 1
    
    k = torch.tensor( mp.mod['k'][mp.pattern_frames[:,0]][:,:dims] )
    
    depth_per_bin, phase_per_bin, phase_interp = _estimate_phases(
        torch.from_numpy(ds.pos), 
        intensities = torch.from_numpy(ds.ibg[:,:,0]),
        k=k,
        frame_ix = torch.from_numpy(ds.frame), 
        pattern_frames = mp.pattern_frames, 
        frame_binsize = frame_binsize, 
        device = torch.device(device) if type(device)==str else device,
        accept_percentile = accept_percentile,
        iterations = iterations, 
        moving_window = mp.moving_window,
        fig_callback = fig_callback, 
        numframes = numframes,
        verbose = verbose)

    npat = mp.pattern_frames.size
    depths = np.zeros((npat,))
    phases = np.zeros((npat,))
    for i, pf in enumerate(mp.pattern_frames):
        depths[pf] = depth_per_bin[i].mean(1)
        phases[pf] = phase_per_bin[i].mean(1)            
                
    mp.mod['phase'] = phases
    mp.mod['depth'] = depths

    # phase interp stores the interpolated phases for all patterns seperately,
    # also at frames where that particular pattern is not being used.
    mp.phase_offset_per_frame = np.zeros((phase_interp.shape[-1]))
    npat = mp.pattern_frames.size
    for i,pf in enumerate(mp.pattern_frames):
        for j in range(len(pf)): 
            mp.phase_offset_per_frame[pf[j]::npat] = phase_interp[i, j, pf[j]::npat] - phases[pf[j]]
        
    mp._center_phase_offsets()
    
    return mp



def estimate_phase_drift(ds: SFDataset, mp: ModulationPattern, frame_binsize, 
                         device=None, me_threshold=0.1, loss_max=0.01, max_iterations=2000, print_step=100, **kwargs):
    """
    Estimate phase drift, keeping same-angle phase steps and depth fixed
    """
    ds = ds[mp.mod_error(ds) < me_threshold]
    
    print(f'size of remaining ds:{len(ds)}')
    #ds = ds[ds.frame % len(mp.mod) == 0]
    
    mp = mp.clone()    
    dev = torch.device(device) if device is not None else None
    intensities = torch.from_numpy(ds.ibg[:,:,0]).to(dev)
        
    from smlmtorch.util.locs_util import indices_per_frame
    frame_bins = indices_per_frame(ds.frame // len(mp.mod)// frame_binsize)# array_split_minimum_binsize(ds.frame, frame_binsize)
    
    for axis, pf in enumerate(mp.pattern_frames):
        if mp.moving_window:
            # Rotate the 1st axis depending on frame index
            I = torch.stack( [ intensities[torch.arange(len(ds.pos)), ( pf[j] #+ npat//2 
                - ds.frame) % mp.pattern_frames.size] for j in range(len(pf)) ], -1 )
        else:
            I = intensities[:,pf]

        drift = torch.from_numpy(mp.phase_offset_per_frame[pf[0]::len(mp.mod)]).to(dev)
        
        if frame_binsize == 1:
            param = [drift]
            drift.requires_grad = True
        
            frame_ix = torch.from_numpy(ds.frame // len(mp.mod)).long()
            get_drift = lambda ix: drift[ix]
        else:
            knots = torch.zeros((len(frame_bins),1))
            for i,fb in enumerate(frame_bins):
                knots[i,0] = drift[i*frame_binsize].mean()
            spl = CatmullRomSpline1D(knots.to(dev))
            param = spl.parameters()
            get_drift = lambda ix: spl(ix / frame_binsize)[:,0]
            frame_ix = torch.from_numpy(ds.frame // len(mp.mod)).to(dev)

        from smlmtorch.adaptive_gd import AdaptiveStepGradientDescent
        
        optim = AdaptiveStepGradientDescent(param, **kwargs)

        phases = torch.from_numpy(mp.mod['phase'][pf]).to(dev)
        depth = mp.mod['depth'][pf].mean()
        
        #I = torch.from_numpy(intensities[:,pf]).to(dev)
        normI = I/I.sum(1,keepdims=True)
        k = mp.mod['k'][:,:ds.pos.shape[1]]
        xyz_phase = torch.from_numpy((ds.pos * k[pf[0]][None]).sum(1)).to(dev)
        
        for i in range(max_iterations):
            def loss_fn():
                optim.zero_grad()
                loc_phase = xyz_phase[:,None] - (get_drift(frame_ix)[:,None] + phases[None])
                mu = 1 + depth * torch.sin(loc_phase)
                mu = mu / mu.sum(1,keepdims=True)
                loss = torch.clamp( (mu-normI)**2, max=loss_max).mean()
                
                loss.backward()
                return loss
        
            with torch.no_grad():
                loss = optim.step(loss_fn)
                
            if i % print_step == 0:
                loss =  loss.detach().cpu().numpy()
                print(f"Iteration={i}. Loss: {loss}. Stepsize: {optim.stepsize}")
                
            if optim.finished:
                break
            
        for i,f in enumerate(pf):
            d = get_drift(torch.arange(len(mp.phase_offset_per_frame[f::len(mp.mod)])).to(dev)).detach().cpu().numpy()
            mp.phase_offset_per_frame[f::len(mp.mod)] = d #  all steps are assigned the same offset
                    
    return mp
    

def estimate_phase_drift_cv(ds, mp : ModulationPattern, result_dir,frame_binsize, **kwargs):
    """
    estimate_phase_drift, but also estimate estimation precision using 2-fold cross validation.
    ds: Dataset
    """ 
    mask = np.random.randint(2, size=len(ds))
    iterations = 3
    
    pat = []
    for j, ds_half in enumerate([ds[mask==1], ds[mask==0]]):
        mp_r = mp.clone()  # iteratively update (modulation error will update the list of localizations)
        mp_r.phase_offset_per_frame[:] = 0
        for i in range(iterations):
            print()
            mp_r = estimate_phase_drift(ds_half, mp_r, frame_binsize, **kwargs)
            
        pat.append(mp_r)
        mp_r.save(result_dir + f"pattern-cv{i}.pickle")

    rms_err = np.sqrt( np.mean( (pat[0].phase_offset_per_frame - pat[1].phase_offset_per_frame)**2 ) )
    print(f"RMS Phase error: {np.rad2deg(rms_err):.5f} deg.")
    
    # Compute final:
    mp_r = mp.clone()
    for i in range(iterations):
        mp_r = estimate_phase_drift(ds, mp_r, frame_binsize, **kwargs)
    
    mp_r.save(result_dir + "pattern.pickle")
            
    return struct(pattern=mp_r, org_mp=mp, cv=pat, 
                  rms_err=rms_err, frame_binsize=frame_binsize)


def plot_phase_drift(result, time_range, mp_gt = None):
    cv = result.cv
    rms_err_rad = result.rms_err

    L=1000
    t = np.arange(L)

    pf = result.pattern.pattern_frames
    pat_ix = [ pf_i[0] for pf_i in pf ]
    fig,ax=plt.subplots(len(pat_ix),1,sharex=True)
    for i in range(len(pat_ix)):
        #ax[i].plot(np.rad2deg(.phase_offset_per_frame[pat_ix[i]::pf.size]), label='Estimated (SMLM)')
        ax[i].plot(t,np.rad2deg(cv[0].phase_offset_per_frame[i::pf.size][s]), '-',label='Crossval. bin 1')
        ax[i].plot(t,np.rad2deg(cv[1].phase_offset_per_frame[i::pf.size][s]), '-',label='Crossval. bin 2')

        ax[i].plot(t,np.rad2deg(result.org_mp.phase_offset_per_frame[pat_ix[i]::pf.size]), label='Estimated')
        ax[i].plot(t,np.rad2deg(result.pattern.phase_offset_per_frame[pat_ix[i]::pf.size]), label='Estimated (Refined)')
        #ax[i].plot(mp_dc.phase_offset_per_frame[i::pattern_frames.size], label='Refined')
        if mp_gt is not None:
            ax[i].plot(np.rad2deg(mp_gt.phase_offset_per_frame[pat_ix[i]::pf.size]), label='GT')
        ax[i].set_xlabel('Frame')
        ax[i].set_ylabel('Phase [deg]')

        k = result.pattern.mod['k'][i]
        ang = np.rad2deg(np.arctan2(k[1],k[0]))

        ax[i].set_title(f'Estimated phase drift for pattern {i} [angle {ang:.1f} deg]. \nRMS Phase err: {np.rad2deg(rms_err_rad):.2f} deg. Frames/bin={result.frame_binsize}')
    plt.legend()

    #ax[i].set_title(f'Estimated phase drift for pattern {i} [angle {ang:.1f} deg]. \nRMS Phase err: {np.rad2deg(rms_err_rad):.2f} deg. Frames/bin={phase_drift_binsize}')
    plt.suptitle('Estimated phase drift')
    plt.legend()
    plt.tight_layout()
    

    
def merge_estimates(ds1: SFDataset, ds2: SFDataset, ndims:int =2):
    
    assert len(ds1) == len(ds2)
    
    combined = ds1[:]
    
    total_fi = ds1.crlb.pos[:,:ndims]**-2 + ds2.crlb.pos[:,:ndims]**-2
    combined.crlb.pos[:,:ndims] = total_fi ** -0.5
    
    combined.pos[:,:ndims] = (
        ds1.pos[:,:ndims] * ds1.crlb.pos[:,:ndims] ** -2 / total_fi + 
        ds2.pos[:,:ndims] * ds2.crlb.pos[:,:ndims] ** -2 / total_fi
    )
    
    return combined


