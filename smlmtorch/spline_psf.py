import torch

import matplotlib.pyplot as plt
import tqdm

import numpy as np
import scipy.io
import tifffile
from smlmtorch.ui.array_view import array_view

from smlmtorch.gaussian_psf import Gaussian2DFixedSigmaPSF
from smlmtorch.detection import SpotDetector
from smlmtorch.adaptive_gd import AdaptiveStepGradientDescent
from smlmtorch.splines import CatmullRomSpline3D, FixedSpline3D

            
class CubicSplinePSF(torch.nn.Module):
    """
    Generates 2D region-of-interests from an underlying 3D cubic spline
    """
    def __init__(self, spline: FixedSpline3D, roisize:int):
        super().__init__()
        self.spline = spline
        self.roisize = roisize
        self.psf_shape = self.spline.shape

    def roi_grids(self, params):
        pr = torch.arange(self.roisize, device=self.device) - self.roisize/2
        py, px = torch.meshgrid(pr,pr,indexing='ij') 
        
        pts = torch.zeros((len(params), self.roisize*self.roisize),3)
        pts[:, :, 0] = px.flatten()[None] + self.shape[2] / 2 - params[:,0,None]
        pts[:, :, 1] = py.flatten()[None] + self.shape[1] / 2 - params[:,1,None]
        pts[:, :, 2] = self.shape[0] / 2 - params[:,2,None]

        return pts
        
    def forward(self, params):
        """
        """
        
        pts = self.roi_grids(params)
        
        self.spline.forward(pts)
        
                                      
    def eval_zstack(self, roisize, zplanes):
        """
        params: [x,y,z,I,bg]
        """
        shape = self.spline.shape
        Y = torch.arange(roisize)
        X = torch.arange(roisize)
        Z = torch.arange(zplanes)
        
        pos = self.positions
        
        Z,Y,X = torch.meshgrid(Z,Y,X, indexing='ij')
        
        # [beads, zplanes, y, x]
        idx = torch.stack((
            (Z[None] - pos[:,2,None,None,None] + shape[0] // 2 - zplanes//2).flatten(), 
            (Y[None] - pos[:,0,None,None,None] + shape[1] // 2 - roisize//2).flatten(),
            (X[None] - pos[:,1,None,None,None] + shape[2] // 2 - roisize//2).flatten()),-1).to(self.device)
        
        result = self.spline(idx)[:,0].reshape((len(pos), zplanes, roisize,roisize))
        result = result / result.sum((2,3),keepdim=True) #* shape[1]*shape[2]
        result = result * self.intensitiesPerPlane[:,:,None,None] + self.backgroundsPerPlane[:,:,None,None]
        
        return result

class CSplineCalibration:
    def __init__(self, z_min, z_max, coefs, z0):
        self.coefs = coefs
        self.z_min = z_min
        self.z_max = z_max
        self.z0 = z0
        self.zrange = [z_min,z_max]

    @classmethod
    def from_smap_file(cls, filename):
        mat = scipy.io.loadmat(filename)
        try:
            spline = mat["SXY"]['cspline'].item()
            coefs = spline['coeff'].item()
            dz = float(spline['dz'].item())
            print(f"dz={dz}")
        except KeyError:
            spline = mat['cspline'].item()
            coefs = spline[0]
            dz = float(spline[1])

        if coefs.dtype == 'O':
            coefs = coefs[0][0]
        
        coefs = np.ascontiguousarray(coefs,dtype=np.float32)
        nx,ny,nz = coefs.shape[:-1]
        coefs = coefs.reshape((nx,ny,nz,4,4,4))
        
        # move the Z slices to the first axis and flip x and y
        # that the coefficients from nature methods PSF have ordering z,x,y, but voxels have x,y,z
        # todo: merge these two
        coefs = np.transpose(coefs, (2,0,1,3,5,4))
        coefs = np.transpose(coefs, (0,1,2,3,4,5))
        
        z0 = spline['z0'].item()[0][0]

        #print(coefs.shape)
        nz = float(coefs.shape[0])
        z_min = -nz * dz * 1e-3 / 2
        z_max = (nz-1) * dz * 1e-3 / 2

        print(f"Z min={z_min}, max={z_max}. step size: {dz:.3f}. z0={z0:.3f}")
        print(f"Voxels X:{coefs.shape[2]}, Y:{coefs.shape[1]}, Z:{coefs.shape[0]}",flush=True)
        return cls(z_min, z_max, coefs, z0)
    

    def save_mat(self, mat_fn):
        from scipy.io import savemat
    
        coefs = self.coefs
    
        nz,ny,nx=coefs.shape[:3]
        coefs = coefs.reshape((nz,ny,nx,4,4,4))
        # move the Z slices to the first axis and flip x and y
        # that the coefficients from nature methods PSF have ordering z,x,y, but voxels have x,y,z
    #    coefs = np.transpose(coefs, (2,0,1,3,5,4)).reshape((nz,ny,nx,64))
        coefs = np.transpose(coefs, (1,2,0,3,4,5)).reshape((ny,nx,nz,64))
        coefs = np.ascontiguousarray(coefs)
        
        zspan = self.z_max-self.z_min
        dz = zspan / nz * 1000
        
        d = {'SXY': {
                 'cspline': {
                     'coeff': coefs,
                     'x0': 0,
                     'y0': 0,
                     'z0': self.z0,
                     'dz': dz
                     }
            }, 'parameters': {
            'dz': dz
            }
        }
            
        savemat(mat_fn, d)

class PoissonWithCutoff(torch.nn.Module):
    def __init__(self, cutoffMin, cutoffMax):
        """
        Cutoff min and cutoff max are multiples of the standard deviation:
        For expected value X, cutoff at the lower side is at X-sqrt(X)*cutoffMin
        """
        super().__init__()
        self.cutoffMin = cutoffMin
        self.cutoffMax = cutoffMax
        
    def forward(self, samples, expval):
        # neg ll = - (k log mu - mu)
        expval = torch.clamp(expval, 1e-9)
        
        minval = expval - torch.sqrt(expval) * self.cutoffMin
        maxval = expval + torch.sqrt(expval) * self.cutoffMax
        
        mask = (samples >= minval) & (samples <= maxval)
        loss = (samples * torch.log(expval) - expval) * mask
        
        return -loss.mean()

def image_edges(images):
    return torch.cat( (images[...,0,:-1], images[...,:-1,-1], images[...,1:,0], images[...,-1,1:]), -1 )
    

header_dtype = [('version', '<i4'), ('dims','<i4', 3), ('zrange', '<f4', 2)]

def save_zstack(zstack, zrange, fn):
    """
    Save a ZStack to a binary file.
    """
    shape = zstack.shape
    with open(fn, "wb") as f:
        version = 1
        np.array([(version, shape, (zrange[0],zrange[-1]))],dtype=header_dtype).tofile(f,"")
        np.ascontiguousarray(zstack,dtype=np.float32).tofile(f,"")


def load_zstack(fn):
    """
    Returns zstack, [zmin, zmax]
    """
    with open(fn, "rb") as f:
        d = np.fromfile(f,dtype=header_dtype,count=1,sep="")
        version, shape, zrange = d[0]
        zstack = np.fromfile(f,dtype='<f4',sep="").reshape(shape)
        return  zstack, zrange
    


class CSplinePSFEstimator:
    def __init__(self, stepsize_nm, device=None):
        self.stepsize_nm =  stepsize_nm
        self.zstacks = []
        self.psf = None
        self.device = device
        
    def add_zstack(self, images_or_fn, threshold, detection_sigma, roisize=30):
        if type(images_or_fn) == str:
            images = torch.tensor( tifffile.imread(images_or_fn).astype(np.float32) )
        else:
            images = images_or_fn
            
        if len(self.zstacks) > 0:
            if self.zstacks[0].shape[0] != images.shape[0]:
                raise ValueError('all zstacks should have same nr of zsteps')
            
        #plt.figure()
        #plt.imshow(images.sum(0))
        
        images = images.to(self.device)
        
        x = torch.tensor([[roisize/2,roisize/2, 1,0]])
        detection_template = Gaussian2DFixedSigmaPSF(roisize, [detection_sigma, detection_sigma]).forward(x)[0][0].to(self.device)
        
        #plt.figure()
        #plt.imshow(detection_template.cpu())
        
        sd = SpotDetector(threshold, images.shape[1:],
                          detection_template, 20, roisize = roisize,
                          bg_filter_size=detection_sigma*5, max_filter_size=detection_sigma*3)
        
        roipos, intensities, rois = sd.forward(images.sum(0))
        print(f"adding zstack with {len(roipos)} beads.")
        
        #for roi in rois:
        #    plt.figure()
        #    plt.imshow(roi.cpu())
            
        zstacks = sd.extract(roipos, images).permute((1,0,2,3))
        self.zstacks.extend(list(zstacks))
        

    def init(self):
        # build an initial psf by just taking means
        
        self.zstacks = torch.stack(self.zstacks)
        shape = self.zstacks.shape
        
        #array_view(self.zstacks)

        self.psf = self.zstacks.mean(0)
        self.psf = self.psf - self.psf.min(1,keepdim=True)[0].min(2,keepdim=True)[0]
        self.psf /= self.psf.sum((1,2),keepdim=True) #* shape[1]*shape[2]
        self.positions = torch.zeros( (shape[0],3) )
        self.backgroundsPerPlane =  image_edges(self.zstacks).mean(-1) #torch.zeros( (shape[0],shape[1]), device=self.device)
        self.intensitiesPerPlane = self.zstacks.sum((2,3))
        
        print(f"mean initial bg: {self.backgroundsPerPlane.mean():.1f} mean initial intensity: {self.intensitiesPerPlane.mean():.1f}")

        self.spline = CatmullRomSpline3D(self.psf[...,None])
        #self.spline = torch.jit.script(self.spline)
        
                                      
    def eval(self, roisize, zplanes):
        """
        params: [x,y,z,I,bg]
        """
        shape = self.psf.shape
        Y = torch.arange(roisize)
        X = torch.arange(roisize)
        Z = torch.arange(zplanes)
        
        pos = self.positions
        
        Z,Y,X = torch.meshgrid(Z,Y,X, indexing='ij')
        
        # [beads, zplanes, y, x]
        idx = torch.stack((
            (Z[None] - pos[:,2,None,None,None] + shape[0] // 2 - zplanes//2).flatten(), 
            (Y[None] - pos[:,0,None,None,None] + shape[1] // 2 - roisize//2).flatten(),
            (X[None] - pos[:,1,None,None,None] + shape[2] // 2 - roisize//2).flatten()),-1).to(self.device)
        
        result = self.spline(idx)[:,0].reshape((len(pos), zplanes, roisize,roisize))
        result = result / result.sum((2,3),keepdim=True) #* shape[1]*shape[2]
        result = result * self.intensitiesPerPlane[:,:,None,None] + self.backgroundsPerPlane[:,:,None,None]
        
        return result
        
    def optimize(self, npass, iterationsPerPass):
        # define the problem
        loss = PoissonWithCutoff(10, 10)
        
        shape = self.zstacks.shape
        """
        W=20
        H=20
        D=80
        eval_roi = self.zstacks[:, shape[1]//2-D//2:shape[1]//2+D//2, 
                                shape[2]//2-H//2:shape[2]//2+H//2,
                                shape[3]//2-W//2:shape[3]//2+W//2]
        """
        
        eval_roi = self.zstacks

        localize_params = [self.positions, self.intensitiesPerPlane] #, self.backgroundsPerPlane]
        psf_params = [self.spline.knots]
        
        def selectParams(localize):
            for p in localize_params:
                p.requires_grad = localize
            for p in psf_params:
                p.requires_grad = not localize
            return localize_params if localize else psf_params

        for j in range(npass):
            if j % 2 == 0:
                print('Optimizing positions')
                params = selectParams(True)
                step=1e-3
                
            else:
                print('Optimizing PSF')
                params = selectParams(False)
                step=1e-3
                
            optimizer = AdaptiveStepGradientDescent(params, initial_step=step)

            def loss_():
                optimizer.zero_grad()
                zstack_expval = self.eval(shape[2],shape[1])

                l = ((eval_roi - zstack_expval)**2).mean()#loss(eval_roi, zstack_expval)
                l.backward()
                return l
        
            N  = iterationsPerPass
            with tqdm.tqdm(total=N) as pb:
                for i in range(N):
                    lossval = optimizer.step(loss_)
                    #print(f"{lossval:.3f}. positions={self.positions}")
                    if optimizer.finished:
                        print(f"finished i={i}")
                        break

                    pb.update(1)                    
                    pb.set_description(f"{lossval:.5f}")
                    
            print(f"intensities: {self.intensitiesPerPlane.mean(1)}. bg: {self.backgroundsPerPlane.mean(1)}")

    def to_calib(self):
        fs = FixedSpline3D.from_catmull_rom(self.spline) 
        zstack = self.spline.get_knots()[:,:,:,0]
        nsteps = zstack.shape[0]
        
        return CSplineCalibration(z_min = -self.stepsize_nm / 1000 * nsteps//2, 
                           z_max = self.stepsize_nm / 1000 * (nsteps//2 - 1),
                           coefs = fs.coeffs.cpu().detach().numpy())

#%%
if __name__ == '__main__':
    #zstack_fn = 'C:/data/dmdsim/beads_pos4_astig_20nm_3_crop0.tif'
    #zstack_fn = [ 'C:/data/multicolor/er_astig_tirf/bead4_3_20nm_right.tif', 
    #             'C:/data/multicolor/er_astig_tirf/bead4_2_20nm_right.tif']

    bead_fn = [  f'C:/data/dmdsim/sf-astig/astig_beads/bead{i+1}.tif' for i in range(3) ] 
    psf_fn = 'C:/data/dmdsim/sf-astig/astig_beads/gd_psf.mat'
    zstack_fn = 'C:/data/dmdsim/sf-astig/astig_beads/gd_psf.zstack'

#%%    
    #zstack_fn = [ 'C:/data/multicolor/er_astig_tirf/bead4_3_20nm_right.tif', 
    #             'C:/data/multicolor/er_astig_tirf/bead4_2_20nm_right.tif']
    
    estimator = CSplinePSFEstimator(20, torch.device('cuda'))
    for fn in bead_fn:
        estimator.add_zstack(fn, threshold = 1.2, detection_sigma = 5)

    calib = CSplineCalibration.from_smap_file('C:/data/dmdsim/sf-astig/astig_psf.mat')
    
    array_view(calib.coefs[:,:,:,0,0,0])

    estimator.init(calib)
    
    #array_view(estimator.eval(20,40))
    
    estimator.optimize(10, 100)
    val = estimator.eval(30, 180)
    
    #%%
    #array_view(torch.cat((val, estimator.zstacks),-1))
    calib = estimator.to_calib()
    
    save_zstack(estimator.spline.knots[:,:,:,0].detach().cpu(), calib.zrange, zstack_fn)
    calib.save_mat(psf_fn)

#%%
    zstack,zrange = load_zstack(zstack_fn)
    
    

    #%%
    psf_fn = 'C:/data/sim-decode/psf2D/psf.mat'
    #psf_fn = 'C:/data/dmdsim/sf-astig/astig_psf.mat'
    #    psf_fn = 'C:/data/multicolor/er_astig_tirf/psf_astig.mat'
    #psf_fn = 'C:/data/decode examples/experimental_data_workflow/spline_calibration_3dcal.mat'
    
    #psf_fn = 'C:/data/dmdsim/sf-astig/astig_beads/gd_psf.mat'
    #calib = CSplineCalibration.from_smap_file('C:/data/dmdsim/sf-astig/astig_psf.mat')
    calib = CSplineCalibration.from_smap_file(psf_fn)
    #calib.save_mat(psf_fn)
    shape = calib.coefs.shape
    dev=torch.device('cpu')

    c = calib.coefs
    c = c.reshape((*c.shape[:3], 4,4,4))
    c = torch.tensor(c).to(dev)
    fs = FixedSpline3D(c[...,None])

    Y, X = torch.meshgrid(torch.arange(shape[1]), 
                   torch.arange(shape[2]), indexing = 'ij')

    Y = Y.flatten()
    X = X.flatten()

    pts = torch.stack( ( X*0+100, Y, X ) , -1).float()
    #array_view(zstack)

    #fs = torch.jit.script(fs)
    
    slices = []
    #for sz in torch.linspace(-5,5, 200 ):
    for sz in torch.linspace(-50,50, 200 ):
        pts = torch.stack( ( X*0+shape[0]//2+sz, Y, X) , -1).float().to(dev)
    
        with torch.no_grad():
            
            values, deriv = fs.deriv(pts)
            deriv = deriv.reshape((3, shape[1], shape[2]))
            values = values.reshape((shape[1],shape[2]))
                                    
            #slices.append(.cpu().numpy())#value2.cpu().numpy())
            slices.append(torch.cat( (values[None], deriv)).cpu().numpy() )

    array_view(slices)
#%%
    if False:
        for i in tqdm.trange(10000):
            with torch.no_grad():
                #values, deriv = spl.deriv(pts)
                values = fs.forward(pts.repeat(100,1))
                
                