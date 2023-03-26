
from typing import List, Tuple, Union, Optional, Final
from torch import Tensor
import torch

class BatchAdaptiveGradientDescent(torch.nn.Module):
    def __init__(self, loss_model : torch.nn.Module, param_range, initial_step:float, step_multiplier:float=2, param_step_factor=None):
        super().__init__()
        self.loss = loss_model
        self.param_range = param_range
        self.initial_step = initial_step
        self.step_multiplier = float(step_multiplier)
        self.param_step_factor = param_step_factor 
        
    def _forward_batch(self, input_data, initial_params, iterations, const_=None):
        stepsize = torch.ones((len(input_data),), device=input_data.device)*self.initial_step
        reject_count = torch.zeros((len(input_data),), dtype=torch.int32, device=input_data.device)
        
        params = initial_params*1

        params = torch.clamp(params, min=self.param_range[0],max=self.param_range[1])
        loss, loss_grad = self.loss(input_data,params,const_)
        best_loss = loss
        best_loss = loss
        best_params = params
        best_params_grad = loss_grad

        for i in range(iterations):
            if i > 0:
                loss, loss_grad = self.loss(input_data,params,const_)

            #assert torch.isnan(loss_grad).sum()==0
            
            is_improved = (loss < best_loss).int()
            best_loss = torch.minimum(loss,best_loss)
            stepsize *= self.step_multiplier**(2*is_improved-1)
            
            reject_count = (reject_count+1) * (1-is_improved)
                
            # if improved, we store the current parameters as the best ones
            best_params = params * is_improved[:,None] + best_params * (1-is_improved[:,None])
            best_params_grad = loss_grad * is_improved[:,None] + best_params_grad * (1-is_improved[:,None])
            
            if self.param_step_factor is not None:
                step = stepsize[:,None] * best_params_grad * self.param_step_factor[None]
            else:
                step = stepsize[:,None] * best_params_grad
            
            #print(f"params: {params}.\nstep={step}")
            params = best_params - step
            #print(f"best_params_grad: {best_params_grad}")
            params = torch.clamp(params, min=self.param_range[0],max=self.param_range[1])
            
            
            #print(is_improved, stepsize, loss, params)
            
        return params, reject_count

    def forward(self, input_data, initial_params, batch_size=10000, gd_iterations=100,const_=None):
        reject_counts = torch.zeros((len(input_data),), dtype=torch.int32, device=input_data.device)
        params = initial_params*1
        
        ix = torch.arange(len(input_data), device=input_data.device)

        iterations = 0        
        while len(ix)>0 and iterations<40:
            print (f"# items: {len(ix)}")
            batches = torch.tensor_split(ix, len(ix)//batch_size+1)
            
            for batch_ix in batches:
                bc = const_[batch_ix] if const_ is not None else None
                r, rc = self._forward_batch(input_data[batch_ix], 
                                            params[batch_ix],
                                            gd_iterations,
                                            const_=bc)

                #print(rc)
                reject_counts[batch_ix] = rc
                params[batch_ix] = r
        
            ix = (reject_counts < 5).nonzero()[:,0]
            iterations += 1
 
        return params, reject_counts, iterations
        

class PoissonLikelihoodLoss(torch.nn.Module):
    def __init__(self, model, epsilon=1e-10):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
    
    def forward(self, samples, x, const_:Optional[Tensor]=None):
        samples = torch.flatten(samples, start_dim=1)
        mu, jac = self.model(x, const_)
        mu = torch.flatten(mu, 1)
        jac = torch.flatten(jac, start_dim = 1, end_dim = -2)
        
        # p(x | d) = 
        
        mu = torch.clamp(mu, min=self.epsilon)
        ll = samples * torch.log(mu) - mu
        ll_grad = (samples/mu)[...,None] * jac - jac
        
        #assert torch.isnan(ll_grad).sum()==0
        
        return -ll.sum(1), -ll_grad.sum(1)


if __name__ == '__main__':
    from gaussian_psf import Gaussian2DFixedSigmaPSF
        
    roisize=15
    
    model = Gaussian2DFixedSigmaPSF(roisize, 1.5)

    dev = torch.device('cpu')
    N=3
    params = torch.ones((N,4),device=dev) * torch.tensor([[4,4,20,10]], device=dev)
    true_params = params * (0.2+torch.rand(size=(N,4), device=dev)*0.4)
    params = params * (0.8+torch.rand(size=(N,4), device=dev)*0.4)
    
    mu, jac = model(true_params)
    
    param_range = torch.tensor([[0,0,0,1],[roisize-1,roisize-1,1e8,1000]], device=dev)

    #array_view(jac.permute((0,3,1,2)))
    
    smp = torch.poisson(mu)
    #array_view(smp)
    
    loss_fn = PoissonLikelihoodLoss(model)
    #loss, loss_grad = loss_fn(smp, params)
    loss_fn = torch.jit.script(loss_fn)
    optimizer = BatchAdaptiveGradientDescent(loss_fn, param_range, initial_step=5e-2, 
                                             param_step_factor=torch.tensor([1,1,10,10], device=dev))

    print('true: ',true_params)
    #est,rc,iterations = optimizer.forward(smp.to(dev), params.to(dev), 2000, 100)
    est,rc = optimizer._forward_batch(smp.to(dev), params.to(dev), 20)

    print('err: ', est-true_params.to(dev))
    print('rc:', rc)
    #print(iterations)
    