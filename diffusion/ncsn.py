import torch
import tqdm
import numpy as np

'''
Noise Conditional Score Based Algorithms
https://github.com/ermongroup/ncsn/ 

sigmas          sigmas = torch.tensor(np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)
'''

def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
    '''
    Function: 
    Loss Function using Denoising Score Matching Objective
    
    Args:
    scorenet            Neural Network Estimating the Score Function
    samples             Input Samples 
    labels
    sigmas
    anneal_power        Hyperparameter controlling the annealing of the weighting in the loss function.
    '''
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    
    # Add Gaussian Noise According to the used_sigmas
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    
    # Calculate the Target Scores for the DSM Objective
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    
    # Compute the Proedicted scores using the scorenet model
    scores = scorenet(perturbed_samples, labels)
    
     # Flatten the target and predicted scores to facilitate batch processing.
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    
    # Loss Function
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    return loss.mean(dim=0)

def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
    '''
    Function: 
    Sampling using Annelead Langevin Dynamics
    
    Args:
    x_mod (torch.Tensor)        Random Torch Tensor of [b, c, h, w]
                                torch.rand(batches, channels, height, width, device=config.device)
    
    scorenet                    Noise Conditional Score Based Model
    
    sigmas (numpy.ndarray)      Sigmas Schedule adjusting the added Noise Level 
                                np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_classes))
    
    n_steps_each (int)
    
    step_lr (float)             Learing Rate per Step
    '''
    # Initialize an empty list to store images generated at each step
    images = []

    # No Learning -> Disable Gradient Calculation 
    with torch.no_grad():
        # Loop over all Noise Scales
        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
            
            # Create Labels for current 
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            
            # Calculate the step-size for Updates
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                
                # Add Image 
                images.append(torch.clamp(x_mod, 0.0, 1.0).cpu())
                
                # Add Scaled Gaussian Noise
                noise = torch.randn_like(x_mod) * torch.sqrt(step_size * 2)
                
                # Compute the Score Function at the current Model state
                grad = scorenet(x_mod, labels)
                
                # Update Model state
                x_mod = x_mod + step_size * grad + noise
        return images   