from ddpm import DDPM
import torch
from tqdm.auto import tqdm


class DDIM(DDPM):
    def __init__(self, timesteps, device) -> None:
        super().__init__(timesteps, device)
        
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        '''
        Reverse Process Sample Step 
        
        Args:
            model                               Denoising Neural Network
            x (torch.Tensor)                    x_T Tensor [B, C, W, H]  
            t (torch.Tensor)                    Timestep Tensor
            t_index (int)                       Timestep Index                 
        
        Returns:

        '''
        # Extract Betas and Alphas
        self.betas_t = self.extract(self.betas, t, x.shape)
        self.sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        self.sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
       # Langevin Like Denoising Process -> 
        self.model_mean = self.sqrt_recip_alphas_t * (
            x - self.betas_t * model(x, t) / self.sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return self.model_mean
        else:
            self.posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return self.model_mean + torch.sqrt(self.posterior_variance_t) * noise 

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        '''
        Reverse Process Sample Loop
        
        '''
        device = next(model.parameters()).device

        b = shape[0]
        
        
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
             
            imgs.append(img.cpu())
        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
        


DDIM.extract
    



    
    
    
    