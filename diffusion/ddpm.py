
import torch
from tqdm.auto import tqdm

class DDPM():
    """
    Class: Denoising Diffusion Probabilistic Model
    
    """
    def __init__(self, timesteps, device) -> None:
        self.timesteps = timesteps
        self.device = device

        # Create Linear Beta Schedule
        self.betas = torch.linspace(0.0001, 0.02, self.timesteps)

        # Calculate Alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        """
        Function: 
        Extraction and Reshaping of Data from tensor a based on indices in tensor t
        
        Args:
        a (torch.Tensor)
        t (torch.Tensor)
        x_shape (torch.Tensor)
        
        Return
        out (Torch.Tensor)
        """
        
        # Obtain Batch_Size
        batch_size = t.shape[0]
        
        # Gather Values from a according to indices of t
        out = a.gather(-1, t.cpu())
        
        # Reshape Tensor
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        
        # Move Tensor back to Device and Return
        return out.to(self.device)
    
    def q_sample(self, x_start, t, noise=None):
        """
        q(x_t | x_0) Forward Process 
        
        Args:
            x_start (torch.Tensor)          Starting Image Tensor
            t (torch.Tensor)                Timestep Tensor
            noise (torch.Tensor)            Noise Tensor
        
        Returns:
            x_Noisy (torch.Tensor)          Noisy Image Tensor
        """
        
        # Optional: Generate Gaussian Noise
        if noise is None:
            noise = torch.randn_like(x_start)

        # Extract Alphas & Alphas Cumprod
        self.sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        self.sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        # Compute Noisy Samples:    sqrt(a_hat) * x_0 + sqrt(1 - a_hat) * eps         
        x_Noisy = self.sqrt_alphas_cumprod_t * x_start + self.sqrt_one_minus_alphas_cumprod_t * noise
        return x_Noisy
    
    def p_losses(self, denoise_model, x_start, t, noise=None,):
        '''
        Reverse Process – Noise Prediction
        
        Args:
            denoise_model                          Denoising Neural Network
            x_start (torch.Tensor)                 Starting Image Tensor [B, C, W, H]
            t                                      Timestep Tensor
            noise                                  Noise Tensor [B, C, W, H]
        
        Returns:
            loss                                    Calculated Loss
        
        '''
        
        # Optional: Generate Gaussian Noise
        if noise is None:
            noise = torch.randn_like(x_start)

        # Generate Noisy Samples q(x_t | x_0)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Predict Noise using Neural Network 
        predicted_noise = denoise_model(x_noisy, t)
        
        # Calculate Loss Function
        loss = torch.nn.functional.l1_loss(noise, predicted_noise)
        return loss
    
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


