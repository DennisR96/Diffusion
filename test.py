import torch
import numpy as np

z = torch.rand(24, 1, 28, 28, device="mps")

sigmas = np.exp(np.linspace(np.log(0.1), np.log(0.999),
                                    10))
print(sigmas)