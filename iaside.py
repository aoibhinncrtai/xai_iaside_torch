import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import functools

def IASIDE_coalition_list(M):
    A = [(0, 1)] * M
    C = list(itertools.product(*A))
    C = sorted(C, key = lambda s: np.sum(s))
    C = [list(s) for s in C]
    return C

def IASIDE_coalition_weights(M):
    C = IASIDE_coalition_list(M - 1)
    P = lambda s: (1.0 / M) * (1.0 / scipy.special.binom(M - 1, s))
    W = [P(sum(s)) for s in C]
    W = np.array(W)
    return W

# The input is in batch, BxCxWxH
def IASIDE_spectral_coalition_ideal_filter_bcwh(x, 
                                          Ci, #coalition_vector
                                          M, 
                                          maskout = "zero", #zero,gaussian,random
                                          device = torch.device("cpu")):

    assert maskout in ["zero", "gaussian", "random"]

    if sum(Ci) == 0:
        return torch.zeros_like(x)

    x = x.to(device)
    
    x_fft2_uncentered = torch.fft.fft2(x, dim = (2, 3))
    x_fft2_centered = torch.fft.fftshift(x_fft2_uncentered, dim = (2, 3))
        
    dim = np.min(x.shape[2:4])
            
    assert dim >= 16
    
    mask = torch.zeros_like(x_fft2_centered)
    
    #look from high-frequency to low-frequency to construct the mask.
    for i, m in enumerate(reversed(Ci)):
        
        i = M - 1 - i       
        step = 1.0 / M
        
        f_high = (i + 1) * step
        f_low = i * step        
        
        if m == 0:
            continue
            
        #Make a low-pass mask using f_high
        r = int(dim * f_high)
        start = (dim - r) // 2
        end = start + r
        mask[:, :, start:end, start:end] = 1
        
        #Make a high-pass mask using f_low
        r = int(dim * f_low)
        start = (dim - r) // 2
        end = start + r
        mask[:, :, start:end, start:end] = 0
    

    if maskout == "zero":
        noise = torch.zeros_like(x_fft2_centered)
    elif maskout == "gaussian":
        #compute the mu and std 
        std = 1./2.
        mu = 0
        noise_real = torch.normal(mu, std, size=x_fft2_centered.size()) 
        noise_img = torch.normal(mu, std, size=x_fft2_centered.size()) 
        noise = torch.complex(noise_real, noise_img)
    elif maskout == "random":
        indecies = torch.randperm(x.shape[0])
        noise = x_fft2_centered[indecies] 

    x_fft2_centered_filtered = x_fft2_centered * mask + noise * (1 - mask)
    #x_fft2_centered_filtered = x_fft2_centered * mask 
            
    x_fft2_centered_filtered_ = torch.fft.ifftshift(x_fft2_centered_filtered, dim = (2, 3))
    x_filtered = torch.fft.ifft2(x_fft2_centered_filtered_, dim = (2, 3))
    
    x_filtered = x_filtered.abs()
    
    x_filtered = x_filtered - torch.min(x_filtered)
    x_filtered = x_filtered / torch.max(x_filtered)
    
    x_filtered = x_filtered.clip(0, 1)
    
    return x_filtered

def IASIDE_compute_spectral_importance_dist(model, 
                                 criterion,
                                 dataset, 
                                 K,                 #num of samples
                                 M,                 #num of spectrum
                                 batch_size = 10,
                                 progressbar = True,
                                 checkpoint_file = None,
                                 seed = 42,
                                 label_dist = None,
                                 maskout = "random",
                                 device = torch.device("cpu")):
    
    
    #device = torch.device("cpu")

    model.to(device)
    model.eval()

    dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size = batch_size, 
                                         shuffle = True,
                                         generator = torch.Generator().manual_seed(seed)
                                        )

    

    W = IASIDE_coalition_weights(M)
    W = np.array(W)

    C = IASIDE_coalition_list(M = M)
    
    if progressbar:
        C_ = tqdm(C)
    else:
        C_ = C

    coalition_index_fn = lambda c: np.array(c).dot(2**np.arange(len(c)))

    U = dict()

    for i, Ci in enumerate(C_):
        
        #Monte Carlo
        num_samples = 0

        worth_list = []

        for x, y in dataloader:
            
            num_samples += batch_size
            
            if num_samples > K:
                break
            
            x = x.float().to(torch.device("cpu"))
            
            x = IASIDE_spectral_coalition_ideal_filter_bcwh(x = x, 
                                                            Ci = Ci, 
                                                            M = M,
                                                            maskout = maskout)

            if torch.isnan(x).any():
                raise ValueError()

            x = x.detach().float().to(device)
            y = y.to(device)
            y_pred = model(x.float())
    
            if torch.isnan(y_pred).any():
                raise ValueError()

            y_prob_ = torch.nn.functional.softmax(y_pred, dim=1)
            y_prob_ = y_prob_.cpu() 
            y_prob = torch.gather(y_prob_, dim = 1, index = y.cpu().long().unsqueeze(1))
            y_prob = y_prob.squeeze().detach() 
            worth = torch.log(y_prob)

            worth = worth.squeeze().detach().cpu().numpy()

            worth = list(worth)
            
            worth_list += worth

        worth = np.mean(worth_list)
        
        Ci = np.array(Ci)
        vidx = coalition_index_fn(Ci)
        U[vidx] = worth
    

    Psi = []
    for i in range(M):
        S_pos = IASIDE_coalition_list(M - 1)
        S_neg = IASIDE_coalition_list(M - 1)

        [s.insert(i, 1) for s in S_pos]
        [s.insert(i, 0) for s in S_neg]

        dU = []
        for s_a, s_b in zip(S_pos, S_neg):
            s_a = list(s_a)
            s_b = list(s_b)
            
            j = coalition_index_fn(s_a)
            k = coalition_index_fn(s_b)
            delta = U[j] - U[k] 
            dU.append(delta)

        dU = np.array(dU)
        Psi_i = np.dot(dU, W)
        Psi.append(Psi_i)

    Psi = np.array(Psi)
    Psi = np.squeeze(Psi)

    Psi = Psi - np.min(Psi)
    Psi = Psi / np.sum(Psi)

    return Psi



def IASIDE_compute_spectral_importance_score(dist, M, beta = 0.75):
    
    b = np.array([beta**k for k in range(M)])
    
    L2norm = lambda x: np.sqrt(np.sum(x**2))
    L1norm = lambda x: np.sum(np.abs(x))
    
    b_norm = b / L2norm(b)
    
    v = np.array(dist)
    v = v / np.sum(v)
    
    
    D = (L1norm(b) / L2norm(b)) * (1 / M)
    
    A = np.sum(b_norm * v) - D
    A = np.abs(A)
    
    B = 1 - D
    B = np.abs(B)

    S = A / B
    
    return S
