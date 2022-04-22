import torch
import numpy as np


def psnr(denoised , ground_truth) :
# Peak Signal to Noise Ratio : denoised and ground ̇truth have range [0 , 1]
    mse = torch.mean((denoised - ground_truth ) ** 2)
    return -10*torch.log10( mse + 10** -8)

def compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate):
    ramp_down_start_iter = iteration_count * (1 - ramp_down_perc)
    if i >= ramp_down_start_iter:
        t = ((i - ramp_down_start_iter) / ramp_down_perc) / iteration_count
        smooth = (0.5+np.cos(t * np.pi)/2)**2
        return learning_rate * smooth
    return learning_rate

