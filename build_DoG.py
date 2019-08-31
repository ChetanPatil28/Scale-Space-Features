import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from numpy.lib import stride_tricks
from scipy.ndimage.filters import convolve 

def Gaussian_kernel(sigma): 
    size = 2*np.ceil(3*sigma)+1 
    neg,pos= -size//2+1,size//2+1
    x,y=np.meshgrid(np.arange(-size//2 + 1,size//2 + 1), np.arange(-size//2 + 1,size//2 + 1))    
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
    return g/g.sum()


def generate_octave(init_level, s, sigma): 
    octave = [init_level] 
    k = 2**(1/s) 
    kernel = gaussian_filter(k * sigma) 
    for _ in range(s+2): 
        next_level = convolve(octave[-1], kernel) 
        octave.append(next_level) 
    return octave