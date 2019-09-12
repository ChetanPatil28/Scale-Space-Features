import cv2
import numpy as np
from scipy.misc import imresize

anti_alias = 0.5
pre_blur = 1.0
num_octave=4
m_numIntervals=2


img = cv2.imread('unzoom.jpg',0)
img = img.astype(np.float32)/255
print(img.min(),img.max())
img = cv2.GaussianBlur(img,ksize=(0,0),sigmaX =anti_alias)
img = imresize(img,200,'bilinear')
img = cv2.GaussianBlur(img,ksize=(0,0),sigmaX = 1.0)


def build_scale_space(img,num_octave=4,m_numIntervals=2):
    table= np.zeros(shape=(num_octave,m_numIntervals+3),dtype=np.float16)
    init_sigma = np.sqrt(2)
    table[0][0] = init_sigma*0.5
    scale_space = []
    for i in range(num_octave):
        octave = [img]
        curr = img.copy()
        sigma = init_sigma
        for j in range(1,m_numIntervals+3):
            sigma_f = np.sqrt(pow(2.0,2.0/m_numIntervals)-1) * sigma
            print(sigma_f)
            sigma = pow(2.0,1.0/m_numIntervals) * sigma
            table[i][j] = sigma * 0.5 * pow(2.0, i)
            nextt = cv2.GaussianBlur(curr,ksize=(0,0),sigmaX=sigma_f)
            octave.append(nextt)
            curr = nextt
        i_th_octave = np.concatenate([o[:,:,np.newaxis] for o in octave], axis=2)
        scale_space.append(i_th_octave)
        if i<num_octave-1:
            table[i+1][0] = table[i][m_numIntervals];
            img = imresize(img,50,'bilinear')
    return scale_space


ssp = build_scale_space(img)

DoGs=[]
for i in ssp:
    DoGs.append(i[:,:,np.arange(0,m_numIntervals+3-1)]-i[:,:,np.arange(1,m_numIntervals+3)])
