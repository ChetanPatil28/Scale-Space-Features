import numpy as np
from numpy.lib import stride_tricks

def get_keypoints(z,l=[0,1,2],size=3):
    h,w = z.shape[:2]
    strs = z[:,:,0].strides
    centre = z[:,:,1]
    keypoint_img = np.zeros_like(centre) 
    views=[]

    for i in l:
        view=stride_tricks.as_strided(z[:,:,i], shape=(h-size+1, w-size+1, size, size), strides=strs*2)
        views.append(view)
    new_view = np.concatenate([o[:,:,:,:,np.newaxis] for o in views], axis=-1).reshape(h-size+1,w-size+1,-1)
    y,x=np.where((new_view.argmax(axis=-1)==13) | (new_view.argmin(axis=-1)==13))
    y+=1
    x+=1    
    points = centre[y,x]
    dyy = centre[y+1,x]+centre[y-1,x] - 2*points
    dxx = centre[y,x+1]+centre[y,x-1] - 2*points
    dxy = (centre[y+1,x+1]+centre[y-1,x-1]-centre[y-1,x+1]-centre[y+1,x-1])/4
    trH = dxx + dyy;
    detH = dxx*dyy - dxy*dxy;
    curvature_ratio = trH*trH/detH;
    filt = np.where(( (curvature_ratio<7.2) &  (detH>0) & (np.abs(points)>0.03)),points,-1.0) #(curvature_ratio<7.2)
    filtered_y,filtered_x = y[filt!=-1],x[filt!=-1]
    keypoint_img[filtered_y,filtered_x]  = 255
    return keypoint_img