# -*- coding: utf-8 -*-
"""
This script contains the FUNCTIONS needed for 'synthetic_feature_maps_main.py'

DESCRIPTION:
    The functions here are used to build sythetic 'truth' and 'model' grids.
    Features are placed in the landscape as square objects using different
    assumptions for how error is structured in the scene. It also includes
    functions to calculate accuracy and other grid metrics.
    
FUNCTION DESCRIPTIONS
    1: 'kernel'           Build objects using scipy.signal.convolve2d
    2: 'object_location'  Iterate over number of objects needed using 'kernel'
    3: 'generate_grid'    Create feature map using 'object_location'
    4: 'model_offset'     Create model grid for translational offset of truth
    5: 'model_rand_err'   Create model grid for random error on truth
    6: 'accuracy_metrics' Calculate TP, FP, FN, TN, F1-score, MCC, nMCC
    7: 'edge_to_area'     Calculate edge to area ratio on given grid

REQUIREMENTS:
    packages: numpy 1.23.5, scipy 1.10.0, matplotlib 3.7.0 
"""

import numpy as np
from scipy import signal as sig
from scipy import ndimage as ndi

## BUILD OBJECTS USING 2D CONVOLUTION OF OBJECT CENTRES FOR GIVEN SIZE ##
def kernel(ksize,arr):
    k = np.ones((ksize, ksize))
    arrc = sig.convolve2d(arr, k, mode='same', boundary='wrap')
    arrc[arrc >= 1] = 1
    
    return arrc

## FEATURE MAP USING OBJECT CENTERS ##
def object_location(leng, frac, seed_no,scale):
    z = np.zeros((leng,leng))
    obj_no = round((leng*leng*frac)/(scale**2), 0)  # initial number of objects
    i=1
    ratio=0.
    
    # while loop iterates on number of objects to match feature fraction
    while ratio < 0.995 or ratio > 1.005:
        if i==1:
            obj_no = obj_no               # first iteration
        else:
            obj_no = round(obj_no*ratio)  # subsequent iterations
        #print(tor_no)
        obj_dens = obj_no/(leng*leng)     # convert to density
        
        z = np.zeros((leng,leng))
        np.random.seed(seed_no)
        z += np.random.rand(leng,leng)
        z[z > (1-obj_dens)] = 1 
        z[z <= (1-obj_dens)] = 0
       
        z = kernel(ksize=scale, arr=z) 
        
        ratio = (frac)/(0.00000000001+np.sum(z)/(leng**2))
        #print(ratio)
        
        if i==50:
            break
        
        i+=1
    
    if ratio < 0.995 or ratio > 1.005:
        print('Did not converge in fifty iterations.')
    
    return z

## GENERATE FEATURE GRID ##
def generate_grid(leng, frac, seed_no, scale):
    # use object location function to generate features
    z = object_location(leng,frac,seed_no,scale) 
    frac_actual = np.size(z[z==1])/(np.size(z))
    
    return [z, frac_actual];
    

## OFFSET INPUT GRID BY DN ##
def model_offset(z,dn):
    [m,n]=z.shape
    zn = np.zeros([m,n])
        
    for i in range(n):
        if i<dn:
            zn[:,i] = z[:,(n-dn+i)]
        else:
            zn[:,i] = z[:,i-dn]
        
    frac_actual = np.size(zn[zn==1])/(np.size(zn))
    
    return [zn, frac_actual]            

## MODIFY INPUT GRID USING A CONSTANT ERROR RATE ##
def model_rand_err(z,err,seed_no):
    [m,n]=z.shape
    zn = np.zeros([m,n])
    ze = np.zeros([m,n])
    
    np.random.seed(seed_no)
    ze += np.random.rand(m,n)
    ze[ze>=err]=1    
    ze[ze<err]=0
    ze=1-ze
     
    #print(ze)   
    for i in range(m):
        for j in range(n):
            if ze[i,j]==1:
                zn[i,j]=z[i,j]+2
            else:
                zn[i,j]=z[i,j]

    zn[zn==2] = 1
    zn[zn==3] = 0
    
    frac_actual = np.size(zn[zn==1])/(np.size(zn))
    
    return [zn, frac_actual]            


## CALCULATE F1 SCORE, MCC, and nMCC ##
def accuracy_metrics(zm,zt):
    # Build Confusion Matrix
    zclass = (zm[:]-zt[:])+(3*zt[:])+1
    TN = float(np.count_nonzero(zclass==1))
    TP = float(np.count_nonzero(zclass==4))
    FN = float(np.count_nonzero(zclass==3))
    FP = float(np.count_nonzero(zclass==2))
    
    # F1-score and MCC
    F1 = (2*TP) / ((2*TP) + FP + FN) 
    MCC = ((TP*TN)-(FP*FN)) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    nMCC = (MCC+1)/2
    
    return [F1, nMCC, zclass, TN, TP, FN, FP];

## CALCULATE EDGE TO AREA RATIO ON FEATURE MAP ##
def edge_to_area(zm):
    ker = np.array(([0,1,0],[1,0,1],[0,1,0]))
    area = np.sum(zm)
    edges = np.sum(zm*(4 - ndi.correlate(zm,ker,mode='wrap')))
    ratio = edges / area
    
    return ratio
